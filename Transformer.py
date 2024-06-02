import torch
from torch import nn
import math

class PositionalEncoding(nn.Module): # From pytorch tutorial on language modeling (with few modifications). Simple pos encoding using sin and cos 
    def __init__(self, d_model, max_len = 50000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position * div_term[0:pe[:,1::2].shape[1]]) 
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return x


class transformermodel(nn.Module):
    def __init__(self, input_size, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout = 0, dimension = None, input_dropout = None):
        super().__init__()
        #dimension = dimension or (input_size//n_heads) # I usually pass a dimension so this line not technically needed
        # dimension * n_heads can be considered equal to embedding size for NLP models     
        self.Xprojection = nn.Linear(input_size, dimension * n_heads) 
        self.yprojection = nn.Linear(1, dimension * n_heads) 
        self.posenc = PositionalEncoding(d_model = dimension*n_heads)

        
        self.transformer = nn.Transformer(d_model = dimension*n_heads, nhead=n_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
    
        self.out_projection = nn.Linear(dimension * n_heads, 1)
        self.dropout = nn.Dropout(p = dropout) # dropout layer. These layers are inactivated when we do model.eval() and activated when we do model.train()
        # It should be okay to only create one instance of dropout and reuse it.
        
        if input_dropout:
             self.in_drop = nn.Dropout(p = input_dropout)
        self.input_dropout = input_dropout # This was added later. Used to determine if dropout should be applied directly to the decoder input
       
    def _generate_square_subsequent_mask(self, size, device):
        return torch.log(torch.tril(torch.ones(size,size, device = device)))

    def forward(self, src, tgt):
        if self.input_dropout:
            tgt = self.in_drop(tgt) 
            
        src = self.Xprojection(src) # Source shape = (Batch size, sequence length, number of features) -> output (batch size, sequence length, dimension*n_heads)
        src = self.dropout(src)
        tgt = self.yprojection(tgt.unsqueeze(2)) # tgt shape = (batch size, seq_length). We unsqueeze it to get (batch size, seq_length, 1). 
                                                 # The output should be (batch size, seq_length, dimension*n_heads)
        tgt = self.dropout(tgt)

        # Positional encoding should be done here
        src = self.posenc(src)
        src = self.dropout(src)
        tgt = self.posenc(tgt)
        tgt = self.dropout(tgt)

        #Masking:
        device = src.device
        mask = self._generate_square_subsequent_mask(tgt.shape[1], device)
        
        # Transformer
        out = self.transformer(src, tgt, tgt_mask = mask)
        out = self.dropout(out)
        out = self.out_projection(out).view(out.shape[0], -1)
        return out


