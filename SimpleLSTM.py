import torch
from torch import nn
import numpy as np

class simpleLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_LSTM_layers: int, num_fc_layers: int, dropout: float, teacher_forcing_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_LSTM_layers = num_LSTM_layers
        
        if num_LSTM_layers == 1:
            lstm_dropout = 0 
        else:
            lstm_dropout = dropout
        
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_LSTM_layers,
                            dropout=lstm_dropout, batch_first = True) 

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.num_fc_layers = num_fc_layers
        self.decoderlstm = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_LSTM_layers,
                            dropout=lstm_dropout, batch_first = True) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout) 
        
        self.linears = nn.ModuleList() 
        for i in range(self.num_fc_layers):
            if i == self.num_fc_layers - 1: 
                self.linears.append(nn.Linear(hidden_size, output_size))
            else: 
                self.linears.append(nn.Linear(hidden_size, hidden_size))

    
    def forward(self, src, tgt):
        encoder_out, encoder_hidden = self.encoder_lstm(src) 

        outputs = []
        decoder_hidden = encoder_hidden
        target_len = tgt.shape[1] 
        decoder_input = tgt[:,0].view(-1,1,1)
        for i in np.arange(target_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            outputs.append(decoder_output)
            
            
            if self.training and torch.rand(1) < self.teacher_forcing_ratio and i != target_len-1: 
                decoder_input = tgt[:, i+1].view(-1,1,1)
            else:
                decoder_input = decoder_output

        outputs = torch.cat(outputs, dim = 1)

        out = outputs.view(outputs.shape[0], -1) 
        return out
    
    def forward_step(self, decoder_input, hidden): 
        lstm_out, hidden = self.decoderlstm(decoder_input, hidden)
        output = self.relu(lstm_out) 
        output = self.dropout(output)

        for i in range(self.num_fc_layers):
            if i == self.num_fc_layers - 1:
                output = self.linears[i](output) 
            else:
                output = self.linears[i](output)
                output = self.relu(output)
                output = self.dropout(output)

        return output, hidden
