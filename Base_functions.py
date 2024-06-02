import time
from utils.utils import showPlot, timeSince
from models.Transformer import transformermodel, transformermodel_old
from models.SimpleLSTM import simpleLSTM
import os
import torch
import torch.nn as nn
import pickle
from data.data_sets import Uloop_dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import pandas as pd

class Experiment: 
    def __init__(self, name, engineering: bool, reduce_dataset: bool, which: str, load_model = False, device = "CUDA", seq_length_out = 60, criterion = nn.L1Loss(), scaler = "standard", large_data = False, resamp_time="300S", archive = False, use_input_dropout = False, target = "C_NO3"):
        '''
           scaler in ["standard", "asinh", "symlog", "pca", "tanh", etc.]
           which in ['Demo', 'Pilot']
        '''
        if which not in ["Demo", "Pilot"]:
            raise ValueError(which, "is not in ['Demo', 'Pilot']")
        self.which = which
        self.name = name
        self.target = target # Which variable to set as target
        torch.manual_seed(42) # now this is definitely 1000% reproducible no cap 4 real
        np.random.seed(42)
        self.device = device
        self.load_path = f"D:/Personal/Dan/Scripts/Project/trained_models/{which}/{name}" 
        if archive:
            self.load_path = f"D:/Personal/Dan/Scripts/Project/archive/trained_models/{which}/{name}" 
        with open(os.path.join(self.load_path, "Parameters.pkl"), 'rb') as file: # Should be a check to see if file exists
            self.parameters = pickle.load(file)
        self.archive = archive
        self.use_input_dropout = use_input_dropout
        # Hyperparameters
        self.criterion = criterion #nn.MSELoss() or nn.L1Loss()

        self.learning_rate = self.parameters["learning_rate"]
        try:
            self.optimizer_name = self.parameters["optimizer"]
        except:
            self.optimizer_name = "Adam"
        self.scheduler_gamma = self.parameters["scheduler_gamma"]
        self.weight_decay = self.parameters["weight_decay"] # L2 regularization 0 if nothing

        self.seq_length = self.parameters["seq_length"] #Can also be called look-back window
        self.seq_length_out = seq_length_out
        self.batch_size = self.parameters["batch_size"]

        self.scaler = scaler # used for create_model()

        self.train_dataset = Uloop_dataset(which = which, flag = "train", seq_length = self.seq_length, seq_length_out = seq_length_out, reset_df=False, scaler=scaler, engineering=engineering, reduce_dataset=reduce_dataset, large_data = large_data, resamp_time=resamp_time, target=self.target) 
        self.val_dataset = Uloop_dataset(which = which, flag = "val", seq_length = self.seq_length, seq_length_out = seq_length_out, reset_df= False, scaler=scaler, engineering=engineering, reduce_dataset=reduce_dataset, large_data=large_data, resamp_time=resamp_time, target=self.target) 
        self.test_dataset = Uloop_dataset(which = which, flag = "test", seq_length = self.seq_length, seq_length_out = seq_length_out, reset_df= True, scaler=scaler, engineering=engineering, reduce_dataset=reduce_dataset, large_data=large_data, resamp_time=resamp_time, target=self.target)

        self.load_model = load_model
        self.create_model() 
        
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers=12, pin_memory=True) #We can set shuffle = True, since we are returning sequences that are ordered. And it doesn't matter what sequence these sequences come in. 
        self.val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, num_workers=12, pin_memory=True) # We don't shuffle the test/val set.
        self.test_loader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False, num_workers=12, pin_memory=True) # We don't shuffle the test/val set.
        self.df_preds = None 
        self.preds_out = None

        self.colors = {"LSTM": "#EF476F", "LSTM (pca)": "#F78C6B", "Transformer": "#FFD166", "Transformer (pca)": "#83D483","Transformer (pca MSE)": "#073B4C", "Transformer (reduced)": "#06D6A0", "Naïve": "#118AB2"} 
        
    
    def create_model(self):
        raise NotImplementedError("You need to overwrite create_model() in a subclass, since models may take different parameters.")
        return None
    
    def train_epoch(self,data_loader, optimizer): # One epoch of training
        num_batches = len(data_loader)
        total_loss = 0
        self.model.train()

        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            out = self.model(X,y[:,:-1])
            
            loss = self.criterion(out, y[:,1:]) 
            
            loss.backward()
            #nn.utils.clip_grad_norm_(self.model.parameters(), 1) # To combat exploding gradients. # Mostly made no difference
        
            optimizer.step()

            
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss # I want to plot the loss :)


    # Looks a lot like the train epoch function. 
    def evaluate_epoch(self, data_loader):
        num_batches = len(data_loader)
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                out = self.model(X,y[:,:-1]) # We don't want the last one, since that is only used for calculating loss

                total_loss += self.criterion(out, y[:,1:]).item() # Remember our y now contains the last previous value as well, which shouldn't be used for loss 

        avg_loss = total_loss / num_batches
        return avg_loss # I want to plot the loss :)

    # Inference is much slower than evaluate, since it is autoregressive
    def inference_epoch(self, data_loader):
        num_batches = len(data_loader)
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                outs = y[:,0].unsqueeze(1) # The first value in y is our starting value
                for i in np.arange(y.shape[1]-1):
                    out = self.model(X, outs)
                    outs = torch.cat((outs[:,0].unsqueeze(1), out), 1)
                
                total_loss += self.criterion(outs[:,1:], y[:,1:]).item()

        avg_loss = total_loss / num_batches
        #print(f"Val loss: {avg_loss}")
        return avg_loss # I want to plot the loss :)


    def train_loop(self, n_epochs, patience = 5, print_every = 1, plot_every = 1, endTime = None, save_model = False, global_min_loss = None): #min_loss is minimum obtained loss
        start = time.time() # To measure how long training takes, so I can plan my time
        if endTime is not None:
            print("Warning: Training has been set to end at:", endTime, ". This may stop training early.")
        plot_losses_train = []
        print_loss_total_train = 0 # Should be reset every print_every
        plot_loss_total_train = 0 # Should be reset every plot_every
        plot_losses_val = []
        print_loss_total_val = 0 # Should be reset every print_every
        plot_loss_total_val = 0 # Should be reset every plot_every

        optimizer = getattr(torch.optim, self.optimizer_name)(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        
        # Learning rate scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        
        min_val_loss = None # minimum obtained loss in this training session. 
        counter = 0 # early stopping counter

        for epoch in np.arange(n_epochs):
            if epoch == 0: # To see how model develops
                torch.save(self.model.state_dict(), os.path.join(self.load_path, f"model_start.pth"))

            loss = self.train_epoch(data_loader = self.train_loader, optimizer=optimizer)
            if "Transformer" in self.name:
                loss_val = self.inference_epoch(self.val_loader)
            else:
                loss_val = self.evaluate_epoch(data_loader = self.val_loader)
            
            # Plotting below
            print_loss_total_train += loss
            plot_loss_total_train += loss
            print_loss_total_val += loss_val
            plot_loss_total_val += loss_val

            if (epoch +1) % print_every == 0:
                print_loss_avg_train = print_loss_total_train / print_every
                print_loss_total_train = 0
                print_loss_avg_val = print_loss_total_val / print_every
                print_loss_total_val = 0
                print("%s || Epoch: %d out of %d || Train loss: %.4f || Val loss: %.6f" % (timeSince(start, (epoch+1) / n_epochs), epoch+1, n_epochs, print_loss_avg_train, print_loss_avg_val))
            
            if (epoch +1) % plot_every == 0: # If plot every is not 1, then the plots are not right currently
                plot_loss_avg_train = plot_loss_total_train / plot_every
                plot_losses_train.append(plot_loss_avg_train)
                plot_loss_total_train = 0
                plot_loss_avg_val = plot_loss_total_val / plot_every
                plot_losses_val.append(plot_loss_avg_val)
                plot_loss_total_val = 0

            # Early stopping and save model
            if min_val_loss == None or loss_val < min_val_loss: # then model is better than current
                counter = 0 # reset early stopping counter
                min_val_loss = loss_val
                # save model
                if save_model:
                    if global_min_loss == None or min_val_loss < global_min_loss: # min_loss is global lowest val. So if fit has been run multiple times. Usually just None
                        if not os.path.exists(self.load_path):
                            os.makedirs(self.load_path)
                        torch.save(self.model.state_dict(), os.path.join(self.load_path, "model.pth"))
                        df_res = pd.DataFrame({"Train": plot_losses_train, "Val": plot_losses_val})
                        df_res.to_csv(f"trained_models/{self.which}/{self.name}/Training.csv", index = False) # Save if I wanna plot later
                        global_min_loss = min_val_loss
            else: # then model is worse (or model returns nan due to exploding/disappearing gradients)
                counter += 1
                if counter >= patience:
                    print(f"Validation loss increased {patience} times in a row. Training stopped early.")
                    break # early stopping
            
            # This part is to continually save models:
            if (epoch+1) % 50 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.load_path, f"model_{epoch+1}.pth"))
                df_res = pd.DataFrame({"Train": plot_losses_train, "Val": plot_losses_val})
                df_res.to_csv(f"trained_models/{self.which}/{self.name}/Training.csv", index = False) # Save if I wanna plot later

            scheduler.step()
            

            if endTime is not None:
                if time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) > endTime:
                    print("Stopped training early since endTime was reached")
                    break # We don't want to keep looping if maxTime has been passed

        
        showPlot(plot_losses_train, plot_losses_val) # plot the vals
        df_res = pd.DataFrame({"Train": plot_losses_train, "Val": plot_losses_val})
        df_res.to_csv(f"trained_models/{self.which}/{self.name}/Training.csv") # Save if I wanna plot later
        return (plot_losses_train, plot_losses_val, min_val_loss) # Return so I can do more advanced plotting with them later

    def save_train(self, losses_train, losses_val, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.load_path, f"model_{epoch+1}.pth"))
        df_res = pd.DataFrame({"Train": losses_train, "Val": losses_val})
        df_res.to_csv(f"trained_models/{self.which}/{self.name}/Training.csv", index = False) # Save if I wanna plot later

    def calc_preds(self):
        if self.preds_out is None: # This part may take extremely long to calculate, so this way we only have to do it once, if the function is called multiple times
            preds = []
            with torch.no_grad(): # We calculate all the predicted values
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.model.eval()
                    
                    outs = y[:,0].unsqueeze(1)
                    for i in np.arange(y.shape[1]-1): # One prediction pr. value in y, so we can calculate the loss. y.shape = (batch_size, seq_length)
                        out = self.model(X, outs)
                        outs = torch.cat((outs[:,0].unsqueeze(1), out), 1)
                    
                    preds.append(outs[:,1:].detach().cpu())

            preds_out = [] # I think this part is the one that could be sped up, cause this is probably exceedingly slow.
            for b in preds: # I arrange them in preds_out so that preds_out becomes 2D. It contains the sequence number ie 0, 1, 2 ... to total amount of sequences
                for i in np.arange(b.shape[0]):
                    preds_out.append(b[i,:])
            
            self.preds_out = preds_out
        
        return self.preds_out
            
    def plot_interactive(self): 
        preds = np.array(self.calc_preds())
        times = self.val_dataset.datalist[0].index # I think
        actual = self.val_dataset.datalist[0] # I think
        _, actual = self.val_dataset.scaler.transform(actual,actual[self.target])
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times, actual, label = "Validation data", color = "black")
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m-%d"))
        ax.set_xlabel("Time (month-day)")
        ax.set_ylabel("Value (standardized)")
        ax.yaxis.grid()
        axis_position = plt.axes([0.2,0.0,0.65,0.03], facecolor = "White")
        slider = mpl.widgets.Slider(axis_position, "", 0, preds.shape[0]-1, valinit =0, valstep = 1)

        time0 = times[0+self.seq_length: 0+self.seq_length+self.seq_length_out]
        pred_line, = ax.plot(time0, preds[0], "-o", label = "preds")
        ax.legend(loc = "upper right")

        def update(val):
            i = slider.val
            time = times[i+self.seq_length: i+self.seq_length+self.seq_length_out]
            pred_line.set_xdata(time)
            pred_line.set_ydata(preds[i])


        slider.on_changed(update)
        
        def start(event): # Could not figure out ;(
            print("button clicked")
            anim.event_source.start()
        #play_ax =  plt.axes([0.15,0.75,0.1,0.1], facecolor = "White")
        #play_button = mpl.widgets.Button(play_ax, ">", color = "green")
        #play_button.on_clicked(start)
        #play_ax._button = play_button
        #anim = FuncAnimation(fig, update,frames= preds.shape[0]-1, interval = 100)
        #play_ax._anim = anim
        plt.show()

    # This function doesn't make sense for LSTM. Should only be used for transformer
    def plot_one(self, flag):
        if flag == "train":
            X, y = next(iter(self.train_loader)) # This will give a random value
        elif flag == "val":
            X, y =self.val_dataset[len(self.val_dataset)-1]
            X = torch.tensor(X).unsqueeze(0)
            y = torch.tensor(y).unsqueeze(0)
        elif flag == "test":
            X, y =self.test_dataset[len(self.test_dataset)-1]
            X = torch.tensor(X).unsqueeze(0)
            y = torch.tensor(y).unsqueeze(0)
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            self.model.eval()

            # First inference:
            outs = y[:,0].unsqueeze(1) # Our beginning values can be found here

            for i in np.arange(y.shape[1]-1): # One prediction pr. value in y, so we can calculate the loss. y.shape = (batch_size, seq_length)
                out = self.model(X, outs)
                outs = torch.cat((outs[:,0].unsqueeze(1), out), 1)
            outs = outs[:,1:]
            

            # Then I do simple step by step/ validation loop
            out = self.model(X,y[:,:-1])

            x_vals = np.arange(1,out.shape[1]+1)
            # Then we plot
            fig, ax = plt.subplots() 
            ax.plot(x_vals,y[0,1:].cpu(), "-o", label = "Target data", color = "black")
            ax.plot(x_vals,out[0,:].detach().cpu(), "-o", label = "Using true target", color = self.colors["Transformer"]) # Yellow
            ax.plot(x_vals,outs[0,:].detach().cpu(), "-o", label = "Autoregressive", color = self.colors["Transformer (reduced)"]) # green
            ax.grid(True)
            ax.set_xlabel("Prediction steps")
            ax.set_ylabel("Value (standardized)")
            plt.legend(loc = "upper right")
            plt.show()

    def plot_series(self, xlim = ("2023-06-21", "2023-06-22"), ylim= (-0.25, 1.5)):
        if self.df_preds is None: # This part may take extremely long to calculate, so this way we only have to do it once, if the function is called multiple times
            preds = []
            ys = []
            with torch.no_grad(): # We calculate all the predicted values
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.model.eval()
                    
                    outs = y[:,0].unsqueeze(1)
                    for i in np.arange(y.shape[1]-1): # One prediction pr. value in y, so we can calculate the loss. y.shape = (batch_size, seq_length)
                        out = self.model(X, outs)
                        outs = torch.cat((outs[:,0].unsqueeze(1), out), 1)
                    
                    preds.append(outs[:,1:].detach().cpu())
                    ys.append(y[:,1:].detach().cpu())

            preds_out = []
            for b in preds: # I arrange them in preds_out so that preds_out becomes 2D. It contains the sequence number ie 0, 1, 2 ... to total amount of sequences
                for i in np.arange(b.shape[0]):
                    preds_out.append(b[i,:])

            preds_ar = np.array(preds_out)[np.arange(0, len(preds_out), self.seq_length_out)] # Here I pick every seq_length_out prediction. This will give a continuous curve which can be plotted.
            preds_ar = np.reshape(preds_ar, (-1)) # Here we stack them all in one columns
            self.df_preds = pd.DataFrame(preds_ar, index = Uloop_dataset.val[0].index[self.seq_length:preds_ar.shape[0]+self.seq_length], columns = ["Preds"]) # Turn it into a pd DataFrame

            ys_out = []
            for b in ys: # I arrange them in preds_out so that preds_out becomes 2D. It contains the sequence number ie 0, 1, 2 ... to total amount of sequences
                for i in np.arange(b.shape[0]):
                    ys_out.append(b[i,:])
            
            ys_ar = np.array(ys_out)[np.arange(0, len(ys_out), self.seq_length_out)] # Here I pick every seq_length_out prediction. This will give a continuous curve which can be plotted.
            ys_ar = np.reshape(ys_ar, (-1)) # Here we stack them all in one columns
            self.df_ys = pd.DataFrame(ys_ar, index = Uloop_dataset.val[0].index[self.seq_length:ys_ar.shape[0]+self.seq_length], columns = ["ys"]) # Turn it into a pd DataFrame

        # start_times causes errors
        #start_times = pd.date_range(self.df_preds["Preds"].index[0], self.df_preds["Preds"].index[-1], freq = str(self.seq_length_out)+"min")
        start_times = self.df_preds.index[np.arange(0, len(self.df_preds), self.seq_length_out)]
        start_vals = self.df_preds["Preds"][start_times]
        
        fig, ax = plt.subplots(figsize = [14, 4.8])

        #ax.plot(train["C_NO3"], label = "Training")
        #ax.plot(val[0]["C_NO3"], label = "Validation", color = "black")
        #ax.plot(Uloop_dataset.val[0]["C_NO3"], label = "Validation", color = "black")
        ax.plot(self.df_ys["ys"], label = "Validation", color = "black")
        ax.scatter(self.df_preds.index, self.df_preds["Preds"], label = "Predictions", s = 1)
        ax.scatter(start_times, start_vals, color = "midnightblue", s = 2)
        plt.legend(loc = "upper right")
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
        ax.set_ylim(ylim[0],ylim[1])
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m-%d %H"))
        ax.yaxis.grid(True)
        plt.xticks(rotation = 90)
        #ax.set_xticklabels(labels = ax.get_xticks(), rotation = 90)
        plt.show()

class TransformerExperiment(Experiment):
    """This class is made to do training and testing of transporter models."""
    def create_model(self):
        if self.which == "Pilot":
            input_size = 8 if self.scaler == "pca" else self.train_dataset.input_size # Here we assume large_data also use 8 PC, maybe change
        elif self.which == "Demo":
            input_size = 8 if self.scaler == "pca" else self.train_dataset.input_size # Actually the pca scaler only supports 8 :/
        
        if self.use_input_dropout: # Only available in some cases
            input_dropout = self.parameters["input_dropout"]
        else:
            input_dropout = None

        if self.archive:
            self.model = transformermodel_old(input_size = input_size, n_heads = self.parameters["n_heads"], num_encoder_layers= self.parameters["num_encoder_layers"], num_decoder_layers= self.parameters["num_decoder_layers"], dim_feedforward=self.parameters["dim_feedforward"], dropout = self.parameters["dropout"], dimension=self.parameters["dimension"]).to(self.device)
        else:
            self.model = transformermodel(input_size = input_size, n_heads = self.parameters["n_heads"], num_encoder_layers= self.parameters["num_encoder_layers"], num_decoder_layers= self.parameters["num_decoder_layers"], dim_feedforward=self.parameters["dim_feedforward"], dropout = self.parameters["dropout"], dimension=self.parameters["dimension"],input_dropout=input_dropout).to(self.device)
        # below loads the model
        if self.load_model:
            self.model.load_state_dict(torch.load(os.path.join(self.load_path, "model.pth")))


class LSTMExperiment(Experiment):
    """This class is made to do training and testing of simple LSTM-based seq2seq models."""
    def create_model(self):
        if self.which == "Pilot":
            input_size = 8 if self.scaler == "pca" else self.train_dataset.input_size 
        elif self.which == "Demo":
            input_size = 9 if self.scaler == "pca" else self.train_dataset.input_size
        output_size = 1
        self.model = simpleLSTM(input_size, output_size, hidden_size = self.parameters["hidden_size"], num_LSTM_layers=self.parameters["num_LSTM_layers"], num_fc_layers = self.parameters["num_fc_layers"], dropout = self.parameters["dropout"], teacher_forcing_ratio = self.parameters["teacher_forcing_ratio"]).to(self.device)
        
        if self.load_model:
            self.model.load_state_dict(torch.load(os.path.join(self.load_path, "model.pth")))

    def plot_one(self):
        X, y = next(iter(self.val_loader))
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            self.model.eval()

            # Only this is needed for the LSTM, unless we add 100% teacher forcing to it, but feel like that doesn't make sense
            outs = self.model(X,y[:,:-1])

            x_vals = np.arange(1,outs.shape[1]+1)
            # Then we plot
            fig, ax = plt.subplots()
            ax.plot(x_vals,y[0,1:].cpu(), "-o", label = "Actual")
            #ax.plot(x_vals,out[0,:].detach().cpu(), "-o", label = "Predicted")#Not applicable for LSTM
            ax.plot(x_vals,outs[0,:].detach().cpu(), "-o", label = "Predicted Autoregressive")
            ax.grid(True)
            ax.set_xlabel("Prediction steps")
            ax.set_ylabel("Value (standardized)")
            plt.legend()
            plt.show()

    def plot_series(self, xlim = ("2023-12-05 01:00:00", "2023-12-12 08:00:00"), ylim= (-0.25, 1.5)):
        if self.df_preds is None: # This part may take extremely long to calculate, so this way we only have to do it once, if the function is called multiple times
            preds = []
            ys = []
            with torch.no_grad(): # We calculate all the predicted values
                self.model.eval()
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    
                    
                    outs = self.model(X,y[:,:-1])
                    preds.append(outs[:,1:].detach().cpu())
                    ys.append(y[:,1:].detach().cpu())

            preds_out = []
            for b in preds: # I arrange them in preds_out so that preds_out becomes 2D. It contains the sequence number ie 0, 1, 2 ... to total amount of sequences
                for i in np.arange(b.shape[0]):
                    preds_out.append(b[i,:])

            preds_ar = np.array(preds_out)[np.arange(0, len(preds_out), self.seq_length_out)] # Here I pick every seq_length_out prediction. This will give a continuous curve which can be plotted.
            preds_ar = np.reshape(preds_ar, (-1)) # Here we stack them all in one columns
            self.df_preds = pd.DataFrame(preds_ar, index = Uloop_dataset.val[0].index[self.seq_length:preds_ar.shape[0]+self.seq_length], columns = ["Preds"]) # Turn it into a pd DataFrame

            ys_out = []
            for b in ys: # I arrange them in preds_out so that preds_out becomes 2D. It contains the sequence number ie 0, 1, 2 ... to total amount of sequences
                for i in np.arange(b.shape[0]):
                    ys_out.append(b[i,:])
            
            ys_ar = np.array(ys_out)[np.arange(0, len(ys_out), self.seq_length_out)] # Here I pick every seq_length_out prediction. This will give a continuous curve which can be plotted.
            ys_ar = np.reshape(ys_ar, (-1)) # Here we stack them all in one columns
            self.df_ys = pd.DataFrame(ys_ar, index = Uloop_dataset.val[0].index[self.seq_length:ys_ar.shape[0]+self.seq_length], columns = ["ys"]) # Turn it into a pd DataFrame

        # start_times causes errors
        #start_times = pd.date_range(self.df_preds["Preds"].index[0], self.df_preds["Preds"].index[-1], freq = str(self.seq_length_out)+"min")
        start_times = self.df_preds.index[np.arange(0, len(self.df_preds), self.seq_length_out)]
        start_vals = self.df_preds["Preds"][start_times]
        
        fig, ax = plt.subplots(figsize = [14, 4.8])

        #ax.plot(train["C_NO3"], label = "Training")
        #ax.plot(val[0]["C_NO3"], label = "Validation", color = "black")
        #ax.plot(Uloop_dataset.val[0]["C_NO3"], label = "Validation", color = "black")
        ax.plot(self.df_ys["ys"], label = "Validation", color = "black")
        ax.scatter(self.df_preds.index, self.df_preds["Preds"], label = "Predictions", s = 1)
        ax.scatter(start_times, start_vals, color = "midnightblue", s = 2)
        plt.legend(loc = "upper right")
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))
        ax.set_ylim(ylim[0],ylim[1])
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m-%d %H"))
        ax.yaxis.grid(True)
        plt.xticks(rotation = 90)
        #ax.set_xticklabels(labels = ax.get_xticks(), rotation = 90)
        plt.show()