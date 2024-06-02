import optuna
import pandas as pd
import pickle
from data.data_sets import Uloop_dataset, softsensorData
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from models.Transformer import transformermodel
from models.SimpleLSTM import simpleLSTM
from models.Softsensor import EncoderSensor, FFsensor
from optuna.trial import TrialState
import os
from utils.utils import memory

class hyperOpt:
    def __init__(self, which, study_name: str, load = True, seq_length_out = 60, criterion = nn.L1Loss(), device = "CUDA", scaler = "standard", engineering = False, reduce_dataset = False, large_data = False, resamp_time = "300S", use_input_dropout = False):

        if which not in ["Demo", "Pilot"]:
            raise ValueError(which, "is not in ['Demo', 'Pilot']")
        self.which = which

        self.engineering = engineering
        self.reduce_dataset = reduce_dataset
        self.large_data = large_data
        self.resamp_time = resamp_time
        self.study_name = study_name
        self.scaler = scaler
        self.seq_length_out = seq_length_out
        self.device = device
        self.criterion = criterion
        self.use_input_dropout = use_input_dropout

        # Optuna stuff        
        logger = optuna.logging.get_logger("optuna")
        if logger.hasHandlers():
            print("Logger has handlers")
        else:
            logger.addHandler(logging.StreamHandler(sys.stdout))
            
        storage_name = f"sqlite:///D:/Personal/Dan/Scripts/Project/Optuna_studies/{which}/{study_name}.db"

        if load:
            self.study = optuna.create_study(direction="minimize", sampler = optuna.samplers.TPESampler(), study_name=study_name, storage = storage_name, load_if_exists = load)
        else: # This is maybe a lil risky :/
            os.remove(f"D:/Personal/Dan/Scripts/Project/Optuna_studies/{which}/{study_name}.db")
            self.study = optuna.create_study(direction="minimize", sampler = optuna.samplers.TPESampler(), study_name=study_name, storage = storage_name, load_if_exists = load)
    

    def _objective(self, trial):
        torch.manual_seed(42)
        np.random.seed(42)
        n_epochs = 15
        
        # Hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log = True)
        optimizer_name = "Adam" #trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) #"Adam" 
        self.optimizer_name = optimizer_name
        batch_size = trial.suggest_int("batch_size",400, 1200) 
        self.batch_size = batch_size 
        scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.1, 1) # rate of exponential decay of learning rate
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-1, log = True)
        

        seq_length = trial.suggest_int("seq_length", 200, 250)
        self.seq_length = seq_length
        train_dataset = Uloop_dataset(which = self.which, flag = "train", seq_length = seq_length, seq_length_out = self.seq_length_out, reset_df=False, scaler=self.scaler, engineering=self.engineering, reduce_dataset=self.reduce_dataset, large_data = self.large_data, resamp_time=self.resamp_time)
        val_dataset = Uloop_dataset(which = self.which, flag = "val", seq_length = seq_length, seq_length_out = self.seq_length_out, reset_df=True, scaler=self.scaler, engineering=self.engineering, reduce_dataset=self.reduce_dataset, large_data = self.large_data, resamp_time=self.resamp_time) 
        # Creating DataLoaders:
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=12, pin_memory=True) #We can set shuffle = True, since we are returning sequences that are ordered. And it doesn't matter what sequence these sequences come in. 
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=12, pin_memory=True) # We don't shuffle the test/val set.
        
        try: # We will se if it is possible to create this model (maybe it will be too large for the GPU)
            model = self.create_model(trial=trial, input_size = train_dataset.input_size) #more hyperparameters are defined here
            # Creating optimizer
            optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
            
            # Learning rate scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
            
            min_val_loss = None # minimum obtained loss in this training session.
            loss_val = None #current validation los 
            counter = 0 # early stopping counter
            patience = 5 # How many times is the val loss allowed to be worse than min_loss before stopping
            print("    ----Training new model----")
            for epoch in range(n_epochs):
                #train
                if loss_val is not None:
                        print(f"    Finished epoch {epoch} with loss: {loss_val}")
                for X, y in train_loader:
                    model.train()
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    out = model(X,y[:,:-1])
                   
                    loss = self.criterion(out, y[:,1:]) 
                    loss.backward()
                    optimizer.step()
                # test
                loss_val = self._evaluate_epoch(data_loader = val_loader, model = model) 
                scheduler.step()
                trial.report(loss_val, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                if min_val_loss == None or loss_val < min_val_loss: # then model is better than current
                    counter = 0 # reset early stopping counter
                    min_val_loss = loss_val
                else: # then model is worse (or model returns nan due to exploding/disappearing gradients)
                    counter += 1
                    if counter >= patience:
                        break # early stopping
            
            torch.cuda.empty_cache()
            return min_val_loss # We would like to optimize the validation loss.
        except RuntimeError as err: # Usually a Cuda OutOfMemory error. But could be something else if "someone" made a mistake in their model
            print("An error was encountered and this run was penalized.\n The error that caused this was:", err)
            torch.cuda.empty_cache() # I think this should be done to remove the model from CUDA, but not entirely sure what else to do
            return 10 # Penalize this model, since too thicc

    def create_model(self, trial, input_size):
        raise NotImplementedError("You need to overwrite create_model() in a subclass, since models may take different parameters.")
        return None


    def _evaluate_epoch(self, data_loader, model): # not autoregressice for transformer
        num_batches = len(data_loader)
        total_loss = 0
        model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                out = model(X,y[:,:-1]) # We don't want the last one, since that is only used for calculating loss

                total_loss += self.criterion(out, y[:,1:]).item() # Remember our y now contains the last previous value as well, which shouldn't be used for loss 

        avg_loss = total_loss / num_batches

        return avg_loss
        


    def runOpt(self, n_trials=1, stopTime = None):
        if stopTime is not None:
            stop = pd.to_datetime(stopTime)# Change the date to the date you want to end at
            timeout = round((stop - pd.Timestamp.now()).total_seconds())
        else:
            timeout = None

        self.study.sampler = optuna.samplers.TPESampler(seed=42)
        self.study.optimize(self._objective, n_trials = n_trials, timeout=timeout, gc_after_trial=True)

        self.save_params() # We save the best parameters found by the study


    def evalOpt(self):
        pruned_trials = self.study.get_trials(deepcopy = False, states = [TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy = False, states = [TrialState.COMPLETE])

        print("Study statistics: ")
        print("Number of finished trials: ", len(self.study.trials))
        print("Number of pruned trials: ", len(pruned_trials))
        print("Number of complete trials: ", len(complete_trials))

        print("\nBest trial:")
        trial = self.study.best_trial
        print("Value: ", trial.value)
        print("Parameters: ")
        for key, value in trial.params.items():
            print(f"{key}: {value}")

        self.save_params() # In addition to in runOpt() we also save here. This is in case runOpt crashes then I think here is good too.
   
    
    def save_params(self):
        # we also want to save the best parameters
        parameters = self.study.best_params
        
        directory = f"D:\Personal\Dan\Scripts\Project\\trained_models/{self.which}/{self.study_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, "Parameters.pkl"), 'wb') as file:
            pickle.dump(parameters, file)


class transformerOpt(hyperOpt):
    def create_model(self, trial, input_size):
        
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 1,10)#1, 10)
        num_decoder_layers = trial.suggest_int("num_decoder_layers", 1,10)#1, 10)
        dim_feedforward = trial.suggest_int("dim_feedforward", 16,1024)#16, 1024) # 10,512


        n_heads = trial.suggest_int("n_heads", 2,18,2)#2, 12, step = 2)

        if self.use_input_dropout:
            input_dropout = trial.suggest_float("input_dropout", 0.05,0.2) # Minimum of 5% gets dropped out
        else:
            input_dropout = None
        if self.which == "Pilot":
            input_size = 8 if self.scaler == "pca" else input_size 
        elif self.which == "Demo":
            input_size = 8 if self.scaler == "pca" else input_size

        dim_max = 40
        dimension = trial.suggest_int("dimension", 4, dim_max,step = 4)#4, dim_max, step = 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.9)#0.1,0.9) # 0, 0.9 for 300S

        
        model = transformermodel(input_size = input_size, n_heads = n_heads, num_encoder_layers= num_encoder_layers, 
                                    num_decoder_layers= num_decoder_layers, dim_feedforward=dim_feedforward, dropout = dropout, 
                                    dimension=dimension, input_dropout=input_dropout).to(self.device)
        return model
    
    def _evaluate_epoch(self, data_loader, model): # this is inference
        num_batches = len(data_loader)
        total_loss = 0
        model.eval()

        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                outs = y[:,0].unsqueeze(1) # The first value in y is our starting value
                for i in np.arange(y.shape[1]-1): # One prediction pr. value in y - 1, since the first value is y_t-1, so we can calculate the loss. y.shape = (batch_size, seq_length)
                    out = model(X, outs)
                    outs = torch.cat((outs[:,0].unsqueeze(1), out), 1)
                
                total_loss += self.criterion(outs[:,1:], y[:,1:]).item() #The first value in y doesn't count since it is previous value
                # We do outs[:,1:] since the first value in outs is the last known value of the output, it should not be included when calculating the loss

        avg_loss = total_loss / num_batches
        #print(f"Val loss: {avg_loss}")
        return avg_loss # I want to plot the loss :)

class simpleLSTMOpt(hyperOpt):
    def create_model(self, trial, input_size):
        hidden_size = trial.suggest_int("hidden_size", 1, 400)
        num_LSTM_layers = trial.suggest_int("num_LSTM_layers", 1, 15)
        num_fc_layers = trial.suggest_int("num_fc_layers", 1, 15)
        teacher_forcing_ratio = trial.suggest_float("teacher_forcing_ratio", 0.0, 1.0)
        
        dropout = trial.suggest_float("dropout", 0.0, 0.9)

        
        if self.which == "Pilot":
            input_size = 8 if self.scaler == "pca" else input_size 
        elif self.which == "Demo":
            input_size = 8 if self.scaler == "pca" else input_size
        output_size = 1
        model = simpleLSTM(input_size, output_size, hidden_size, num_LSTM_layers=num_LSTM_layers, num_fc_layers = num_fc_layers, dropout = dropout, teacher_forcing_ratio = teacher_forcing_ratio).to(self.device)
    
        return model
    
