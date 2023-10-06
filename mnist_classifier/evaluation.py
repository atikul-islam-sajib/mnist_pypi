import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings

# Ignore the specific warning about the deprecated softmax dimension choice
warnings.filterwarnings("ignore")

class evaluation:
    def __init__(self, model = None, TRAIN_LOADER = None, TEST_LOADER = None):
        self.ACC = []
        self.PRE = []
        self.REC = []
        self.F1  = []
        self.TD  = []
        
        self.model = model
        self.TRAIN_LOADER = TRAIN_LOADER
        self.TEST_LOADER  = TEST_LOADER
    
    def train_evaluation(self):
        for (train_data, train_label) in self.TRAIN_LOADER:
            self.TD.append(train_data.shape[0])
            model_predicted = self.model(train_data)
            model_predcited = torch.argmax(model_predicted, dim = 1)

            self.ACC.append(accuracy_score(model_predcited, train_label))
            self.PRE.append(precision_score(model_predcited, train_label, average = 'macro'))
            self.REC.append(recall_score(model_predcited, train_label, average = 'macro'))
            self.F1.append(f1_score(model_predcited, train_label, average = 'macro'))

        print("*"*50, " TRAIN RESULTS: for {} data ".format(np.array(self.TD).sum()), "*"*50,'\n\n')

        print("ACCURACY  # {} ".format(np.array(self.ACC).mean(),'\n'))
        print("PRECISION # {} ".format(np.array(self.PRE).mean(),'\n'))
        print("RECALL    # {} ".format(np.array(self.REC).mean(),'\n'))
        print("F1_SCORE  # {} ".format(np.array(self.F1).mean(),'\n'))
    
    
  
    def validation_evaluation(self):
        ACC = []
        PRE = []
        REC = []
        F1  = []
        TD  = []
        for (val_data, val_label) in self.TEST_LOADER:
            TD.append(val_data.shape[0])
            model_predicted = self.model(val_data)
            model_predcited = torch.argmax(model_predicted, dim = 1)

            ACC.append(accuracy_score(model_predcited, val_label))
            PRE.append(precision_score(model_predcited, val_label, average = 'macro'))
            REC.append(recall_score(model_predcited, val_label, average = 'macro'))
            F1.append(f1_score(model_predcited, val_label, average = 'macro'))
        
        warnings.filterwarnings("ignore")
        
        print("\n","*"*50, " TEST RESULTS for {} data: ".format(np.array(TD).sum()), "*"*50,'\n')

        print("ACCURACY  # {} ".format(np.array(ACC).mean(),'\n'))
        print("PRECISION # {} ".format(np.array(PRE).mean(),'\n'))
        print("RECALL    # {} ".format(np.array(REC).mean(),'\n'))
        print("F1_SCORE  # {} ".format(np.array(F1).mean(),'\n'))