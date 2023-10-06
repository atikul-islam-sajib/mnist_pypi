import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader

def KFold_CV(model = None, X = None, y = None, epochs = None, fold = None):
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
    
    Kfold = KFold(n_splits = fold, shuffle = True, random_state = 42)

    acc = []
    pre = []
    rec = []
    f1  = []
    
    count = 1

    for train_index, test_index in Kfold.split(X):
        print("CV # {} ".format(count),'\n\n')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = torch.tensor(data = X_train, dtype = torch.float32)
        X_test  = torch.tensor(data = X_test, dtype = torch.float32)

        # y_train = torch.tensor(data = y_train, dtype = torch.float32)
        # y_test  = torch.tensor(data = y_test, dtype = torch.float32)

        train_loader = DataLoader(dataset = list(zip(X_train, y_train)),
                                    batch_size = 16,
                                    shuffle = True)

        test_loader  = DataLoader(dataset = list(zip(X_test, y_test)),
                                    batch_size = 16,
                                    shuffle = True)
        

        EPOCHS = epochs
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        TRAIN_LOSS = []
        VAL_LOSS   = []
        TRAIN_ACCURACY = []
        VAL_ACCURACY   = []

        ########################
        #       Training       #
        ########################

        # train the model
        model.train()
        # Run a loop with respect to defined Epoch
        for epoch in range(EPOCHS):
            """
                1. Extract the data(X_batch), label(y_batch) from the `train_loader`
                2. Pass X_batch as a training data into the model and do the prediction
                3. Compute the Loss Function
                4. Store computed loss into TRAIN_LOSS
            """
            for (X_batch, y_batch) in train_loader:
                # Do the prediction
                train_prediction = model(X_batch)
                # Compute the loss with the predicted and orginal
                train_loss = loss_function(train_prediction, y_batch)
                """
                    1. Initiate the Optimizer
                    2. Do the backward propagation with respect to train_loss
                    3. Do the step with optimizer
                """
                # Initialize the optimizer
                optimizer.zero_grad()
                # Do back propagation
                train_loss.backward()
                # Do the step with respect to optimizer
                optimizer.step()

                ########################
                # Compute the Accuracy #
                ########################

            # Do the prediction of training
            train_predicted = torch.argmax(train_prediction, dim = 1)
            # Append the train accuracy
            TRAIN_ACCURACY.append(accuracy_score(train_predicted, y_batch))
            # Append the train loss
            history['accuracy'].append(accuracy_score(train_predicted, y_batch))
                
            with torch.no_grad():
                # Append the train loss
                TRAIN_LOSS.append(train_loss.item())
                # Append the train loss into the history
                history['loss'].append(train_loss.item())

            ########################
            #       Testing        #
            ########################

            """
                1. Extract the data(val_batch), label(val_batch) from the `test_loader`
                2. Pass val_batch as a training data into the model and do the prediction
                3. Compute the Loss Function
                4. Store computed loss into VAL_LOSS & VAL_ACCURACY
            """
            # Run a loop with respect to test_loader
            for (val_data, val_label) in test_loader:
                # Do the prediction
                test_prediction = model(val_data)
                # Compute the loss
                test_loss = loss_function(test_prediction, val_label)

            ##########################
            #  Compute the Accuracy  #
            ##########################

            # Append the test loss
            with torch.no_grad():
                VAL_LOSS.append(test_loss.item())
                history['val_loss'].append(test_loss.item())
                # Compute the accuracy
                test_predicted = torch.argmax(test_prediction, dim = 1)
                # Append the accuracy of testing data
                VAL_ACCURACY.append(accuracy_score(test_predicted, val_label))
                history['val_accuracy'].append(accuracy_score(test_predicted, val_label))

            #########################
            #        Display        #
            #########################

            # print("Epoch {}/{} ".format(epoch + 1, EPOCHS))
            # print("{}/{} [=========================] loss: {} - accuracy: {} - val_loss: {} - val_accuracy: {} ".format(train_loader.batch_size,\
            #                                                                                                             train_loader.batch_size,\
            #                                                                                                             np.array(train_loss.item()).mean(),
            #                                                                                                             accuracy_score(train_predicted, y_batch),\
            #                                                                                                             np.array(test_loss.item()).mean(),\
            #                                                                                                             accuracy_score(test_predicted, val_label)))
                
            
        predicted = model(X_test)
        predicted = torch.argmax(predicted, dim = 1)
                
        acc.append(accuracy_score(predicted, y_test))
        pre.append(precision_score(predicted, y_test, average = 'macro'))
        rec.append(recall_score(predicted, y_test, average = 'macro'))
        f1.append(f1_score(predicted, y_test, average = 'macro'))

        count = count + 1
    
    print("_"*50, " With KFold - {} ".format(fold), "_"*50) 
    print("ACCURACY # {} ".format(np.array(acc).mean()),'\n')
    print("PRECISON # {} ".format(np.array(pre).mean()),'\n')
    print("RECALL   # {} ".format(np.array(rec).mean()),'\n')
    print("F1_SCORE # {} ".format(np.array(f1).mean()),'\n')