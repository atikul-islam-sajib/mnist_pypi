import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class model(nn.Module):
    """
    Custom artificial neural network (ANN) model for the Iris dataset classification.

    This model consists of two separate fully connected layers (LEFT and RIGHT)
    followed by an output layer that combines their outputs.

    Args:
        None

    Attributes:
        LEFT_LAYER (nn.Sequential): Left fully connected layers.
        RIGHT_LAYER (nn.Sequential): Right fully connected layers.
        OUT_LAYER (nn.Sequential): Output layer for classification.

    Methods:
        left_fully_connected_layer: Create the left fully connected layer.
        right_fully_connected_layer: Create the right fully connected layer.
        output_layer: Create the output layer.
        forward: Forward pass through the model.
        total_trainable_parameters: Calculate and display the total trainable parameters.

    """

    def __init__(self):
        super().__init__()

        # Initialize the left and right fully connected layers
        self.LEFT_LAYER  = self.left_fully_connected_layer(dropout=0.2)
        self.RIGHT_LAYER = self.right_fully_connected_layer(dropout=0.2)

        # Initialize the output layer
        self.OUT_LAYER = self.output_layer()

    def left_fully_connected_layer(self, dropout=None):
        """
        Create the left fully connected layers.

        Args:
            dropout (float): Dropout probability for regularization (default: None).

        Returns:
            nn.Sequential: Left fully connected layers.
        """
        return nn.Sequential(
            nn.Linear(in_features = 784, out_features = 256, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = dropout),

            nn.Linear(in_features = 256, out_features = 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(in_features = 64, out_features = 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def right_fully_connected_layer(self, dropout=None):
        """
        Create the right fully connected layers.

        Args:
            dropout (float): Dropout probability for regularization (default: None).

        Returns:
            nn.Sequential: Right fully connected layers.
        """
        return nn.Sequential(
            nn.Linear(in_features = 784, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(in_features = 128, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def output_layer(self):
        """
        Create the output layer.

        Returns:
            nn.Sequential: Output layer for classification.
        """
        return nn.Sequential(
            nn.Linear(in_features=32 + 64, out_features = 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(in_features=16, out_features = 10),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Model output.
        """
        # Pass input through the left and right fully connected layers
        LEFT = self.LEFT_LAYER(x)
        RIGHT = self.RIGHT_LAYER(x)

        # Concatenate the outputs of the left and right layers
        CONCAT = torch.cat((LEFT, RIGHT), dim=1)

        # Pass the concatenated output through the output layer
        OUTPUT = self.OUT_LAYER(CONCAT)

        return OUTPUT

    def total_trainable_parameters(self, model = None):
        """
        Calculate and display the total number of trainable parameters in the model.

        Args:
            model (nn.Module): The PyTorch model for which to calculate the parameters.

        Returns:
            None
        """
        if model is None:
            raise Exception("Model is not found !")
        else:
            print("\nModel architecture\n".upper())
            print(model.parameters)
            print("\n", "_" * 50, "\n")

            TOTAL_PARAMS = 0
            for layer_name, params in model.named_parameters():
                if params.requires_grad:
                    print("Layer # {} & trainable parameters # {} ".format(layer_name, params.numel()))
                    TOTAL_PARAMS = TOTAL_PARAMS + params.numel()

            print("\n", "_" * 50, '\n')
            print("Total trainable parameters # {} ".format(TOTAL_PARAMS).upper(), '\n\n')


if __name__ == "__main__":
    model = model()
    model.total_trainable_parameters(model = model)
    

class model_train:
    """
            A classifier for the Minist dataset using PyTorch.

            Parameters:
            - model: PyTorch model for classification.
            - train_loader: DataLoader for the training dataset.
            - test_loader: DataLoader for the testing dataset.
            - epochs: Number of training epochs.

            Attributes:
            - train_loader: DataLoader for the training dataset.
            - test_loader: DataLoader for the testing dataset.
            - model: PyTorch model for classification.
            - EPOCHS: Number of training epochs.
            - loss_function: Loss function for training.
            - optimizer: Optimizer for model parameters.

            Methods:
            - start_training():
                Trains the classifier on the provided data and returns training history.

            Usage:
            classifier = IrisClassifier(model, train_loader, test_loader, epochs)
            history = classifier.start_training()
    """
    def __init__(self, model = None, train_loader = None, test_loader = None, epochs = None):
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.model  = model
        self.EPOCHS = epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer     = optim.AdamW(params = model.parameters(), lr = 0.01)
    
    def start_training(self):
        history    = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        TRAIN_LOSS = []
        VAL_LOSS   = []
        TRAIN_ACCURACY = []
        VAL_ACCURACY   = []

        # train the model
        self.model.train()
        # Run a loop with respect to defined Epoch
        for epoch in range(self.EPOCHS):
            """
                1. Extract the data(X_batch), label(y_batch) from the `train_loader`
                2. Pass X_batch as a training data into the model and do the prediction
                3. Compute the Loss Function
                4. Store computed loss into TRAIN_LOSS
            """
            for (X_batch, y_batch) in self.train_loader:
                y_batch = y_batch.long()
                # Do the prediction
                train_prediction = self.model(X_batch)
                # Compute the loss with the predicted and orginal
                train_loss = self.loss_function(train_prediction, y_batch)
                """
                    1. Initiate the Optimizer
                    2. Do the backward propagation with respect to train_loss
                    3. Do the step with optimizer
                """
                # Initialize the optimizer
                self.optimizer.zero_grad()
                # Do back propagation
                train_loss.backward()
                # Do the step with respect to optimizer
                self.optimizer.step()

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
            for (val_data, val_label) in self.test_loader:
                val_label = val_label.long()
                # Do the prediction
                test_prediction = self.model(val_data)
                # Compute the loss
                test_loss = self.loss_function(test_prediction, val_label)

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

            print("Epoch {}/{} ".format(epoch + 1, self.EPOCHS))
            print("{}/{} [=========================] loss: {} - accuracy: {} - val_loss: {} - val_accuracy: {} ".format(self.train_loader.batch_size,\
                                                                                                                        self.train_loader.batch_size,\
                                                                                                                        np.array(train_loss.item()).mean(),
                                                                                                                        accuracy_score(train_predicted, y_batch),\
                                                                                                                        np.array(test_loss.item()).mean(),\
                                                                                                                        accuracy_score(test_predicted, val_label)))
            
        return history
            
            