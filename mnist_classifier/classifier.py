import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchsummary import summary

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

            nn.Linear(in_features=128, out_features=64, bias=True),
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

            nn.Linear(in_features=16, out_features=9),
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