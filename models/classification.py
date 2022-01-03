import torch
import torch.nn as nn

class Classification(nn.Module):

    def __init__(self, linear_input_dimension, n_classes,):
        super(Classification, self).__init__()

        self.classification = nn.Sequential(
            nn.Linear(linear_input_dimension, int(linear_input_dimension / 2)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 2), int(linear_input_dimension / 4)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 4), int(linear_input_dimension / 8)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 8), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, features):
        classification = self.classification(features)

        return classification