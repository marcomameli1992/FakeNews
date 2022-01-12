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
            nn.Linear(int(linear_input_dimension / 4), int(linear_input_dimension / 4)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 4), int(linear_input_dimension / 8)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 8), int(linear_input_dimension / 16)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 16), int(linear_input_dimension / 32)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 32), int(linear_input_dimension / 64)),
            nn.ReLU(),
            nn.Linear(int(linear_input_dimension / 64), n_classes),
        )

    def forward(self, features):
        classification = self.classification(features)

        return classification