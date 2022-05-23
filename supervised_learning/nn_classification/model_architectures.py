"""

model_architectures

Architecture parameters are varied here. To help track, label ## each change.

Nick Tulshibagwale

Updated: 2022-04-07

"""
from torch import nn

class NeuralNetwork_01(nn.Module): 
    
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 10),
            nn.ReLU(),
            nn.Linear(10,num_classes)
        )

    def forward(self, x):
        z = self.layers(x)
        return z # predictions
    