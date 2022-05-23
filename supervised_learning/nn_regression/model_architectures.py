"""

model_architectures

Architecture parameters are varied here. To help track, label ## each change.

Nick Tulshibagwale

Updated: 2022-04-20

"""
from torch import nn

class NeuralNetwork_01(nn.Module): 
    # classification
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

class NeuralNetwork_02(nn.Module): 
    # regression
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 50),
            nn.ReLU(),
            nn.Linear(50,1)
        )
        
    def forward(self, x):
        z = self.layers(x)
        z = z.flatten()
        return z # predictions   