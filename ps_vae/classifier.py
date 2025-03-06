import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    
    def __init__(self, input_dim: int, num_classes: int, num_layers: int = 1, hidden_dim: int = 128, activation: str = 'relu'):
        
        super(Classifier, self).__init__()
        
        self.layers = nn.ModuleList()
        
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        activation_fn = activations[activation]
        
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, num_classes))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.layers.append(activation_fn())
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.layers.append(activation_fn())
            self.layers.append(nn.Linear(hidden_dim, num_classes))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x