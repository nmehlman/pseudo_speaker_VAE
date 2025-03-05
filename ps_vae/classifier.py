import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentClassifier(nn.Module):
    
    def __init__(self, latent_dim: int, num_classes: int, num_layers: int = 1, hidden_dim: int = 128, activation: nn.Module = nn.ReLU):
        
        super(LatentClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(nn.Linear(latent_dim, num_classes))
        else:
            self.layers.append(nn.Linear(latent_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.layers.append(activation())
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.layers.append(activation())
            self.layers.append(nn.Linear(hidden_dim, num_classes))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x