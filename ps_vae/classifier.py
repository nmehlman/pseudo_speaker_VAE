import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    
    def __init__(self, input_dim: int, num_classes: dict | int, num_layers: int = 1, hidden_dim: int = 128, activation: str = 'relu'):
        
        super(Classifier, self).__init__()

        # Handle backward compatibility
        self.single_label_mode = isinstance(num_classes, int)
          
        self.num_classes = num_classes  # Dictionary where keys are labels and values are the number of classes per label
        self.layers = nn.ModuleList()
        self.output_layers = nn.ModuleDict()
        
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        activation_fn = activations[activation]
        
        if self.single_label_mode: # Single-label mode
            if num_layers == 1:
                self.layers.append(nn.Linear(input_dim, num_classes))
            else:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                
                for _ in range(num_layers - 2):
                    self.layers.append(activation_fn())
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                
                self.layers.append(activation_fn())
                self.layers.append(nn.Linear(hidden_dim, num_classes))
        else: # Multi-label mode
            if num_layers == 1:
                for label, num_classes in num_classes.items():
                    self.output_layers[label] = nn.Linear(input_dim, num_classes)
            else:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                
                for _ in range(num_layers - 2):
                    self.layers.append(activation_fn())
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                
                self.layers.append(activation_fn())
                
                for label, num_classes in num_classes.items():
                    self.output_layers[label] = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.single_label_mode:  # Single-label mode
            return x
        else:
            outputs = {}
            for label, output_layer in self.output_layers.items():
                outputs[label] = output_layer(x)
            
            # Handle backward compatibility for single-label mode        
            return outputs
    
if __name__ == "__main__":
    # Example usage
    input_dim = 256
    num_classes = {'label1': 10, 'label2': 5} # Example label classes
    num_layers = 2
    hidden_dim = 128
    activation = 'relu'
    model = Classifier(input_dim, num_classes, num_layers, hidden_dim, activation)
    x = torch.randn(32, input_dim)  # Example input
    outputs = model(x)  # Forward pass  
    for label, output in outputs.items():
        print(f"{label}: {output.shape}")  # Output shape for each label