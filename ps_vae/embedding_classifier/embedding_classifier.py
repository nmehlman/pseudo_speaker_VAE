import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

class EmbeddingClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for a simple feedforward neural network classifier
    that operates on embeddings.

    Args:
        input_dim (int): The dimensionality of the input embeddings.
        num_classes (int): The number of output classes.
        hidden_dim (int, optional): The dimensionality of the hidden layers. Defaults to 128.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, optimizer_cfg: dict = {}) -> None:
        
        super(EmbeddingClassifier, self).__init__()
        
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.optimizer_cfg = optimizer_cfg

        # Define the neural network layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.num_classes)

        # Define activation function and loss function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
        
    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for a single batch.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_acc', acc, sync_dist=True)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for a single batch.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        return loss