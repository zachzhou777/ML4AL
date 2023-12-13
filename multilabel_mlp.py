import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class MultilabelMLP(nn.Sequential):
    """Multilayer perceptron for multilabel classification.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    
    hidden_dims : list[int]
        Sizes of hidden layers.
    
    out_dim : int
        Number of outputs.
    
    dropout : float, optional
        The probability for dropout layers.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0
    ):
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        # Do not include sigmoid in model, loss function accepts logits
        super().__init__(*layers)
    
    @classmethod
    def load_model(cls, filepath: str = 'model.pt', dropout: float = 0.0) -> 'MultilabelMLP':
        """Load model from saved state_dict.

        For ease of use, infer model architecture from state_dict.

        Parameters
        ----------
        filepath : str, optional
            Location of model's state_dict.
        
        dropout : float, optional
            The probability for dropout layers. Not saved in state_dict which is why it must be provided here.
        
        Returns
        -------
        MultilabelMLP
            Model with parameters loaded from state_dict.
        """
        state_dict = torch.load(filepath)
        layer_dims = [state_dict['0.weight'].shape[1]] + [param.shape[0] for name, param in state_dict.items() if 'weight' in name]
        model = cls(layer_dims[0], layer_dims[1:-1], layer_dims[-1], dropout)
        model.load_state_dict(state_dict)
        return model
    
    def predict(
        self,
        X: torch.Tensor,
        batch_size: int = 128,
        verbose: bool = True
    ) -> torch.Tensor:
        """Predict label probabilities.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, in_dim)
            Input.
        
        batch_size : int, optional
            Batch size.
        
        verbose : bool, optional
            Whether to print progress bar.
        
        Returns
        -------
        torch.Tensor of shape (n_samples, out_dim)
            Output.
        """
        self.eval()
        with torch.no_grad():
            y = []
            dataloader = DataLoader(X, batch_size=batch_size, shuffle=False)
            for X_batch in tqdm(dataloader, desc='Predicting', disable=not verbose):
                y_batch = F.sigmoid(self(X_batch))
                y.append(y_batch)
            y = torch.cat(y)
        self.train()
        return y
    
    def train_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_dev: torch.Tensor,
        y_dev: torch.Tensor,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        tolerance: float = 1e-4,
        filepath: str = 'model.pt',
        verbose: bool = True
    ):
        """Train model using early stopping.

        Parameters
        ----------
        X_train : torch.Tensor of shape (train_size, in_dim)
            Training set inputs.
        
        y_train : torch.Tensor of shape (train_size, out_dim)
            Training set targets.
        
        X_dev : torch.Tensor of shape (dev_size, in_dim)
            Dev set inputs.
        
        y_dev : torch.Tensor of shape (dev_size, out_dim)
            Dev set targets.
        
        batch_size : int, optional
            Batch size.
        
        max_epochs : int, optional
            Maximum number of epochs.
        
        patience : int, optional
            Number of epochs to wait for improvement before early stopping.
        
        tolerance : float, optional
            Minimum amount dev loss must decrease to be considered as an improvement.
        
        filepath : str, optional
            Location to save model's state_dict. Save whenever dev loss improves.
        
        verbose : bool, optional
            Whether to print progress bars, train and dev losses after each epoch, and early stopping message.
        """
        optimizer = torch.optim.Adam(self.parameters())
        best_dev_loss = float('inf')
        for epoch in range(max_epochs):
            # Iterate over batches
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_loss = 0.0  # Rolling sum
            for X_batch, y_batch in tqdm(dataloader, desc=f'Training (epoch {epoch+1}/{max_epochs})', disable=not verbose):
                # Forward pass
                logits = self(X_batch)
                batch_loss = F.binary_cross_entropy_with_logits(logits, y_batch)
                # Backprop and update parameters
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # Update training loss
                train_loss += batch_loss.item() * X_batch.shape[0]
            train_loss /= X_train.shape[0]
            # Evaluate dev set, break if patience runs out
            dev_loss = self.evaluate_model(X_dev, y_dev, batch_size, verbose)
            if verbose:
                print(f'Avg train loss: {train_loss}, dev loss: {dev_loss}')
            if dev_loss <= best_dev_loss - tolerance:
                no_improvement_count = 0
                best_dev_loss = dev_loss
                torch.save(self.state_dict(), filepath)
            else:
                no_improvement_count += 1
            if no_improvement_count >= patience:
                if verbose:
                    print('Early stopping')
                break
        self.load_state_dict(torch.load(filepath))
    
    def evaluate_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 128,
        verbose: bool = True
    ) -> float:
        """Evaluate loss on samples.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, in_dim)
            Inputs.
        
        y : torch.Tensor of shape (n_samples, out_dim)
            Targets.
        
        batch_size : int, optional
            Batch size.
        
        verbose : bool, optional
            Whether to print progress bar.
        
        Returns
        -------
        float
            Loss.
        """
        self.eval()
        with torch.no_grad():
            total_loss = 0.0
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for X_batch, y_batch in tqdm(dataloader, desc='Evaluating', disable=not verbose):
                logits = self(X_batch)
                loss = F.binary_cross_entropy_with_logits(logits, y_batch, reduction='sum')
                total_loss += loss.item()
            loss = total_loss / (y.shape[0] * y.shape[1])
        self.train()
        return loss
