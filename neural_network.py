from typing import Callable, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import trange

class PositiveLinear(nn.Module):
    """A linear layer where the weights are positive.

    Source: https://discuss.pytorch.org/t/positive-weights/19701/7
    """
    def __init__(self, in_features: int, out_features: int):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, input):
        return F.linear(input, self.log_weight.exp(), self.bias)

class ModifiedSigmoid(nn.Module):
    """A function that returns sigmoid(z) for z > 0, 0.25 * z + 0.5
    otherwise.

    The MIP formulation implements a piecewise linear approximation of
    this function.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(z > 0, torch.sigmoid(z), 0.25 * z + 0.5)

class MLP(nn.Sequential):
    """Multilayer perceptron for predicting an ambulance location
    performance metric.

    The model takes an allocation of ambulances to facilities as input,
    and produces a single metric.

    All hidden layers except the last use the ReLU activation function.
    If the MLP predicts coverage, then the final hidden layer uses the
    ModifiedSigmoid activation function. Otherwise if the MLP predicts
    average response time, then the final hidden layer uses the ReLU
    activation function.

    Parameters
    ----------
    in_dim : int
        Number of input features (facilities).
    
    hidden_dims : list[int]
        Sizes of hidden layers.
    
    final_hidden_activation : nn.Module, optional
        Activation function for the final hidden layer. Defaults to ReLU.
    
    positive_final_weights : bool, optional
        Whether to use positive weights for the final linear layer.
    
    dropout : float, optional
        Dropout probability.
    
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        Loss function to apply to predictions and targets.
    
    optimizer : type[torch.optim.Optimizer], optional
        Optimizer to use.
    
    optimizer_params : dict, optional
        Parameters (excluding model parameters) to pass to optimizer
        (e.g., {'lr': 1e-5, 'weight_decay': 1e-3}).
    
    lr_scheduler : type[torch.optim.lr_scheduler.LRScheduler], optional
        Learning rate scheduler to use. If None, no scheduler is used.

        We only support updating the learning rate at the end of each
        epoch (so CyclicLR and OneCycleLR should not be used).
    
    lr_scheduler_params : dict, optional
        Parameters (excluding optimizer) to pass to learning rate
        scheduler (e.g., {'patience': 5, 'factor': 0.5}). Ignored if
        lr_scheduler is None.
    
    batch_size : int, optional
        Batch size for training and predicting.
    
    max_epochs : int, optional
        Maximum number of epochs.
    
    validation_fraction : float, optional
        Fraction of training set to use as validation set for early
        stopping. Must be strictly between 0 and 1.
    
    rel_tol, abs_tol : float, optional
        Relative and absolute tolerances for optimization. Validation
        loss is considered improving if it decreases by at least
        max(rel_tol * abs(best_val_loss), abs_tol) from best_val_loss,
        the best validation loss so far.
    
    patience : int, optional
        Number of allowed epochs without validation loss improvement
        before early stopping.
    
    name : str, optional
        Model name. Save parameters during training to <name>.pt, and as
        np.ndarrays to <name>.npz.
    
    verbose : bool, optional
        Whether to print progress bar during fit.
    
    Attributes
    ----------
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function to apply to predictions and targets.
    
    optimizer : torch.optim.Optimizer
        Optimizer to use during training.
    
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler | None
        Learning rate scheduler to use. If None, no scheduler is used.
    
    batch_size : int
        Batch size for training and predicting.
    
    max_epochs : int
        Maximum number of epochs.
    
    validation_fraction : float
        Fraction of training set to use as validation set for early
        stopping.
    
    rel_tol, abs_tol : float
        Relative and absolute tolerances for optimization. Validation
        loss is considered improving if it decreases by at least
        max(rel_tol * abs(best_val_loss), abs_tol) from best_val_loss,
        the best validation loss so far.
    
    patience : int
        Number of allowed epochs without validation loss improvement
        before early stopping.
    
    name : str
        Model name. Save parameters during training to <name>.pt, and as
        np.ndarrays to <name>.npz.
    
    verbose : bool
        Whether to print progress bar during fit.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        final_hidden_activation: Optional[nn.Module] = None,
        positive_final_weights: bool = True,
        dropout: float = 0.3,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[dict[str, Any]] = None,
        lr_scheduler: Optional[type[torch.optim.lr_scheduler.LRScheduler]] = None,
        lr_scheduler_params: Optional[dict[str, Any]] = None,
        batch_size: int = 128,
        max_epochs: int = 100,
        validation_fraction: float = 0.2,
        rel_tol: float = 1e-4,
        abs_tol: float = 0.0,
        patience: int = 20,
        name: str = 'model',
        verbose: bool = True
    ):
        if final_hidden_activation is None:
            final_hidden_activation = nn.ReLU()
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU() if i < len(hidden_dims) - 1 else final_hidden_activation)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        last_linear_layer = (
            PositiveLinear(hidden_dims[-1], 1) if positive_final_weights
            else nn.Linear(hidden_dims[-1], 1)
        )
        layers.append(last_linear_layer)
        super().__init__(*layers)

        # Set attributes
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.parameters(), **(optimizer_params or {}))
        self.lr_scheduler = None
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer, **(lr_scheduler_params or {}))
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.patience = patience
        self.name = name
        self.verbose = verbose
    
    def save(self):
        """Save model parameters to .pt file."""
        torch.save(self.state_dict(), f'{self.name}.pt')
    
    def load(self):
        """Load model parameters from .pt file."""
        self.load_state_dict(torch.load(f'{self.name}.pt'))
    
    def save_npz(self):
        """Save model parameters to .npz file."""
        # Doesn't include PositiveLinear if that's the last layer
        linear_layers = [layer for layer in self if isinstance(layer, nn.Linear)]
        weights_and_biases = {}
        for i, layer in enumerate(linear_layers):
            weights_and_biases[f'weight_{i}'] = layer.weight.detach().cpu().numpy()
            weights_and_biases[f'bias_{i}'] = layer.bias.detach().cpu().numpy()
        if isinstance(self[-1], PositiveLinear):
            weights_and_biases[f'weight_{len(linear_layers)}'] = self[-1].log_weight.exp().detach().cpu().numpy()
            weights_and_biases[f'bias_{len(linear_layers)}'] = self[-1].bias.detach().cpu().numpy()
        np.savez(f'{self.name}.npz', **weights_and_biases)
    
    @staticmethod
    def load_npz(filepath: str = 'model.npz') -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Load model parameters from .npz file.

        Parameters
        ----------
        filepath : str, optional
            Location of .npz file.
        
        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Weights and biases.
        """
        weights_and_biases = np.load(filepath)
        n_layers = len(weights_and_biases) // 2
        weights = [weights_and_biases[f'weight_{i}'] for i in range(n_layers)]
        biases = [weights_and_biases[f'bias_{i}'] for i in range(n_layers)]
        return weights, biases
    
    def set_dropout(self, dropout: float):
        """Set dropout probability.

        Parameters
        ----------
        dropout : float
            New dropout probability.
        """
        for layer in self:
            if isinstance(layer, nn.Dropout):
                layer.p = dropout
    
    def set_optimizer_and_lr_scheduler(
        self,
        optimizer: Optional[type[torch.optim.Optimizer]] = None,
        optimizer_params: Optional[dict[str, Any]] = None,
        lr_scheduler: Optional[type[torch.optim.lr_scheduler.LRScheduler]] = None,
        lr_scheduler_params: Optional[dict[str, Any]] = None
    ):
        """Set a new optimizer and learning rate scheduler.

        If optimizer is None, we keep the current optimizer and do not
        update its parameters (even if optimizer_params is not None). If
        lr_scheduler is None, no learning rate scheduler is used.

        Parameters
        ----------
        optimizer : type[torch.optim.Optimizer], optional
            Optimizer.
        
        optimizer_params : dict, optional
            Optimizer parameters. Ignored if optimizer is None.
        
        lr_scheduler : type[torch.optim.lr_scheduler.LRScheduler], optional
            Learning rate scheduler. If None, no scheduler is used.
        
        lr_scheduler_params : dict, optional
            Learning rate scheduler parameters. Ignored if lr_scheduler
            is None.
        """
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), **(optimizer_params or {}))
        self.lr_scheduler = None
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer, **(lr_scheduler_params or {}))
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Train model.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, in_dim)
            Inputs.
        
        y : torch.Tensor of shape (n_samples,)
            Targets.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_fraction)
        y_train = y_train.view(-1, 1)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_val_loss = self.loss_fn(self.predict(X_val), y_val).item()
        no_improvement_count = 0
        pbar = trange(self.max_epochs, unit="epoch", disable=not self.verbose)
        for _ in pbar:
            train_loss = 0.0
            for X_batch, y_batch in dataloader:
                # Forward pass
                batch_loss = self.loss_fn(self(X_batch), y_batch)
                # Backprop and update parameters
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                # Update training loss
                train_loss += batch_loss.item() * X_batch.shape[0]
            # Compute training and validation losses
            train_loss /= X_train.shape[0]
            val_loss = self.loss_fn(self.predict(X_val), y_val).item()
            # If validation loss improves, save checkpoint
            threshold = max(self.rel_tol * abs(best_val_loss), self.abs_tol)
            if best_val_loss - val_loss > threshold:
                no_improvement_count = 0
                best_val_loss = val_loss
                self.save()
            else:
                no_improvement_count += 1
            # Early stopping if patience runs out
            if no_improvement_count > self.patience:
                pbar.close()
                if self.verbose:
                    print("Early stopping")
                break
            # Update learning rate if applicable
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            # Update progress bar message
            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss, best_val_loss=best_val_loss)
        # Load best checkpoint
        self.load()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, in_dim)
            Input.
        
        Returns
        -------
        torch.Tensor of shape (n_samples,)
            Predictions.
        """
        self.eval()
        with torch.no_grad():
            y = []
            dataloader = DataLoader(X, batch_size=self.batch_size, shuffle=False)
            for X_batch in dataloader:
                y.append(self(X_batch))
            y = torch.cat(y)
        self.train()
        return y.flatten()
