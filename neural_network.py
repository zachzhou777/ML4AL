from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class MLP(nn.Sequential):
    """Multilayer perceptron.

    Hidden units apply ReLU activation function.

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
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        super().__init__(*layers)
    
    def save_model(self, filepath: str = 'model.pt'):
        """Save model's state_dict to file.

        Note that dropout probability is not saved in state_dict.

        Parameters
        ----------
        filepath : str, optional
            Location to save model's state_dict.
        """
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load_model(cls, filepath: str = 'model.pt', dropout: float = 0.0) -> 'MLP':
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
        MLP
            Model with parameters loaded from state_dict.
        """
        state_dict = torch.load(filepath)
        layer_dims = [state_dict['0.weight'].shape[1]] + [param.shape[0] for name, param in state_dict.items() if 'weight' in name]
        model = cls(layer_dims[0], layer_dims[1:-1], layer_dims[-1], dropout)
        model.load_state_dict(state_dict)
        return model
    
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_dev: torch.Tensor,
        y_dev: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        tolerance: float = 0.0,
        filepath: str = 'model.pt',
        verbose: bool = True
    ):
        """Train model.

        Saves model's state_dict to file whenever dev loss improves. Performs early stopping.

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
        
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Loss function. Takes prediction and ground truth as inputs, and returns a tensor of shape (1,) containing the loss.
        
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use.
        
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
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optimizer(self.parameters())
        best_dev_loss = float('inf')
        for epoch in range(max_epochs):
            train_loss = 0.0  # Rolling sum
            for X_batch, y_batch in tqdm(dataloader, desc=f"Training (epoch {epoch+1}/{max_epochs})", disable=not verbose):
                # Forward pass
                batch_loss = loss_fn(self(X_batch), y_batch)
                # Backprop and update parameters
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # Update train loss
                train_loss += batch_loss.item() * X_batch.shape[0]
            # Compute train and dev losses
            train_loss /= X_train.shape[0]
            dev_loss = self.evaluate_loss(X_dev, y_dev, loss_fn, batch_size, verbose)
            if verbose:
                print(f"Train loss: {train_loss}, dev loss: {dev_loss}")
            # If dev loss improves, save model
            if dev_loss <= best_dev_loss - tolerance:
                no_improvement_count = 0
                best_dev_loss = dev_loss
                torch.save(self.state_dict(), filepath)
            else:
                no_improvement_count += 1
            # Break if patience runs out
            if no_improvement_count >= patience:
                if verbose:
                    print("Early stopping")
                break
        # Load best model
        self.load_state_dict(torch.load(filepath))
    
    def predict(
        self,
        X: torch.Tensor,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        batch_size: int = 128,
        verbose: bool = True
    ) -> torch.Tensor:
        """Predict outputs.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, in_dim)
            Input.
        
        activation : Callable[[torch.Tensor], torch.Tensor], optional
            Output activation function. If None, do nothing.
        
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
            for X_batch in tqdm(dataloader, desc="Predicting", disable=not verbose):
                y_batch = self(X_batch)
                if activation is not None:
                    y_batch = activation(y_batch)
                y.append(y_batch)
            y = torch.cat(y)
        self.train()
        return y

    def evaluate_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_size: int = 128,
        verbose: bool = True
    ) -> float:
        """Evaluate loss function.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, in_dim)
            Input.
        
        y : torch.Tensor of shape (n_samples, out_dim)
            Target.
        
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Loss function. Takes prediction and ground truth as inputs, and returns a tensor of shape (1,) containing the loss.
        
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
            loss = 0.0
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for X_batch, y_batch in tqdm(dataloader, desc="Evaluating", disable=not verbose):
                batch_loss = loss_fn(self(X_batch), y_batch)
                loss += batch_loss.item() * X_batch.shape[0]
            loss /= X.shape[0]
        self.train()
        return loss
    
    @staticmethod
    def modified_sigmoid(z: torch.Tensor) -> torch.Tensor:
        """A smooth concave function that returns sigmoid(z) for z > 0, 0.25*z + 0.5 otherwise.
        
        Reasons for using this instead of the true sigmoid function:
        - The MIP formulation implements a piecewise linear approximation of this function.
        - This function may incentivize the model to predict probabilities above 0.5.

        Parameters
        ----------
        z : torch.Tensor
            Logits.
        
        Returns
        -------
        torch.Tensor
            sigmoid(z) if z > 0, 0.25*z + 0.5 otherwise.
        """
        return torch.where(z > 0, torch.sigmoid(z), 0.25*z + 0.5)
    
    @staticmethod
    def demand_weighted_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        demand: torch.Tensor,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sum_outputs: bool = True,
        weight_overestimate: float = 1.0,
        weight_underestimate: float = 1.0
    ) -> torch.Tensor:
        """Custom loss for predicting ambulance location metrics.

        To use in MLP.fit and MLP.evaluate_loss, wrap in another function that takes only input and target as arguments (e.g., a lambda function).

        Parameters
        ----------
        input : torch.Tensor of shape (n_samples, out_dim)
            Model prediction (before output activation function is applied).
        
        target : torch.Tensor of shape (n_samples, out_dim)
            Ground truth.
        
        demand : torch.Tensor of shape (out_dim,)
            Demand weights.
        
        activation : Callable[[torch.Tensor], torch.Tensor], optional
            Output activation function. If None, do nothing.
        
        sum_outputs : bool, optional
            Whether to sum weighted outputs before feeding into MSE loss.
        
        weight_overestimate : float, optional
            Multiplier applied when prediction exceeds target.
        
        weight_underestimate : float, optional
            Multiplier applied when prediction falls short of target.
        
        Returns
        -------
        torch.Tensor
            A scalar tensor containing the loss.
        """
        if activation is not None:
            input = activation(input)
        mul = torch.matmul if sum_outputs else torch.mul
        weighted_predictions = mul(input, demand)
        weighted_targets = mul(target, demand)
        # When weights are 1.0, this is equivalent to F.mse_loss(weighted_predictions, weighted_targets)
        loss = torch.where(weighted_predictions > weighted_targets, weight_overestimate, weight_underestimate) * (weighted_predictions - weighted_targets)**2
        return loss.mean()
