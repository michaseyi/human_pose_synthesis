import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Optional
import os


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            dataset_path: str,
            checkpoint_path: str,
            lr: float = 3e-4,
            epochs: int = 5000,
            batch_size: int = 64,
            train_ratio=0.8,
            early_stopper_patience=5,
            device = 'cpu'
    ):
        torch.manual_seed(0)

        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        self.epochs = epochs
        self.epoch = 0
        self.train_loss: Optional[float] = None
        self.val_loss: Optional[float] = None

        self.early_stopper = EarlyStopping(patience=early_stopper_patience)

        self.data = torch.load(dataset_path, map_location="cpu")

        dataset = self.data['dataset'].to(device)

        X, y = dataset[:, :-1], dataset[:, 1:]

        dataset = TensorDataset(X, y)

        test_ratio = (1.0 - train_ratio) / 2.0
        val_ratio = test_ratio

        train_dataset, test_dataset, val_dataset = random_split(
            dataset, [train_ratio, test_ratio, val_ratio])

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        self.checkpoint_path = checkpoint_path

        if os.path.exists(self.checkpoint_path) and os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}:")
            self.load_checkpoint(self.checkpoint_path)
            self.log_stat()

    def log_stat(self):
        print(f"Epoch {self.epoch}: train loss - {self.train_loss}, val loss - {self.val_loss}")

    @torch.no_grad()
    def evaluate_loss(self, data_loader):
        self.model.eval()
        total_loss = 0
        for batch in data_loader:
            X, y = batch
            logits = self.model.forward(X)
            total_loss += self.loss(logits, y).item()
        self.model.train()
        return total_loss / len(data_loader)

    def train(self):
        self.model.train()
        for _ in range(self.epoch, self.epochs):
            self.epoch += 1
            total_loss = 0
            for batch in self.train_loader:
                X, y = batch
                logits = self.model.forward(X)
                loss = self.loss(logits, y)
                total_loss += loss.item()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            self.train_loss = total_loss / len(self.train_loader)
            self.val_loss = self.evaluate_loss(self.val_loader)
            self.log_stat()
            self.save_checkpoint(self.checkpoint_path)

            self.early_stopper(self.val_loss)
            if self.early_stopper.early_stop:
                print("Early stopping")
                break

        self.model.eval()

    def loss(self, X: torch.Tensor, y: torch.Tensor):
        return F.mse_loss(X, y).sqrt()

    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.val_loss = checkpoint['val_loss']
        self.train_loss = checkpoint['train_loss']


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
