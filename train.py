import torch
from torch import nn


class Trainer:
    def __init__(self, model: nn.Module, lr: float = 3e-4, epochs: int = 5000):
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        self.epochs = epochs
        pass

    @torch.no_grad()
    def evaluate_loss(self, data_loader):
        self.model.eval()
        total = 0
        length = 0
        for batch in data_loader:
            X, y = batch
            logits = self.model.forward(X)
            loss = self.loss(logits, y)
            total += loss
            length += 1
        print(f"loss: {total / length}")
        self.model.train()

    def train(self, train_loader):
        self.model.train()
        for i in range(self.epochs):
            total = 0
            length = 0
            for batch in train_loader:
                X, y = batch
                logits = self.model.forward(X)
                loss = self.loss(logits, y)
                total += loss
                length += 1
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {i + 1}: loss - {total / length}")
        self.model.eval()

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = X - y
        loss = torch.norm(loss, p=2, dim=-1)
        loss = torch.square(loss)
        loss = torch.mean(loss)
        return torch.sqrt(loss)

    def save_checkpoint(self, epoch: int, loss: float, filename='checkpoint.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename='checkpoint.pth') -> tuple[int, float]:
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model, optimizer, epoch):
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
