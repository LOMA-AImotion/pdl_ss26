"""Reusable training and evaluation loops for the pdl_ss26 course."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        (avg_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Evaluate the model on a data loader.

    Returns:
        (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 10,
    device: torch.device | str = "cpu",
    scheduler: object | None = None,
    callback: Callable[[int, float, float, float, float], None] | None = None,
) -> dict[str, list[float]]:
    """Full training loop with optional scheduler and callback.

    Args:
        callback: Called at the end of each epoch with
                  (epoch, train_loss, val_loss, train_acc, val_acc).

    Returns:
        History dict with keys "train_loss", "val_loss", "train_acc", "val_acc".
    """
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:>3}/{n_epochs} | "
            f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"val loss:   {val_loss:.4f}  acc: {val_acc:.4f}"
        )

        if callback is not None:
            callback(epoch, train_loss, val_loss, train_acc, val_acc)

    return history
