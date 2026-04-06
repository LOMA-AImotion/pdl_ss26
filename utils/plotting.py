"""Shared plotting utilities for the pdl_ss26 course notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    title: str = "Training Curves",
) -> None:
    """Plot loss (and optionally accuracy) curves for training and validation."""
    has_acc = train_accs is not None and val_accs is not None
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(12 if has_acc else 6, 4))
    if not has_acc:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()

    if has_acc:
        axes[1].plot(epochs, train_accs, label="Train")
        axes[1].plot(epochs, val_accs, label="Val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> None:
    """Display a confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(cm.shape[0])
    labels = class_names if class_names is not None else [str(i) for i in tick_marks]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def show_image_grid(
    images: np.ndarray | list,
    labels: list[str] | None = None,
    n_cols: int = 8,
    title: str = "",
) -> None:
    """Display a grid of images (CHW or HWC format)."""
    images = list(images)
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx < n:
            img = np.array(images[idx])
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW → HWC
                img = img.transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img.squeeze(-1)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            if labels is not None and idx < len(labels):
                ax.set_title(str(labels[idx]), fontsize=8)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
