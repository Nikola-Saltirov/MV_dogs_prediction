"""Train one selected neural model at a time."""

import argparse
from collections.abc import Sized
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    PATIENCE,
    RESULTS_DIR,
    SEED,
)
from data import create_dataloaders
from models import VALID_MODEL_NAMES, get_model


@dataclass
class TrainingSetup:
    """Resources needed to train one selected neural model."""

    model_name: str
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    model: nn.Module
    device: torch.device
    criterion: nn.Module
    optimizer: Optimizer
    result_dir: Path
    history: list[dict[str, int | float]] = field(default_factory=list)


def get_dataset_size(loader: DataLoader, split_name: str) -> int:
    """Return a DataLoader dataset size, rejecting unsized iterable datasets."""
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError(
            f"The {split_name} dataset has no known size and cannot be logged."
        )
    return len(dataset)


def create_training_setup(model_name: str) -> TrainingSetup:
    """Create the resources required to train only the selected model."""
    print(f"[Setup] Creating data loaders for the selected model: {model_name}.")
    train_loader, validation_loader, test_loader, class_names = create_dataloaders()
    train_images = get_dataset_size(train_loader, "training")
    validation_images = get_dataset_size(validation_loader, "validation")
    test_images = get_dataset_size(test_loader, "test")
    print(
        "[Setup] Data loaders are ready: "
        f"{train_images} training images, "
        f"{validation_images} validation images, and "
        f"{test_images} test images across {len(class_names)} classes."
    )
    print("[Setup] The test loader is prepared but will not be used during training.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for neural-model training, but PyTorch cannot access it."
        )
    device = torch.device("cuda")
    print(f"[Setup] Using device: {device}.")
    print(f"[Setup] Creating only the {model_name} model.")
    model = get_model(model_name, num_classes=NUM_CLASSES).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"[Setup] Model is ready with {total_parameters:,} parameters "
        f"({trainable_parameters:,} currently trainable)."
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"[Setup] Configured CrossEntropyLoss and AdamW (lr={LEARNING_RATE}).")

    result_dir = Path(RESULTS_DIR) / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Results for this run will be written to: {result_dir}.")

    return TrainingSetup(
        model_name=model_name,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        class_names=class_names,
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        result_dir=result_dir,
    )


def train_one_epoch(
    setup: TrainingSetup, epoch: int, total_epochs: int
) -> dict[str, float]:
    """Train the selected model for one epoch and return its metrics."""
    setup.model.train()
    print(
        f"[Train] Epoch {epoch}/{total_epochs}: training mode enabled; "
        f"processing {len(setup.train_loader)} batches."
    )
    total_loss = 0.0
    correct_predictions = 0
    total_images = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for images, labels in setup.train_loader:
        images = images.to(setup.device, non_blocking=True)
        labels = labels.to(setup.device, non_blocking=True)

        setup.optimizer.zero_grad()
        predictions = setup.model(images)
        loss = setup.criterion(predictions, labels)
        loss.backward()
        setup.optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predicted_classes = predictions.argmax(dim=1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_images += batch_size
        true_labels.extend(labels.detach().cpu().tolist())
        predicted_labels.extend(predicted_classes.detach().cpu().tolist())

    if total_images == 0:
        raise ValueError("The training DataLoader contains no images.")

    return {
        "train_loss": total_loss / total_images,
        "train_accuracy": correct_predictions / total_images,
        "train_macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def validate_model(setup: TrainingSetup) -> dict[str, float]:
    """Evaluate the selected model on the validation split for one epoch."""
    setup.model.eval()
    print(
        "[Validation] Evaluation mode enabled; "
        f"processing {len(setup.validation_loader)} validation batches."
    )
    total_loss = 0.0
    correct_predictions = 0
    total_images = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.no_grad():
        for images, labels in setup.validation_loader:
            images = images.to(setup.device, non_blocking=True)
            labels = labels.to(setup.device, non_blocking=True)

            predictions = setup.model(images)
            loss = setup.criterion(predictions, labels)
            predicted_classes = predictions.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (predicted_classes == labels).sum().item()
            total_images += batch_size
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predicted_classes.cpu().tolist())

    if total_images == 0:
        raise ValueError("The validation DataLoader contains no images.")

    return {
        "val_loss": total_loss / total_images,
        "val_accuracy": correct_predictions / total_images,
        "val_macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def save_history(setup: TrainingSetup) -> None:
    """Save all recorded epoch metrics in the selected model's result folder."""
    history_path = setup.result_dir / "history.csv"
    pd.DataFrame(setup.history).to_csv(history_path, index=False)
    print(f"[Results] Updated training history: {history_path}.")


def save_json(file_path: Path, values: dict[str, object]) -> None:
    """Write a JSON result file using readable indentation."""
    with file_path.open("w", encoding="utf-8") as result_file:
        json.dump(values, result_file, indent=2)


def confirm_checkpoint_overwrite(checkpoint_path: Path) -> None:
    """Ask before replacing an existing best-model checkpoint."""
    if not checkpoint_path.exists():
        print(f"[Checkpoint] No existing checkpoint found at {checkpoint_path}.")
        return

    print(f"[Checkpoint] An existing checkpoint was found at {checkpoint_path}.")
    response = input(
        f"{checkpoint_path} already exists. Overwrite it? [y/N]: "
    ).strip().casefold()
    if response not in {"y", "yes"}:
        raise FileExistsError(
            "Training cancelled to preserve the existing checkpoint."
        )


def save_best_model(
    setup: TrainingSetup,
    epoch: int,
    validation_metrics: dict[str, float],
    checkpoint_path: Path,
) -> None:
    """Save the current best weights and their validation metrics."""
    torch.save(setup.model.state_dict(), checkpoint_path)
    save_json(
        setup.result_dir / "validation_metrics.json",
        {
            "model_name": setup.model_name,
            "best_epoch": epoch,
            "val_loss": validation_metrics["val_loss"],
            "val_accuracy": validation_metrics["val_accuracy"],
            "val_macro_f1": validation_metrics["val_macro_f1"],
        },
    )
    print(
        f"[Checkpoint] New best validation macro F1 "
        f"({validation_metrics['val_macro_f1']:.4f}) at epoch {epoch}; "
        f"saved weights to {checkpoint_path}."
    )


def train_model(model_name: str) -> TrainingSetup:
    """Train only the selected model for the configured number of epochs."""
    if EPOCHS < 1:
        raise ValueError("EPOCHS must be at least 1.")
    if PATIENCE < 1:
        raise ValueError("PATIENCE must be at least 1.")

    print(f"[Pipeline] Starting a training run for {model_name}.")
    setup = create_training_setup(model_name)
    checkpoint_path = setup.result_dir / "best_model.pth"
    confirm_checkpoint_overwrite(checkpoint_path)

    best_validation_macro_f1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_started_at = time.perf_counter()
    training_started_at_display = datetime.now().astimezone()
    print(
        "[Timing] Training started at "
        f"{training_started_at_display:%Y-%m-%d %H:%M:%S %Z}."
    )

    for epoch in range(1, EPOCHS + 1):
        print(f"[Pipeline] Beginning epoch {epoch}/{EPOCHS}.")
        epoch_started_at = time.perf_counter()
        training_metrics = train_one_epoch(setup, epoch, EPOCHS)
        validation_metrics = validate_model(setup)
        epoch_time_seconds = time.perf_counter() - epoch_started_at
        elapsed_training_time_seconds = time.perf_counter() - training_started_at
        epoch_metrics = {
            "epoch": epoch,
            **training_metrics,
            **validation_metrics,
            "epoch_time_seconds": epoch_time_seconds,
            "elapsed_training_time_seconds": elapsed_training_time_seconds,
        }
        setup.history.append(epoch_metrics)
        save_history(setup)

        if validation_metrics["val_macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = validation_metrics["val_macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_best_model(setup, epoch, validation_metrics, checkpoint_path)
        else:
            epochs_without_improvement += 1
            print(
                "[Checkpoint] Validation macro F1 did not improve; "
                f"patience is {epochs_without_improvement}/{PATIENCE}."
            )

        print(
            f"Epoch {epoch}/{EPOCHS} - "
            f"train loss: {training_metrics['train_loss']:.4f}, "
            f"train accuracy: {training_metrics['train_accuracy']:.4f}, "
            f"train macro F1: {training_metrics['train_macro_f1']:.4f}, "
            f"val loss: {validation_metrics['val_loss']:.4f}, "
            f"val accuracy: {validation_metrics['val_accuracy']:.4f}, "
            f"val macro F1: {validation_metrics['val_macro_f1']:.4f}"
        )
        print(
            f"[Timing] Epoch {epoch}/{EPOCHS} took {epoch_time_seconds:.1f} seconds; "
            f"total training time is {elapsed_training_time_seconds:.1f} seconds."
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping after {PATIENCE} epochs without an improvement "
                "in validation macro F1."
            )
            break

    total_training_time_seconds = time.perf_counter() - training_started_at
    print("[Results] Saving training configuration and timing information.")
    save_json(
        setup.result_dir / "training_info.json",
        {
            "model_name": setup.model_name,
            "seed": SEED,
            "epochs": EPOCHS,
            "epochs_completed": len(setup.history),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_validation_macro_f1,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "total_training_time_seconds": total_training_time_seconds,
        },
    )
    print(
        f"[Pipeline] Training completed after {len(setup.history)} epochs in "
        f"{total_training_time_seconds:.1f} seconds. "
        f"Best validation macro F1: {best_validation_macro_f1:.4f} "
        f"at epoch {best_epoch}."
    )

    return setup


def parse_arguments() -> argparse.Namespace:
    """Parse the explicitly selected model name from the command line."""
    parser = argparse.ArgumentParser(
        description="Train one dog-breed classification model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=VALID_MODEL_NAMES,
        help="Model to train.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the requested model and report the location of its results."""
    arguments = parse_arguments()
    try:
        setup = train_model(arguments.model)
    except FileExistsError as error:
        print(error)
        return

    print(
        f"Finished training {setup.model_name} on {setup.device.type}. "
        f"Results will be saved in {setup.result_dir}."
    )


if __name__ == "__main__":
    main()
