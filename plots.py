"""Create paper-ready plots for one evaluated neural model."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR
from models import VALID_MODEL_NAMES


HISTORY_COLUMNS = ["epoch", "train_loss", "val_loss", "train_macro_f1", "val_macro_f1"]
PREDICTION_COLUMNS = ["true_class", "predicted_class"]
PER_CLASS_COLUMNS = ["class_name", "class_index", "precision", "recall", "f1", "support"]


def load_result_csv(
    result_dir: Path, filename: str, required_columns: list[str]
) -> pd.DataFrame:
    """Load a result CSV and verify that it contains the expected columns."""
    csv_path = result_dir / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required result file does not exist: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}."
        )
    if dataframe.empty:
        raise ValueError(f"Required result file contains no rows: {csv_path}")

    return dataframe


def plot_training_curves(result_dir: Path) -> Path:
    """Save one figure with training/validation loss and macro-F1 curves."""
    history = load_result_csv(result_dir, "history.csv", HISTORY_COLUMNS)
    history = history.sort_values("epoch")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Training")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        history["epoch"],
        history["train_macro_f1"],
        label="Training",
    )
    axes[1].plot(
        history["epoch"],
        history["val_macro_f1"],
        label="Validation",
    )
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.suptitle("Training and validation curves")
    figure.tight_layout()
    output_path = result_dir / "training_curves.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[Plots] Saved training curves: {output_path}.")
    return output_path


def plot_confusion_matrix(result_dir: Path) -> Path:
    """Save the normalized test confusion matrix for one selected model."""
    predictions = load_result_csv(result_dir, "predictions.csv", PREDICTION_COLUMNS)
    per_class_metrics = load_result_csv(
        result_dir,
        "per_class_metrics.csv",
        PER_CLASS_COLUMNS,
    ).sort_values("class_index")
    class_names = per_class_metrics["class_name"].tolist()
    known_class_names = set(class_names)
    observed_class_names = set(predictions["true_class"]) | set(
        predictions["predicted_class"]
    )
    unknown_class_names = observed_class_names - known_class_names
    if unknown_class_names:
        raise ValueError(
            f"Prediction data contains unknown classes: {sorted(unknown_class_names)}."
        )

    normalized_matrix = confusion_matrix(
        predictions["true_class"],
        predictions["predicted_class"],
        labels=class_names,
        normalize="true",
    )
    matrix_dataframe = pd.DataFrame(
        normalized_matrix,
        index=class_names,
        columns=class_names,
    )
    figure, axis = plt.subplots(figsize=(28, 24))
    sns.heatmap(
        matrix_dataframe,
        ax=axis,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Fraction of true class"},
    )
    axis.set_title("Normalized test confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.tick_params(axis="both", labelsize=4)

    output_path = result_dir / "confusion_matrix.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[Plots] Saved normalized confusion matrix: {output_path}.")
    return output_path


def plot_per_class_f1(result_dir: Path) -> Path:
    """Save a lowest-to-highest horizontal chart of per-class F1 scores."""
    per_class_metrics = load_result_csv(
        result_dir,
        "per_class_metrics.csv",
        PER_CLASS_COLUMNS,
    ).sort_values("f1", ascending=True)
    figure_height = max(8, len(per_class_metrics) * 0.22)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    axis.barh(
        per_class_metrics["class_name"],
        per_class_metrics["f1"],
        color="#4c72b0",
    )
    axis.set_title("Per-class F1 score (lowest to highest)")
    axis.set_xlabel("F1 score")
    axis.set_ylabel("Dog breed")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.3)
    axis.tick_params(axis="y", labelsize=7)

    output_path = result_dir / "per_class_f1.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[Plots] Saved sorted per-class F1 chart: {output_path}.")
    return output_path


def create_model_plots(model_name: str) -> list[Path]:
    """Create all required plot files for only the selected model."""
    result_dir = Path(RESULTS_DIR) / model_name
    if not result_dir.is_dir():
        raise FileNotFoundError(f"No result directory exists for {model_name}: {result_dir}")

    print(f"[Plots] Creating plots for {model_name} in {result_dir}.")
    return [
        plot_training_curves(result_dir),
        plot_confusion_matrix(result_dir),
        plot_per_class_f1(result_dir),
    ]


def parse_arguments() -> argparse.Namespace:
    """Parse the one neural model selected for plot generation."""
    parser = argparse.ArgumentParser(
        description="Create plots for one evaluated dog-breed classification model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=VALID_MODEL_NAMES,
        help="Model whose result files should be plotted.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the selected model's result plots."""
    arguments = parse_arguments()
    create_model_plots(arguments.model)


if __name__ == "__main__":
    main()
