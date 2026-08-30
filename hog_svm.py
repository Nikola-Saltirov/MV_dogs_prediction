"""Train and evaluate the HOG plus linear-SVM baseline independently.

This baseline deliberately does not import or instantiate any PyTorch model.
It reuses the fixed ``results/split.csv`` train/validation split, selects the
linear SVM regularisation value using validation macro F1, then writes all
test artefacts to ``results/hog_svm``.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC
from skimage.feature import hog
from tqdm.auto import tqdm

from config import NUM_CLASSES, RESULTS_DIR, TEST_DIR, TRAIN_DIR
from data import IMAGE_EXTENSIONS, SPLIT_COLUMNS, get_class_info


MODEL_NAME = "hog_svm"
HOG_IMAGE_SIZE = 128
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
C_VALUES = (0.01, 0.1, 1.0, 10.0)
MAX_ITERATIONS = 5_000
RANDOM_SEED = 42
PREDICTION_COLUMNS = [
    "image_path",
    "true_class",
    "predicted_class",
    "confidence",
    "top5_predictions",
]


def list_image_files(directory: Path) -> list[Path]:
    """Return supported images in a directory in a reproducible order."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_saved_split() -> tuple[list[Path], np.ndarray, list[Path], np.ndarray, list[str]]:
    """Load and validate the already-created fixed train/validation split."""
    split_path = Path(RESULTS_DIR) / "split.csv"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"The saved split does not exist: {split_path}. "
            "Run a neural-model training command first to create the fixed split."
        )

    print(f"[HOG-SVM] Reading the saved train/validation split: {split_path}.")
    split_dataframe = pd.read_csv(split_path)
    missing_columns = set(SPLIT_COLUMNS) - set(split_dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{split_path} is missing required columns: {sorted(missing_columns)}."
        )
    if split_dataframe.empty:
        raise ValueError(f"The saved split contains no rows: {split_path}.")
    if set(split_dataframe["split"]) != {"train", "val"}:
        raise ValueError(f"{split_path} must contain exactly the train and val splits.")

    class_names, class_to_index = get_class_info(TRAIN_DIR)
    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, but found {len(class_names)} in {TRAIN_DIR}."
        )
    if set(split_dataframe["class_name"]) != set(class_names):
        raise ValueError("The saved split class names do not match the training folders.")

    try:
        split_dataframe["class_index"] = split_dataframe["class_index"].astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("The saved split has non-integer class_index values.") from error

    expected_indices = split_dataframe["class_name"].map(class_to_index)
    if expected_indices.isna().any() or not np.array_equal(
        split_dataframe["class_index"].to_numpy(),
        expected_indices.to_numpy(dtype=int),
    ):
        raise ValueError(
            "Class indices in the saved split do not match the shared sorted mapping."
        )

    missing_images = [
        image_path
        for image_path in split_dataframe["image_path"].tolist()
        if not Path(image_path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            "The saved split references missing images: "
            f"{', '.join(missing_images[:3])}."
        )

    train_dataframe = split_dataframe[split_dataframe["split"] == "train"]
    validation_dataframe = split_dataframe[split_dataframe["split"] == "val"]
    if train_dataframe.empty or validation_dataframe.empty:
        raise ValueError("The saved split must contain at least one train and one val image.")

    print(
        "[HOG-SVM] Loaded "
        f"{len(train_dataframe)} training and {len(validation_dataframe)} validation images "
        f"across {len(class_names)} classes."
    )
    return (
        [Path(path) for path in train_dataframe["image_path"].tolist()],
        train_dataframe["class_index"].to_numpy(dtype=np.int64),
        [Path(path) for path in validation_dataframe["image_path"].tolist()],
        validation_dataframe["class_index"].to_numpy(dtype=np.int64),
        class_names,
    )


def load_test_samples(class_names: list[str]) -> tuple[list[Path], np.ndarray]:
    """Load test image paths using the class mapping from the saved split."""
    test_directory = Path(TEST_DIR)
    if not test_directory.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_directory}.")

    test_class_names = {
        directory.name for directory in test_directory.iterdir() if directory.is_dir()
    }
    expected_class_names = set(class_names)
    missing_classes = sorted(expected_class_names - test_class_names)
    unexpected_classes = sorted(test_class_names - expected_class_names)
    if missing_classes or unexpected_classes:
        details: list[str] = []
        if missing_classes:
            details.append(f"missing classes: {missing_classes[:3]}")
        if unexpected_classes:
            details.append(f"unexpected classes: {unexpected_classes[:3]}")
        raise ValueError(
            f"Test directory {test_directory} does not match the saved class mapping "
            f"({'; '.join(details)})."
        )

    image_paths: list[Path] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(class_names):
        class_paths = list_image_files(test_directory / class_name)
        if not class_paths:
            raise ValueError(
                f"Test class directory contains no supported images: "
                f"{test_directory / class_name}."
            )
        image_paths.extend(class_paths)
        labels.extend([class_index] * len(class_paths))

    print(
        f"[HOG-SVM] Found {len(image_paths)} test images in {test_directory}."
    )
    return image_paths, np.asarray(labels, dtype=np.int64)


def extract_hog_descriptor(image_path: Path) -> np.ndarray:
    """Open one RGB image, resize it to 128 by 128, and calculate its HOG vector."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        resized_image = rgb_image.resize(
            (HOG_IMAGE_SIZE, HOG_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        image_array = np.asarray(resized_image)

    descriptor = hog(
        image_array,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        channel_axis=-1,
        feature_vector=True,
    )
    return descriptor.astype(np.float32, copy=False)


def extract_features(image_paths: list[Path], split_name: str) -> np.ndarray:
    """Extract an equally sized HOG descriptor for every image in one split."""
    if not image_paths:
        raise ValueError(f"Cannot extract HOG features for an empty {split_name} split.")

    print(
        f"[HOG-SVM] Extracting HOG features for {len(image_paths)} {split_name} images "
        f"(resize={HOG_IMAGE_SIZE}x{HOG_IMAGE_SIZE}, "
        f"pixels_per_cell={HOG_PIXELS_PER_CELL})."
    )
    features: list[np.ndarray] = []
    for image_path in tqdm(image_paths, desc=f"HOG {split_name}", unit="image"):
        try:
            features.append(extract_hog_descriptor(image_path))
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Could not extract HOG features from {image_path}."
            ) from error

    feature_matrix = np.stack(features).astype(np.float32, copy=False)
    print(
        f"[HOG-SVM] Finished {split_name} feature extraction: "
        f"{feature_matrix.shape[0]} x {feature_matrix.shape[1]} descriptors."
    )
    return feature_matrix


def create_linear_svm(regularization_c: float) -> LinearSVC:
    """Create a deterministic linear SVM for the HOG feature matrix."""
    return LinearSVC(
        C=regularization_c,
        dual=False,
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_SEED,
    )


def select_regularization(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[LinearSVC, float, list[dict[str, float]]]:
    """Choose the SVM C value with the highest validation macro F1."""
    best_model: LinearSVC | None = None
    best_c = 0.0
    best_macro_f1 = float("-inf")
    validation_runs: list[dict[str, float]] = []

    print(
        f"[HOG-SVM] Selecting C using validation macro F1 across {len(C_VALUES)} values."
    )
    for regularization_c in C_VALUES:
        candidate_started_at = time.perf_counter()
        model = create_linear_svm(regularization_c)
        model.fit(train_features, train_labels)
        validation_predictions = model.predict(validation_features)
        validation_macro_f1 = float(
            f1_score(
                validation_labels,
                validation_predictions,
                average="macro",
                zero_division=0,
            )
        )
        validation_accuracy = float(
            accuracy_score(validation_labels, validation_predictions)
        )
        elapsed_seconds = time.perf_counter() - candidate_started_at
        validation_runs.append(
            {
                "c": regularization_c,
                "macro_f1": validation_macro_f1,
                "top1_accuracy": validation_accuracy,
                "fit_and_validation_seconds": elapsed_seconds,
            }
        )
        print(
            f"[HOG-SVM] C={regularization_c:g}: validation macro F1="
            f"{validation_macro_f1:.4f}, top-1 accuracy={validation_accuracy:.4f}, "
            f"time={elapsed_seconds:.1f}s."
        )

        if validation_macro_f1 > best_macro_f1:
            best_model = model
            best_c = regularization_c
            best_macro_f1 = validation_macro_f1

    if best_model is None:
        raise RuntimeError("No HOG-SVM candidate model was trained.")

    print(
        f"[HOG-SVM] Selected C={best_c:g} with validation macro F1 "
        f"{best_macro_f1:.4f}."
    )
    return best_model, best_c, validation_runs


def confirm_model_overwrite(model_path: Path) -> None:
    """Ask before replacing the existing independently saved SVM model."""
    if not model_path.exists():
        print(f"[HOG-SVM] No existing saved model at {model_path}.")
        return

    print(f"[HOG-SVM] An existing saved model was found at {model_path}.")
    response = input(f"{model_path} already exists. Overwrite it? [y/N]: ").strip().casefold()
    if response not in {"y", "yes"}:
        raise RuntimeError("HOG-SVM run cancelled; the existing model was not replaced.")


def softmax_confidences(decision_scores: np.ndarray) -> np.ndarray:
    """Convert SVM decision scores to normalized relative confidences for CSV output."""
    shifted_scores = decision_scores - decision_scores.max(axis=1, keepdims=True)
    exponentiated_scores = np.exp(shifted_scores)
    return exponentiated_scores / exponentiated_scores.sum(axis=1, keepdims=True)


def save_predictions_and_metrics(
    model: LinearSVC,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_paths: list[Path],
    class_names: list[str],
    result_dir: Path,
    training_time_seconds: float,
    best_c: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate the selected SVM and save predictions, metrics, and per-class data."""
    print(f"[HOG-SVM] Evaluating the selected model on {len(test_paths)} test images.")
    inference_started_at = time.perf_counter()
    decision_scores = np.asarray(model.decision_function(test_features), dtype=np.float64)
    test_predictions = model.predict(test_features).astype(np.int64, copy=False)
    total_inference_time_seconds = time.perf_counter() - inference_started_at

    if decision_scores.ndim != 2:
        raise RuntimeError(
            "Expected multiclass decision scores for the 120-class HOG-SVM model."
        )
    if len(test_paths) != len(test_labels) or len(test_paths) != len(test_predictions):
        raise RuntimeError("Test paths, labels, and predictions must have the same length.")

    model_classes = model.classes_.astype(np.int64, copy=False)
    if not np.array_equal(model_classes, np.arange(len(class_names))):
        raise RuntimeError("The SVM class order does not match the saved split class mapping.")

    top_k = min(5, len(class_names))
    ranking_indices = np.argsort(decision_scores, axis=1)[:, ::-1][:, :top_k]
    ranked_classes = model_classes[ranking_indices]
    confidences = softmax_confidences(decision_scores).max(axis=1)
    prediction_rows = [
        {
            "image_path": image_path.as_posix(),
            "true_class": class_names[true_index],
            "predicted_class": class_names[predicted_index],
            "confidence": float(confidence),
            "top5_predictions": json.dumps(
                [class_names[class_index] for class_index in top_class_indices]
            ),
        }
        for image_path, true_index, predicted_index, confidence, top_class_indices in zip(
            test_paths,
            test_labels.tolist(),
            test_predictions.tolist(),
            confidences.tolist(),
            ranked_classes.tolist(),
        )
    ]
    pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).to_csv(
        result_dir / "predictions.csv",
        index=False,
    )

    top1_accuracy = float(accuracy_score(test_labels, test_predictions))
    top5_accuracy = float(
        np.mean(
            [
                true_label in top_class_indices
                for true_label, top_class_indices in zip(
                    test_labels.tolist(),
                    ranked_classes.tolist(),
                )
            ]
        )
    )
    labels = np.arange(len(class_names))
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            test_labels,
            test_predictions,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    parameter_count = int(model.coef_.size + model.intercept_.size)
    inference_ms_per_image = total_inference_time_seconds / len(test_paths) * 1_000
    overall_metrics: dict[str, Any] = {
        "model": MODEL_NAME,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "parameters": parameter_count,
        "training_time_seconds": training_time_seconds,
        "total_inference_time_seconds": total_inference_time_seconds,
        "inference_ms_per_image": inference_ms_per_image,
        "best_epoch": None,
        "best_c": best_c,
        "confidence_type": "softmax_normalized_decision_score",
    }
    with (result_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(overall_metrics, file, indent=2)

    per_class_metrics = pd.DataFrame(
        {
            "class_name": class_names,
            "class_index": labels,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
            "support": per_class_support,
        }
    )
    per_class_metrics.to_csv(result_dir / "per_class_metrics.csv", index=False)
    print(
        f"[HOG-SVM] Saved predictions, test metrics, and per-class metrics to "
        f"{result_dir}. Macro F1={float(macro_f1):.4f}; "
        f"top-1 accuracy={top1_accuracy:.4f}; top-5 accuracy={top5_accuracy:.4f}; "
        f"inference={inference_ms_per_image:.3f} ms/image."
    )
    return overall_metrics, test_predictions


def save_confusion_matrix(
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
    class_names: list[str],
    result_dir: Path,
) -> Path:
    """Save the normalized 120-class test confusion-matrix figure."""
    normalized_matrix = confusion_matrix(
        test_labels,
        test_predictions,
        labels=np.arange(len(class_names)),
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
    axis.set_title("HOG-SVM normalized test confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.tick_params(axis="both", labelsize=4)

    output_path = result_dir / "confusion_matrix.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[HOG-SVM] Saved normalized confusion matrix: {output_path}.")
    return output_path


def run_hog_svm() -> None:
    """Run the complete independent HOG plus linear-SVM baseline pipeline."""
    pipeline_started_at = time.perf_counter()
    result_dir = Path(RESULTS_DIR) / MODEL_NAME
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = result_dir / "model.joblib"
    confirm_model_overwrite(model_path)

    print("[HOG-SVM] Starting the independent HOG plus linear-SVM baseline.")
    train_paths, train_labels, validation_paths, validation_labels, class_names = (
        load_saved_split()
    )
    test_paths, test_labels = load_test_samples(class_names)

    training_started_at = time.perf_counter()
    train_features = extract_features(train_paths, "training")
    validation_features = extract_features(validation_paths, "validation")
    selected_model, best_c, validation_runs = select_regularization(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )

    selected_validation_run = next(
        run for run in validation_runs if run["c"] == best_c
    )
    validation_metrics = {
        "model": MODEL_NAME,
        "best_c": best_c,
        "validation_macro_f1": selected_validation_run["macro_f1"],
        "validation_top1_accuracy": selected_validation_run["top1_accuracy"],
        "candidate_runs": validation_runs,
    }
    with (result_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(validation_metrics, file, indent=2)
    print(f"[HOG-SVM] Saved validation-selection metrics to {result_dir}.")

    print(f"[HOG-SVM] Saving the selected linear SVM to {model_path}.")
    joblib.dump(
        {
            "model": selected_model,
            "class_names": class_names,
            "hog_settings": {
                "image_size": HOG_IMAGE_SIZE,
                "orientations": HOG_ORIENTATIONS,
                "pixels_per_cell": HOG_PIXELS_PER_CELL,
                "cells_per_block": HOG_CELLS_PER_BLOCK,
            },
            "best_c": best_c,
        },
        model_path,
        compress=3,
    )

    training_time_seconds = time.perf_counter() - training_started_at
    print(
        f"[HOG-SVM] Training and validation selection completed in "
        f"{training_time_seconds:.1f}s."
    )
    test_features = extract_features(test_paths, "test")
    _, test_predictions = save_predictions_and_metrics(
        selected_model,
        test_features,
        test_labels,
        test_paths,
        class_names,
        result_dir,
        training_time_seconds,
        best_c,
    )
    save_confusion_matrix(test_labels, test_predictions, class_names, result_dir)

    total_pipeline_time_seconds = time.perf_counter() - pipeline_started_at
    print(
        f"[HOG-SVM] Complete. Total pipeline time: {total_pipeline_time_seconds:.1f}s. "
        f"Results are in {result_dir}."
    )


def main() -> None:
    """Run the HOG-SVM baseline from ``python hog_svm.py``."""
    run_hog_svm()


if __name__ == "__main__":
    main()
