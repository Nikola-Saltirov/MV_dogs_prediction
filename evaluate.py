"""Create test-set predictions for one selected trained neural model."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader

from config import NUM_CLASSES, RESULTS_DIR
from data import DogBreedDataset, create_dataloaders
from models import VALID_MODEL_NAMES, get_model


PREDICTION_COLUMNS = [
    "image_path",
    "true_class",
    "predicted_class",
    "confidence",
    "top5_predictions",
]


@dataclass(frozen=True)
class EvaluationResult:
    """Inference details needed to calculate and save test metrics."""

    total_inference_time_seconds: float
    num_images: int
    parameter_count: int
    class_names: list[str]


def get_test_image_paths(test_loader: DataLoader) -> list[Path]:
    """Return test paths in the same order used by the non-shuffled loader."""
    dataset = test_loader.dataset
    if not isinstance(dataset, DogBreedDataset):
        raise TypeError("The test DataLoader must use DogBreedDataset.")
    return dataset.image_paths


def load_best_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Create one selected model and load its saved best-model weights."""
    result_dir = Path(RESULTS_DIR) / model_name
    checkpoint_path = result_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint found for {model_name}: {checkpoint_path}"
        )

    print(f"[Evaluation] Creating only the {model_name} model.")
    model = get_model(model_name, num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Evaluation] Loaded best checkpoint: {checkpoint_path}.")
    return model


def evaluate_model(model_name: str) -> EvaluationResult:
    """Predict the test split for one model and save its prediction CSV.

    Returns timing, parameter-count, and class information for test-metric
    calculation from the saved prediction CSV.
    """
    print(f"[Evaluation] Preparing the test DataLoader for {model_name}.")
    _, _, test_loader, class_names = create_dataloaders()
    test_image_paths = get_test_image_paths(test_loader)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for neural-model evaluation, but PyTorch cannot access it."
        )
    device = torch.device("cuda")
    model = load_best_model(model_name, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    top_k = min(5, len(class_names))
    prediction_rows: list[dict[str, str | float]] = []
    image_index = 0

    print(
        f"[Evaluation] Running {len(test_loader)} test batches on {device}; "
        f"recording the top {top_k} predictions for each image."
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started_at = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)
            confidences, predicted_indices = probabilities.max(dim=1)
            _, top_indices = probabilities.topk(top_k, dim=1)

            batch_size = labels.size(0)
            batch_paths = test_image_paths[image_index : image_index + batch_size]
            if len(batch_paths) != batch_size:
                raise RuntimeError("Test image paths do not match prediction batches.")

            for image_path, true_index, predicted_index, confidence, top5_indices in zip(
                batch_paths,
                labels.tolist(),
                predicted_indices.cpu().tolist(),
                confidences.cpu().tolist(),
                top_indices.cpu().tolist(),
            ):
                prediction_rows.append(
                    {
                        "image_path": image_path.as_posix(),
                        "true_class": class_names[true_index],
                        "predicted_class": class_names[predicted_index],
                        "confidence": float(confidence),
                        "top5_predictions": json.dumps(
                            [class_names[index] for index in top5_indices]
                        ),
                    }
                )

            image_index += batch_size

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_inference_time_seconds = time.perf_counter() - inference_started_at

    if image_index != len(test_image_paths):
        raise RuntimeError("Not every test image received a prediction.")

    predictions_path = Path(RESULTS_DIR) / model_name / "predictions.csv"
    pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).to_csv(
        predictions_path,
        index=False,
    )
    print(
        f"[Evaluation] Saved {len(prediction_rows)} predictions to {predictions_path}. "
        f"Total inference time: {total_inference_time_seconds:.2f} seconds."
    )

    return EvaluationResult(
        total_inference_time_seconds=total_inference_time_seconds,
        num_images=len(prediction_rows),
        parameter_count=parameter_count,
        class_names=class_names,
    )


def load_predictions(result_dir: Path) -> pd.DataFrame:
    """Load and validate the prediction CSV saved for the selected model."""
    predictions_path = result_dir / "predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    missing_columns = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction file {predictions_path} is missing columns: "
            f"{sorted(missing_columns)}."
        )
    if predictions.empty:
        raise ValueError(f"Prediction file {predictions_path} contains no rows.")

    return predictions


def parse_top5_predictions(serialized_predictions: object) -> list[str]:
    """Return the class names serialized in one prediction CSV row."""
    if not isinstance(serialized_predictions, str):
        raise ValueError("top5_predictions values must be JSON strings.")

    try:
        top5_predictions = json.loads(serialized_predictions)
    except json.JSONDecodeError as error:
        raise ValueError("top5_predictions contains invalid JSON.") from error

    if not isinstance(top5_predictions, list) or not all(
        isinstance(class_name, str) for class_name in top5_predictions
    ):
        raise ValueError("top5_predictions must contain a JSON list of class names.")

    return top5_predictions


def load_training_info(result_dir: Path) -> dict[str, object]:
    """Read the training details needed for the final test-metric record."""
    training_info_path = result_dir / "training_info.json"
    if not training_info_path.is_file():
        raise FileNotFoundError(
            f"Training information does not exist: {training_info_path}"
        )

    with training_info_path.open(encoding="utf-8") as training_info_file:
        return json.load(training_info_file)


def calculate_test_metrics(
    model_name: str, evaluation_result: EvaluationResult
) -> dict[str, int | float | str]:
    """Calculate metrics from saved predictions and write model-specific files."""
    result_dir = Path(RESULTS_DIR) / model_name
    predictions = load_predictions(result_dir)
    if len(predictions) != evaluation_result.num_images:
        raise ValueError("Saved prediction count does not match the evaluated image count.")

    true_classes = predictions["true_class"].tolist()
    predicted_classes = predictions["predicted_class"].tolist()
    class_names = evaluation_result.class_names
    known_class_names = set(class_names)
    unknown_classes = (set(true_classes) | set(predicted_classes)) - known_class_names
    if unknown_classes:
        raise ValueError(
            f"Predictions contain unknown classes: {sorted(unknown_classes)}."
        )

    top5_accuracy = sum(
        true_class in parse_top5_predictions(top5_predictions)
        for true_class, top5_predictions in zip(
            true_classes,
            predictions["top5_predictions"].tolist(),
        )
    ) / len(predictions)

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            true_classes,
            predicted_classes,
            labels=class_names,
            average=None,
            zero_division=0,
        )
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_classes,
        predicted_classes,
        labels=class_names,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        true_classes,
        predicted_classes,
        labels=class_names,
        average="weighted",
        zero_division=0,
    )

    training_info = load_training_info(result_dir)
    try:
        training_time_seconds = float(training_info["total_training_time_seconds"])
        best_epoch = int(training_info["best_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "training_info.json is missing the required training-time or best-epoch value."
        ) from error

    top1_accuracy = float(accuracy_score(true_classes, predicted_classes))
    macro_precision_value = float(macro_precision)
    macro_recall_value = float(macro_recall)
    macro_f1_value = float(macro_f1)
    weighted_f1_value = float(weighted_f1)
    inference_ms_per_image = (
        evaluation_result.total_inference_time_seconds
        / evaluation_result.num_images
        * 1_000
    )
    overall_metrics: dict[str, int | float | str] = {
        "model": model_name,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": float(top5_accuracy),
        "macro_precision": macro_precision_value,
        "macro_recall": macro_recall_value,
        "macro_f1": macro_f1_value,
        "weighted_f1": weighted_f1_value,
        "parameters": evaluation_result.parameter_count,
        "training_time_seconds": training_time_seconds,
        "total_inference_time_seconds": evaluation_result.total_inference_time_seconds,
        "inference_ms_per_image": inference_ms_per_image,
        "best_epoch": best_epoch,
    }
    per_class_metrics = pd.DataFrame(
        {
            "class_name": class_names,
            "class_index": range(len(class_names)),
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
            "support": per_class_support,
        }
    )

    with (result_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(overall_metrics, file, indent=2)
    per_class_metrics.to_csv(result_dir / "per_class_metrics.csv", index=False)
    print(
        f"[Metrics] Saved test metrics and per-class metrics to {result_dir}. "
        f"Macro F1: {macro_f1_value:.4f}; "
        f"top-1 accuracy: {top1_accuracy:.4f}; "
        f"top-5 accuracy: {top5_accuracy:.4f}."
    )

    return overall_metrics


def parse_arguments() -> argparse.Namespace:
    """Parse the one neural model selected for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate one dog-breed classification model on the test set."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=VALID_MODEL_NAMES,
        help="Model to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run prediction generation for the model chosen on the command line."""
    arguments = parse_arguments()
    evaluation_result = evaluate_model(arguments.model)
    calculate_test_metrics(arguments.model, evaluation_result)


if __name__ == "__main__":
    main()
