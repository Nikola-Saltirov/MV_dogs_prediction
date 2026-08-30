"""Dataset class discovery and shared class-index mapping."""

import random
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import BATCH_SIZE, NUM_CLASSES, NUM_WORKERS, TEST_DIR, TRAIN_DIR
from config import IMAGE_SIZE, RESULTS_DIR, SEED, VALIDATION_IMAGES_PER_CLASS


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_TRAIN_IMAGES_PER_CLASS = 100
SPLIT_COLUMNS = ["image_path", "class_name", "class_index", "split"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DogBreedDataset(Dataset[tuple[Tensor, int]]):
    """Dataset that loads dog images from paths and returns image-label pairs.

    Parameters
    ----------
    image_paths:
        Paths to the images to load. Paths may be strings or ``Path`` objects.
    labels:
        Integer class labels corresponding to ``image_paths``.
    transform:
        Optional torchvision transform applied after converting an image to RGB.

    The paths and labels are kept as parallel sequences so a saved split can be
    used directly without recreating an ``ImageFolder`` directory structure.
    """

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        labels: Sequence[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                "image_paths and labels must contain the same number of items."
            )

        self.image_paths = [Path(image_path) for image_path in image_paths]
        self.labels = [int(label) for label in labels]
        self.transform = (
            transform if transform is not None else transforms.ToTensor()
        )

    def __len__(self) -> int:
        """Return the number of images in this dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load, convert, transform, and return one image and its label."""
        image_path = self.image_paths[index]
        label = self.labels[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        if not isinstance(image, Tensor):
            raise TypeError(
                "The dataset transform must return a torch.Tensor. "
                "Use transforms.ToTensor() in the transform pipeline."
            )

        return image, label


# A concise generic name is useful to callers that do not need the project-
# specific class name. Keep the explicit name above as the primary API.
ImageDataset = DogBreedDataset


def get_class_names(train_dir: str | Path = TRAIN_DIR) -> list[str]:
    """Return the alphabetically ordered breed folders in ``train_dir``.

    The order is the single source of truth for class labels in every split.
    """
    train_path = Path(train_dir)
    if not train_path.is_dir():
        raise FileNotFoundError(
            f"Training directory does not exist: {train_path}"
        )

    class_names = sorted(
        folder.name for folder in train_path.iterdir() if folder.is_dir()
    )

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} breed folders in {train_path}, "
            f"found {len(class_names)}."
        )

    return class_names


def get_class_to_index(train_dir: str | Path = TRAIN_DIR) -> dict[str, int]:
    """Return the shared mapping from breed name to integer class index."""
    class_names = get_class_names(train_dir)
    return {class_name: index for index, class_name in enumerate(class_names)}


def get_class_info(
    train_dir: str | Path = TRAIN_DIR,
) -> tuple[list[str], dict[str, int]]:
    """Return ordered class names and their matching index mapping.

    Consumers should use this mapping for training, validation, and test data
    so that a breed always receives the same integer label.
    """
    class_names = get_class_names(train_dir)
    class_to_index = {
        class_name: index for index, class_name in enumerate(class_names)
    }
    return class_names, class_to_index


def _list_class_images(class_dir: Path) -> list[Path]:
    """Return the sorted image files belonging to one breed folder."""
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _build_split(
    train_dir: str | Path,
    class_names: list[str],
    class_to_index: dict[str, int],
) -> pd.DataFrame:
    """Build a deterministic, per-class train/validation split."""
    train_path = Path(train_dir)
    random_generator = random.Random(SEED)
    records: list[dict[str, str | int]] = []

    for class_name in class_names:
        image_paths = _list_class_images(train_path / class_name)
        if len(image_paths) != EXPECTED_TRAIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Expected {EXPECTED_TRAIN_IMAGES_PER_CLASS} images in "
                f"{train_path / class_name}, found {len(image_paths)}."
            )

        random_generator.shuffle(image_paths)
        validation_paths = image_paths[:VALIDATION_IMAGES_PER_CLASS]
        training_paths = image_paths[VALIDATION_IMAGES_PER_CLASS:]

        for split_name, split_paths in (
            ("train", training_paths),
            ("val", validation_paths),
        ):
            records.extend(
                {
                    "image_path": image_path.as_posix(),
                    "class_name": class_name,
                    "class_index": class_to_index[class_name],
                    "split": split_name,
                }
                for image_path in split_paths
            )

    return pd.DataFrame(records, columns=SPLIT_COLUMNS)


def _validate_split(
    split_df: pd.DataFrame,
    split_path: Path,
    class_names: list[str],
    class_to_index: dict[str, int],
) -> pd.DataFrame:
    """Validate a saved split before reusing it."""
    if list(split_df.columns) != SPLIT_COLUMNS:
        raise ValueError(
            f"Split file {split_path} must contain columns: {SPLIT_COLUMNS}."
        )

    expected_rows = NUM_CLASSES * EXPECTED_TRAIN_IMAGES_PER_CLASS
    if len(split_df) != expected_rows:
        raise ValueError(
            f"Split file {split_path} must contain {expected_rows} rows, "
            f"found {len(split_df)}."
        )

    if set(split_df["class_name"]) != set(class_names):
        raise ValueError(f"Split file {split_path} contains an unexpected class set.")

    expected_indexes = split_df["class_name"].map(class_to_index)
    if not split_df["class_index"].equals(expected_indexes):
        raise ValueError(
            f"Class indices in {split_path} do not match the sorted class mapping."
        )

    if set(split_df["split"]) != {"train", "val"}:
        raise ValueError(f"Split file {split_path} must contain train and val rows.")

    for class_name in class_names:
        class_rows = split_df[split_df["class_name"] == class_name]
        split_counts = class_rows["split"].value_counts().to_dict()
        expected_counts = {
            "train": EXPECTED_TRAIN_IMAGES_PER_CLASS - VALIDATION_IMAGES_PER_CLASS,
            "val": VALIDATION_IMAGES_PER_CLASS,
        }
        if split_counts != expected_counts:
            raise ValueError(
                f"Class {class_name} in {split_path} has split counts "
                f"{split_counts}; expected {expected_counts}."
            )

    missing_paths = [
        image_path
        for image_path in split_df["image_path"]
        if not Path(image_path).is_file()
    ]
    if missing_paths:
        preview = ", ".join(missing_paths[:3])
        raise FileNotFoundError(
            f"Split file {split_path} references missing images: {preview}"
        )

    return split_df


def create_or_load_split(
    train_dir: str | Path = TRAIN_DIR,
    results_dir: str | Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Create ``results/split.csv`` once, then reuse the saved split."""
    class_names, class_to_index = get_class_info(train_dir)
    split_path = Path(results_dir) / "split.csv"

    if split_path.exists():
        split_df = pd.read_csv(split_path, dtype={"class_index": "int64"})
        return _validate_split(
            split_df,
            split_path,
            class_names,
            class_to_index,
        )

    split_df = _build_split(train_dir, class_names, class_to_index)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_path, index=False)
    return split_df


def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Create the random augmentation pipeline used for training images."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Create the deterministic pipeline shared by validation and test images."""
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_transforms(
    image_size: int = IMAGE_SIZE,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return the training transform and shared validation/test transform."""
    return get_train_transform(image_size), get_eval_transform(image_size)


def _get_test_samples(
    test_dir: str | Path,
    class_names: list[str],
    class_to_index: dict[str, int],
) -> tuple[list[Path], list[int]]:
    """Return test image paths and labels in the shared class order."""
    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_path}")

    test_class_names = {
        folder.name for folder in test_path.iterdir() if folder.is_dir()
    }
    expected_class_names = set(class_names)
    missing_classes = sorted(expected_class_names - test_class_names)
    unexpected_classes = sorted(test_class_names - expected_class_names)

    if missing_classes or unexpected_classes:
        details = []
        if missing_classes:
            details.append(f"missing classes: {missing_classes[:3]}")
        if unexpected_classes:
            details.append(f"unexpected classes: {unexpected_classes[:3]}")
        raise ValueError(
            f"Test directory {test_path} does not match the training classes "
            f"({'; '.join(details)})."
        )

    image_paths: list[Path] = []
    labels: list[int] = []
    for class_name in class_names:
        class_image_paths = _list_class_images(test_path / class_name)
        if not class_image_paths:
            raise ValueError(
                f"Test class directory {test_path / class_name} contains no images."
            )

        image_paths.extend(class_image_paths)
        labels.extend([class_to_index[class_name]] * len(class_image_paths))

    return image_paths, labels


def create_dataloaders(
    train_dir: str | Path = TRAIN_DIR,
    test_dir: str | Path = TEST_DIR,
    results_dir: str | Path = RESULTS_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create train, validation, and test DataLoaders using the saved split.

    The training DataLoader uses the augmentation transform and shuffles each
    epoch. Validation and test use the deterministic evaluation transform and
    preserve their dataset order for reproducible evaluation.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    class_names, class_to_index = get_class_info(train_dir)
    split_df = create_or_load_split(train_dir, results_dir)

    train_df = split_df[split_df["split"] == "train"]
    validation_df = split_df[split_df["split"] == "val"]
    train_transform, eval_transform = create_transforms(image_size)

    train_dataset = DogBreedDataset(
        train_df["image_path"].tolist(),
        train_df["class_index"].tolist(),
        transform=train_transform,
    )
    validation_dataset = DogBreedDataset(
        validation_df["image_path"].tolist(),
        validation_df["class_index"].tolist(),
        transform=eval_transform,
    )

    test_paths, test_labels = _get_test_samples(
        test_dir,
        class_names,
        class_to_index,
    )
    test_dataset = DogBreedDataset(
        test_paths,
        test_labels,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader, class_names
