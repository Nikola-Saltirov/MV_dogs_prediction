"""Shared configuration for the dog-breed classification project."""

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
ANNOTATION_DIR = "dataset/annotations/Annotation"
RESULTS_DIR = "results"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 120
VALIDATION_IMAGES_PER_CLASS = 20
SEED = 42
NUM_WORKERS = 4

EPOCHS = 100
LEARNING_RATE = 0.0001
PATIENCE = 5

USE_BOUNDING_BOX = False
BOUNDING_BOX_PADDING = 0.10
