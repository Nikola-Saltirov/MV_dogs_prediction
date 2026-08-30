"""Neural-network architectures for dog-breed classification."""

from torch import Tensor, nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
    vit_b_16,
)


_CLASSIFIER_PATH_ATTRIBUTE = "_dog_breed_classifier_path"


def _validate_num_classes(num_classes: int) -> None:
    """Ensure that a classifier has at least one output."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")


class CustomCNN(nn.Module):
    """A compact convolutional baseline trained from scratch.

    The network accepts RGB images of any spatial size. Adaptive global average
    pooling reduces the final feature map to one value per channel before the
    classifier, producing one logit for every dog-breed class.
    """

    def __init__(self, num_classes: int = 120, dropout: float = 0.5) -> None:
        """Initialize the convolutional feature extractor and classifier."""
        super().__init__()

        _validate_num_classes(num_classes)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalized class scores for a batch of RGB images."""
        features = self.features(images)
        pooled_features = self.global_average_pool(features)
        return self.classifier(pooled_features)


def set_feature_layers_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze a pretrained model's feature layers.

    When ``trainable`` is ``False``, the replacement classification head remains
    trainable. This supports training only that new head before fine-tuning the
    complete model. Passing ``True`` enables gradients for every layer.
    """
    classifier_path = getattr(model, _CLASSIFIER_PATH_ATTRIBUTE, None)
    if classifier_path is None:
        raise ValueError(
            "The model does not have a registered pretrained classification head."
        )

    for parameter in model.parameters():
        parameter.requires_grad = trainable

    if not trainable:
        classifier = model.get_submodule(classifier_path)
        for parameter in classifier.parameters():
            parameter.requires_grad = True


def create_resnet50(
    num_classes: int = 120, freeze_features: bool = False
) -> nn.Module:
    """Create an ImageNet-pretrained ResNet-50 with a new classifier head."""
    _validate_num_classes(num_classes)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    setattr(model, _CLASSIFIER_PATH_ATTRIBUTE, "fc")

    if freeze_features:
        set_feature_layers_trainable(model, trainable=False)

    return model


def create_efficientnet_b0(
    num_classes: int = 120, freeze_features: bool = False
) -> nn.Module:
    """Create an ImageNet-pretrained EfficientNet-B0 with a new classifier."""
    _validate_num_classes(num_classes)

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("EfficientNet-B0's final classifier layer must be linear.")

    model.classifier[-1] = nn.Linear(final_layer.in_features, num_classes)
    setattr(model, _CLASSIFIER_PATH_ATTRIBUTE, "classifier")

    if freeze_features:
        set_feature_layers_trainable(model, trainable=False)

    return model


def create_mobilenet_v3_large(
    num_classes: int = 120, freeze_features: bool = False
) -> nn.Module:
    """Create an ImageNet-pretrained MobileNetV3-Large with a new classifier."""
    _validate_num_classes(num_classes)

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("MobileNetV3-Large's final classifier layer must be linear.")

    model.classifier[-1] = nn.Linear(final_layer.in_features, num_classes)
    setattr(model, _CLASSIFIER_PATH_ATTRIBUTE, "classifier")

    if freeze_features:
        set_feature_layers_trainable(model, trainable=False)

    return model


def create_vit_b16(num_classes: int = 120, freeze_features: bool = False) -> nn.Module:
    """Create an ImageNet-pretrained ViT-B/16 with a new classification head."""
    _validate_num_classes(num_classes)

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    final_layer = model.heads.head
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("ViT-B/16's classification head must be linear.")

    model.heads.head = nn.Linear(final_layer.in_features, num_classes)
    setattr(model, _CLASSIFIER_PATH_ATTRIBUTE, "heads")

    if freeze_features:
        set_feature_layers_trainable(model, trainable=False)

    return model


MODEL_BUILDERS = {
    "custom_cnn": CustomCNN,
    "resnet50": create_resnet50,
    "efficientnet_b0": create_efficientnet_b0,
    "vit_b16": create_vit_b16,
    "mobilenet_v3_large": create_mobilenet_v3_large,
}
VALID_MODEL_NAMES = tuple(MODEL_BUILDERS)


def get_model(model_name: str, num_classes: int = 120) -> nn.Module:
    """Create and return only the neural model selected by ``model_name``.

    MobileNetV3-Large is included as the optional lightweight model requested
    for this project. Pretrained models initially have trainable feature layers;
    call :func:`set_feature_layers_trainable` to freeze or unfreeze them during
    training.
    """
    try:
        model_builder = MODEL_BUILDERS[model_name]
    except KeyError:
        valid_choices = ", ".join(VALID_MODEL_NAMES)
        message = f"Invalid model name '{model_name}'. Valid choices: {valid_choices}."
        print(message)
        raise ValueError(message) from None

    return model_builder(num_classes=num_classes)
