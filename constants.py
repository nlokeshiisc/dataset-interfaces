from pathlib import Path
from torchvision import transforms

PROJ_DIR = Path(".").absolute()

# %% Imagenet constants
imagenet_star_dir: Path = Path(
    "/raid/infolab/nlokesh/dataset-interfaces/data/imagenet_star"
)
BASE = "base"
DUSK = "at_dusk"
NIGHT = "at_night"
SUNLIGHT = "in_bright_sunlight"
FOG = "in_the_fog"
FOREST = "in_the_forest"
RAIN = "in_the_rain"
SNOW = "in_the_snow"
STUDIO = "studio_lighting"

# %% Dataset constants
TRN = "trn"
VAL = "val"
TST = "tst"
TRNTST = "trntst"

# %% model constants

RESNET_TRANSFORMS = {
    TRN: transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    TST: transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}
RESNET_TRANSFORMS[VAL] = RESNET_TRANSFORMS[TST]
RESNET_TRANSFORMS[TRNTST] = RESNET_TRANSFORMS[TST]
