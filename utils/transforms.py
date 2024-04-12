import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            always_apply=True,
        ),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.CoarseDropout(
            min_holes=1,
            max_holes=1,
            min_height=8,
            min_width=8,
            max_height=8,
            max_width=8,
            fill_value=(0.4914, 0.4822, 0.4465),  # type: ignore
            p=0.5,
        ),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            always_apply=True,
        ),
        ToTensorV2(),
    ]
)
