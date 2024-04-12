import torchvision
from utils.transforms import train_transform, test_transform


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


train_dataset = Cifar10SearchDataset(
    root="./data", train=True, transform=train_transform
)

test_dataset = Cifar10SearchDataset(
    root="./data", train=False, transform=test_transform
)