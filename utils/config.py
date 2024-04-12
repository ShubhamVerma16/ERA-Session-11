import toml
from pydantic import BaseModel

TOML_PATH = "config.toml"


class Data(BaseModel):
    batch_size: int = 512
    shuffle: bool = True
    num_workers: int = 4


class LRFinder(BaseModel):
    numiter: int = 600
    endlr: float = 10
    startlr: float = 1e-2


class Training(BaseModel):
    epochs: int = 20
    optimizer: str = "adam"
    criterion: str = "crossentropy"
    lr: float = 0.003
    weight_decay: float = 1e-4
    lrfinder: LRFinder


class Config(BaseModel):
    data: Data
    training: Training


with open(TOML_PATH) as f:
    toml_config = toml.load(f)

config = Config(**toml_config)
