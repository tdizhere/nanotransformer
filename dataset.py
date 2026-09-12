
from dataclasses import dataclass
import kagglehub
import torch


@dataclass
class ShakespeareDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[character] for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[token] for token in tokens)


def load_dataset(
    validation_split: float = 0.1,
    dataset: str = "kingburrito666/shakespeare-plays",
) -> ShakespeareDataset:
    """Download the Kaggle dataset and return encoded train/validation data."""
    path = kagglehub.dataset_download(dataset)
    with open(f"{path}/alllines.txt", "r", encoding="utf-8") as file:
        text = file.read()

    characters = sorted(set(text))
    stoi = {character: index for index, character in enumerate(characters)}
    itos = {index: character for index, character in enumerate(characters)}
    encoded = torch.tensor([stoi[character] for character in text], dtype=torch.long)

    split = int((1.0 - validation_split) * len(encoded))
    return ShakespeareDataset(encoded[:split], encoded[split:], stoi, itos)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one random batch of input and target sequences."""
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[index : index + block_size] for index in indices])
    y = torch.stack([data[index + 1 : index + block_size + 1] for index in indices])
    return x.to(device), y.to(device)
