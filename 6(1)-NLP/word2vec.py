import torch
from torch import nn, Tensor
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from typing import Literal, Callable
from torch.utils.data import DataLoader, Dataset


class CBOWDataset(Dataset):
    def __init__(self, data: list[tuple[list[int], int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.data[idx]


class SkipGramDataset(Dataset):
    def __init__(self, data: list[tuple[int, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.data[idx]


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        token_ids = tokenizer(corpus, add_special_tokens=False, padding=False, truncation=False)["input_ids"]

        train_fn: Callable[[DataLoader, nn.CrossEntropyLoss, Adam], None]
        loader: DataLoader

        if self.method == "cbow":
            cbow_data: list[tuple[list[int], int]] = self._generate_cbow_data(token_ids)
            cbow_dataset: CBOWDataset = CBOWDataset(cbow_data)
            loader = DataLoader(cbow_dataset, batch_size=256, shuffle=True, collate_fn=self._collate_cbow)
            train_fn = self._train_cbow
        else:
            skipgram_data: list[tuple[int, int]] = self._generate_skipgram_data(token_ids)
            skipgram_dataset: SkipGramDataset = SkipGramDataset(skipgram_data)
            loader = DataLoader(skipgram_dataset, batch_size=256, shuffle=True)
            train_fn = self._train_skipgram

        for _ in range(num_epochs):
            train_fn(loader, criterion, optimizer)

    def _generate_cbow_data(self, token_ids: list[list[int]]) -> list[tuple[list[int], int]]:
        pairs = []
        for seq in token_ids:
            for idx in range(self.window_size, len(seq) - self.window_size):
                context = seq[idx - self.window_size:idx] + seq[idx + 1:idx + self.window_size + 1]
                target = seq[idx]
                pairs.append((context, target))
        return pairs

    def _generate_skipgram_data(self, token_ids: list[list[int]]) -> list[tuple[int, int]]:
        pairs = []
        for seq in token_ids:
            for idx in range(len(seq)):
                for j in range(-self.window_size, self.window_size + 1):
                    if j == 0 or not (0 <= idx + j < len(seq)):
                        continue
                    center = seq[idx]
                    context = seq[idx + j]
                    pairs.append((center, context))
        return pairs

    def _collate_cbow(self, batch: list[tuple[list[int], int]]) -> tuple[Tensor, Tensor]:
        contexts, targets = zip(*batch)
        context_tensor = torch.tensor(contexts, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        return context_tensor, target_tensor

    def _train_cbow(
        self,
        loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> None:
        self.train()
        device = self.embeddings.weight.device
        for contexts, targets in loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            embeds = self.embeddings(contexts)
            mean_embeds = embeds.mean(dim=1)
            logits = self.weight(mean_embeds)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _train_skipgram(
        self,
        loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> None:
        self.train()
        device = self.embeddings.weight.device
        for centers, contexts in loader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            embeds = self.embeddings(centers)
            logits = self.weight(embeds)
            loss = criterion(logits, contexts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
