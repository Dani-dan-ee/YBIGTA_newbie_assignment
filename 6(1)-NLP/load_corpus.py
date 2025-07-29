from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    for sample in dataset:
        corpus.append(sample["verse_text"])
    return corpus
