"""Train a BPE tokenizer (vocab 16,384) from scratch on wikitext-103 train.

This is statistical preprocessing, not a pretrained model (per task spec).
"""

import os
import sys

import yaml


def train_tokenizer(cfg):
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    path = cfg["tokenizer"]["path"]
    if os.path.exists(path):
        print(f"tokenizer already at {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["dataset_config"],
                      split="train")

    tok = Tokenizer(models.BPE(unk_token=None))
    # Byte-level: lossless on any input, no unk needed
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def text_iter(batch=10000):
        for s in range(0, len(ds), batch):
            yield "".join(ds[s : s + batch]["text"])

    tok.train_from_iterator(text_iter(), trainer=trainer, length=len(ds) // 10000)
    tok.save(path)
    print(f"saved {path} (vocab={tok.get_vocab_size()})")


def load_tokenizer(cfg):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(cfg["tokenizer"]["path"])


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
    train_tokenizer(cfg)
