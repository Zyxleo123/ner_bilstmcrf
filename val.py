import os
from src.dataset import NERDataset
from src.collator import NERDataCollator
from src.pl_module import LightningBiLSTMCRF
from src.variable import LABEL_TO_IDX
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset, builder
from pytorch_lightning import Trainer

builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

def test(path):

    print("Loading model...")
    model = LightningBiLSTMCRF.load_from_checkpoint(os.path.join(path))

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(f"bert-base-chinese")

    print("Initializing dataset...")
    val_dataset = NERDataset('dev', LABEL_TO_IDX)
    def test_generator():
        for i in range(len(val_dataset)):
            yield {"text": val_dataset.text[i], "labels": val_dataset.labels[i]}
    val_dataset = Dataset.from_generator(test_generator)

    print("Tokenizing dataset...")
    def tokenize(example):
        encoding = tokenizer(example["text"], is_split_into_words=True)
        encoding['word_ids'] = [encoding.word_ids(b) for b in range(len(example['text']))]
        if not model.hparams.char_level:
            encoding['labels'] = example['labels']
        else:
            # align labels with word_ids
            labels = example['labels']
            word_ids = encoding['word_ids']
            new_labels = []
            for b in range(len(labels)):
                new_label = []
                for w in word_ids[b]:
                    if w is not None:
                        new_label.append(labels[b][w])
                    else:
                        new_label.append('O')
                new_labels.append(new_label)
            encoding['labels'] = new_labels
        return encoding
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"], batch_size=32)

    print("Initializing dataloader...")
    collator = NERDataCollator(tokenizer)
    val_dataloader = DataLoader(val_dataset, collate_fn=collator, batch_size=1)

    trainer = Trainer()
    print("Start validating...")
    trainer.validate(model, dataloaders=val_dataloader)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    path = args.path
    test(path)