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
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.pretrained_model_name)

    print("Initializing dataset...")
    test_dataset = NERDataset('test', LABEL_TO_IDX, upsample=False)
    def test_generator():
        for i in range(len(test_dataset)):
            yield {"text": test_dataset.text[i]}
    test_dataset = Dataset.from_generator(test_generator)

    print("Tokenizing dataset...")
    def tokenize(example):
        encoding = tokenizer(example["text"], is_split_into_words=True)
        encoding['word_ids'] = [encoding.word_ids(b) for b in range(len(example['text']))]
        encoding['word_ids'] = [list(map(lambda x: -1 if x is None else x, word_id)) for word_id in encoding['word_ids']]
        return encoding
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text"], batch_size=32)

    print("Initializing dataloader...")
    collator = NERDataCollator(tokenizer)
    test_dataloader = DataLoader(test_dataset, collate_fn=collator, batch_size=1)

    trainer = Trainer()
    print("Start testing...")
    trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    path = args.path
    test(path)