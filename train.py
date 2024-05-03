from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset, builder
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from argparse import ArgumentParser
from src.dataset import NERDataset
from src.collator import NERDataCollator
from src.pl_module import LightningBiLSTMCRF
from src.variable import LABEL_TO_IDX, PAD_LABEL

builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

import torch
torch.set_float32_matmul_precision('medium' | 'high')

def print_hparams(hparams):
    # haparams: Namespace
    print("Hyperparameters:")
    for k, v in vars(hparams).items():
        print(f"{k}: {v}")

def get_run_name(hparams):
    # include(in order): pretrained_model_name, bert_lr, lstm_lr, char_level(if true)
    run_name = hparams.pretrained_model_name.split("/")[-1]
    run_name += f"_bertlr{hparams.bert_lr}"
    run_name += f"_lstmlr{hparams.lstm_lr}"
    if hparams.char_level:
        run_name += "_char"
    return run_name

def main(hparams):

    print_hparams(hparams)
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    print("Initializing model...")
    model = LightningBiLSTMCRF(LABEL_TO_IDX, hparams.lstm_layer_num, hparams.lstm_state_dim, 
                            bert_lr=hparams.bert_lr, lstm_lr=hparams.lstm_lr, crf_lr=hparams.crf_lr,
                            char_level=hparams.char_level, optimizer=hparams.optimizer, pretrained_model_name=hparams.pretrained_model_name, freeze_bert=hparams.bert_lr==0.0)

    print("Initializing dataset...")
    train_dataset_name = "toy" if hparams.toy else "train"
    train_dataset = NERDataset(train_dataset_name, LABEL_TO_IDX)
    val_dataset_name = "toy" if hparams.toy else "dev"
    val_dataset = NERDataset(val_dataset_name, LABEL_TO_IDX)
    def train_generator():
        for i in range(len(train_dataset)):
            yield {"text": train_dataset.text[i], "labels": train_dataset.labels[i]}
    def val_generator():
        for i in range(len(val_dataset)):
            yield {"text": val_dataset.text[i], "labels": val_dataset.labels[i]}
    train_dataset = Dataset.from_generator(train_generator)
    val_dataset = Dataset.from_generator(val_generator)

    print("Tokenizing dataset...")
    def tokenize(example):
        encoding = tokenizer(example["text"], is_split_into_words=True)
        encoding['word_ids'] = [encoding.word_ids(b) for b in range(len(example['labels']))]
        if not hparams.char_level:
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

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=hparams.batch_size, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=hparams.batch_size, remove_columns=["text"])

    print("Training model...")
    collator = NERDataCollator(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=hparams.batch_size, num_workers=47)
    val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=hparams.batch_size, num_workers=47)
    logger = TensorBoardLogger("tb_logs", name=get_run_name(hparams))
    trainer = Trainer(max_epochs=hparams.epoch, fast_dev_run=hparams.fast_dev_run, logger=logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fast_dev_run", default=False, action="store_true")
    parser.add_argument("--toy", default=False, action="store_true")
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--bert_lr", default=1e-2, type=float, help="Seperate learning rate for bert model; 0.0 means freeze bert model")
    parser.add_argument("--lstm_lr", default=1e-2, type=float)
    parser.add_argument("--crf_lr", default=1e-2, type=float)
    parser.add_argument("--lstm_state_dim", default=256, type=int)
    parser.add_argument("--lstm_layer_num", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--char_level", default=False, action="store_true")
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--pretrained_model_name", default="bert-base-chinese", type=str)
    args = parser.parse_args()
    main(args)
