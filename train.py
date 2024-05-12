from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset, builder
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser
from src.dataset import NERDataset
from src.collator import NERDataCollator
from src.pl_module import LightningBiLSTMCRF
from src.variable import LABEL_TO_IDX, get_run_name

builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

def print_hparams(hparams):
    # haparams: Namespace
    print("Hyperparameters:")
    for k, v in vars(hparams).items():
        print(f"{k}: {v}")

def main(hparams):

    print_hparams(hparams)
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_model_name)

    print("Initializing dataset...")
    train_dataset_name = "toy" if hparams.toy else "train"
    train_dataset = NERDataset(train_dataset_name, LABEL_TO_IDX, upsample=hparams.upsample)
    val_dataset_name = "toy" if hparams.toy else "dev"
    val_dataset = NERDataset(val_dataset_name, LABEL_TO_IDX, upsample=False)
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
        encoding['word_ids'] = [list(map(lambda x: -1 if x is None else x, word_id)) for word_id in encoding['word_ids']]
        encoding['labels'] = [[LABEL_TO_IDX[y] for y in b] for b in example['labels']]
        return encoding

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=hparams.batch_size, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=hparams.batch_size, remove_columns=["text"])
    collator = NERDataCollator(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=hparams.batch_size, num_workers=47, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn=collator, batch_size=hparams.batch_size, num_workers=47)

    print("Initializing model...")
    model = LightningBiLSTMCRF(LABEL_TO_IDX, hparams.lstm_layer_num, hparams.lstm_state_dim, 
                            bert_lr=hparams.bert_lr, lstm_lr=hparams.lstm_lr, crf_lr=hparams.crf_lr,
                            optimizer=hparams.optimizer, scheduler=hparams.scheduler,
                            pretrained_model_name=hparams.pretrained_model_name, freeze_bert=hparams.bert_lr==0.0,
                            epochs=hparams.epoch, steps_per_epoch=len(train_loader))


    print("Training model...")
    run_name = get_run_name(hparams)
    logger = TensorBoardLogger("tb_logs3" if hparams.search else "run", name=run_name)
    if not hparams.search:
        on_val_loss = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3, dirpath=f"best_models/{run_name}", filename="{epoch}-{val_loss:.4f}-{val_f1:.4f}")
        on_val_f1 = ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=3, dirpath=f"best_models/{run_name}", filename="{epoch}-{val_loss:.4f}-{val_f1:.4f}")
    trainer = Trainer(max_epochs=hparams.epoch, fast_dev_run=hparams.fast_dev_run, logger=logger, log_every_n_steps=1, 
                    enable_checkpointing=not hparams.search, callbacks=[on_val_loss, on_val_f1] if not hparams.search else None)
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
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--pretrained_model_name", default="bert-base-chinese", type=str)
    parser.add_argument("--search", default=False, action="store_true")
    parser.add_argument("--scheduler", default="none", type=str)
    parser.add_argument("--upsample", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
