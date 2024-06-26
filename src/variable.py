START_LABEL = "<START>"
STOP_LABEL = "<STOP>"
PAD_LABEL = "<PAD>"
LABEL_TO_IDX = {
    'O': 0,
    'S-GPE': 1,
    'S-PER': 2,
    'B-ORG': 3,
    'E-ORG': 4,
    'S-ORG': 5,
    'M-ORG': 6,
    'S-LOC': 7,
    'E-GPE': 8,
    'B-GPE': 9,
    'B-LOC': 10,
    'E-LOC': 11,
    'M-LOC': 12,
    'M-GPE': 13,
    'B-PER': 14,
    'E-PER': 15,
    'M-PER': 16,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
SPECIAL_LABELS = [START_LABEL, STOP_LABEL, PAD_LABEL]
LABELS = [k for k in LABEL_TO_IDX.keys() if k not in SPECIAL_LABELS]
ENTITY_SUB_TYPE = ['GPE', 'LOC', 'ORG', 'PER']

def get_run_name(hparams):
    run_name = hparams.pretrained_model_name.split("/")[-1]
    run_name += f"_bertlr={hparams.bert_lr}"
    run_name += f"_lstmlr={hparams.lr}"
    run_name += f"_opt={hparams.optimizer}"
    run_name += f"_dim={hparams.lstm_state_dim}"
    run_name += f"_scheduler={hparams.scheduler}"
    return run_name
