import subprocess
from copy import deepcopy
def get_run_name(command):
    command = command.split(" ")
    pretrained_model_name = command[command.index("--pretrained_model_name") + 1].split("/")[-1]
    bert_lr = command[command.index("--bert_lr") + 1]
    lstm_lr = command[command.index("--lstm_lr") + 1]
    return f"{pretrained_model_name}_bertlr={bert_lr}_lstmlr={lstm_lr}"
env_command = \
    "export CUDA_VISIBLE_DEVICES=6 && " + \
    "export http_proxy=127.0.0.1:7890 && " + \
    "export https_proxy=127.0.0.1:7890 && "

command = [ "nohup",
    "python", "train.py",
    "--epoch", "10",
    "--lstm_state_dim", "256",
    "--lstm_layer_num", "1",
    "--batch_size", "32",
]

LR_FREEZE = 0
LR_SAME = -1
LR_MODEL = -2

MODEL_LR = {'ernie': 6e-5, 'bert': 3e-5}

lstm_lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
bert_lr = [LR_FREEZE, LR_SAME, LR_MODEL]
pretrained_model_names = ['hfl/chinese-roberta-wwm-ext-large',
	        'hfl/chinese-roberta-wwm-ext',
            'hfl/chinese-bert-wwm-ext',
	        'hfl/chinese-bert-wwm',
            'nghuyong/ernie-1.0-base-zh']

process = None

for pretrained_model_name in pretrained_model_names:
    for lr in lstm_lr:
        for b_lr in bert_lr:
            python_command = deepcopy(command)
            if b_lr == LR_FREEZE:
                python_command.extend(["--bert_lr", "0.0"])
            elif b_lr == LR_SAME:
                python_command.extend(["--bert_lr", str(lr)])
            else:
                if 'bert' in pretrained_model_name:
                    python_command.extend(["--bert_lr", str(MODEL_LR['bert'])])
                else:
                    python_command.extend(["--bert_lr", str(MODEL_LR['ernie'])])
            python_command.extend(["--lstm_lr", str(lr)])
            python_command.extend(["--crf_lr", str(lr)])
            python_command.extend(["--pretrained_model_name", pretrained_model_name])
            python_command = env_command + " ".join(python_command)
            nohup_command = python_command + " > nohup/" + get_run_name(python_command) + ".log 2>&1 &"
            if process is None or process.poll() is not None:
                process = subprocess.Popen(nohup_command, shell=True)
