import subprocess, time, os
from copy import deepcopy
from itertools import product
from src.variable import get_run_name
from dataclasses import dataclass

env_command = \
    "TOKENIZERS_PARALLELISM=False " + \
    "http_proxy=127.0.0.1:7890 " + \
    "https_proxy=127.0.0.1:7890 "

batch_size = 12
command = [
    "python", "train.py",
    "--epoch", "10",
    "--batch_size", str(batch_size),
    "--search"
]

LR_FREEZE = 0
LR_MODEL = -2

MODEL_LR = {'ernie': 6e-5, 'bert': 3e-5}

available_gpus = [4,5,6,7]
process = {gpu: None for gpu in available_gpus}

@dataclass
class Hparams:
    pretrained_model_name: str
    optimizer: str
    lstm_state_dim: int
    bert_lr: float
    lr: float
    scheduler: str

def vast():
    lr = [1e-3, 5e-4, 5e-3]
    bert_lr = [LR_FREEZE]
    optimizer = ['adamw']
    pretrained_model_names = [
                'ckiplab/bert-base-chinese-ner'
            ]
    lstm_state_dims = [1024, 512, 256]
    scheduler = ['linear', 'anneal', 'onecycle', 'none']
    params = product(pretrained_model_names, optimizer, bert_lr, lstm_state_dims, lr, scheduler)

    for pretrained_model_name, opt, b_lr, lstm_state_dim, lr, scheduler in params:
        python_command = deepcopy(command)
        python_command.extend(["--optimizer", opt])
        python_command.extend(["--lstm_state_dim", str(lstm_state_dim)])
        if b_lr == LR_FREEZE:
            python_command.extend(["--bert_lr", "0.0"])
        else:
            if 'bert' in pretrained_model_name:
                python_command.extend(["--bert_lr", str(MODEL_LR['bert'])])
            else:
                python_command.extend(["--bert_lr", str(MODEL_LR['ernie'])])
        python_command.extend(["--lr", str(lr)])
        python_command.extend(["--pretrained_model_name", pretrained_model_name])
        python_command.extend(["--scheduler", scheduler])
        
        hparams = Hparams(pretrained_model_name, opt, lstm_state_dim, b_lr, lr, scheduler)

        run_name = get_run_name(hparams)
        final_command = env_command + " ".join(python_command)
        log_file = f"logs/{run_name}.log"

        if os.path.exists(log_file):
            print("Skip ", run_name, flush=True)
            continue

        final_command += f" > {log_file} 2>&1"

        found = False
        while True:
            for gpu in available_gpus:
                if process[gpu] is None or process[gpu].poll() is not None:
                    if process[gpu] is not None and process[gpu].poll() != 0:
                        print(f"Error in {final_command} on GPU {gpu}")
                        exit(1)
                    process[gpu] = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} {final_command} > nohup.log", shell=True)
                    found = True
                    print(f"Running {final_command} on GPU {gpu}", flush=True)
                    break
            if found:
                break
            else:
                time.sleep(60)

if __name__ == "__main__":
    vast()
