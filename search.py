import subprocess, time
from copy import deepcopy
def get_run_name(command):
    command = command.split(" ")
    pretrained_model_name = command[command.index("--pretrained_model_name") + 1].split("/")[-1]
    bert_lr = command[command.index("--bert_lr") + 1]
    lstm_lr = command[command.index("--lstm_lr") + 1]
    opt = command[command.index("--optimizer") + 1]
    return f"{pretrained_model_name}_bertlr={bert_lr}_lstmlr={lstm_lr}_opt={opt}"
env_command = \
    "TOKENIZERS_PARALLELISM=False" + \
    "http_proxy=127.0.0.1:7890 " + \
    "https_proxy=127.0.0.1:7890 "

command = [
    "python", "train.py",
    "--epoch", "10",
    "--lstm_state_dim", "256",
    "--lstm_layer_num", "1",
]

LR_FREEZE = 0
LR_SAME = -1
LR_MODEL = -2

MODEL_LR = {'ernie': 6e-5, 'bert': 3e-5}
BATCH_SIZE = {'big': 24, 'small': 32}

lstm_lr = [1e-2, 1e-3, 1e-4, 1e-5]
bert_lr = [LR_FREEZE, LR_SAME, LR_MODEL]
optimizer = ['sgd', 'adamw', 'adam']
pretrained_model_names = [
            'hfl/chinese-roberta-wwm-ext-large',
	        'hfl/chinese-roberta-wwm-ext',
            'bert-base-chinese',
            'hfl/chinese-bert-wwm-ext',
	        'hfl/chinese-bert-wwm',
            'nghuyong/ernie-1.0-base-zh']

available_gpus = [4,5,6,7]
process = {gpu: None for gpu in available_gpus}

for pretrained_model_name in pretrained_model_names:
    for opt in optimizer:
        for lr in lstm_lr:
            for b_lr in bert_lr:
                python_command = deepcopy(command)
                python_command.extend(["--optimizer", opt])
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
                if 'large' in pretrained_model_name:
                    python_command.extend(["--batch_size", str(BATCH_SIZE['big'])])
                else:
                    python_command.extend(["--batch_size", str(BATCH_SIZE['small'])])
                final_command = env_command + " ".join(python_command)
                log_file = f"logs/{get_run_name(final_command)}.log"
                final_command += f" > {log_file} "
                # loop until there is an available gpu
                found = False
                while True:
                    for gpu in available_gpus:
                        if process[gpu] is None or process[gpu].poll() is not None:
                            process[gpu] = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} {final_command}", shell=True)
                            found = True
                            break
                    if found:
                        break
                    else:
                        time.sleep(60)
                        
