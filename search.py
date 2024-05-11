import subprocess, time, os
from copy import deepcopy
def get_run_name(command):
    command = command.split(" ")
    pretrained_model_name = command[command.index("--pretrained_model_name") + 1].split("/")[-1]
    bert_lr = command[command.index("--bert_lr") + 1]
    lstm_lr = command[command.index("--lstm_lr") + 1]
    opt = command[command.index("--optimizer") + 1]
    lstm_state_dim = command[command.index("--lstm_state_dim") + 1]
    return f"{pretrained_model_name}_bertlr={bert_lr}_lstmlr={lstm_lr}_opt={opt}_dim={lstm_state_dim}"

def already_run(log_file_path):
    if os.path.exists(log_file_path):
        return True
    return False

env_command = \
    "TOKENIZERS_PARALLELISM=False " + \
    "http_proxy=127.0.0.1:7890 " + \
    "https_proxy=127.0.0.1:7890 "

command = [
    "python", "train.py",
    "--epoch", "20",
    "--lstm_layer_num", "1",
    "--search"
]

LR_FREEZE = 0
LR_SAME = -1
LR_MODEL = -2

MODEL_LR = {'ernie': 6e-5, 'bert': 3e-5}
BATCH_SIZE = {'big': 24, 'small': 24}

available_gpus = [4,5,6,7]
process = {gpu: None for gpu in available_gpus}

def vast():
    lstm_lr = [1e-3, 5e-4, 5e-3]
    bert_lr = [LR_FREEZE, LR_MODEL]
    optimizer = ['adamw']
    pretrained_model_names = [
                'hfl/chinese-roberta-wwm-ext-large',
            ]
    lstm_state_dims = [1024, 512, 256]

    for pretrained_model_name in pretrained_model_names:
        for opt in optimizer:
            for b_lr in bert_lr:
                for lstm_state_dim in lstm_state_dims:
                    for lr in lstm_lr:
                        python_command = deepcopy(command)
                        python_command.extend(["--optimizer", opt])
                        python_command.extend(["--lstm_state_dim", str(lstm_state_dim)])
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
                        final_command += f" > {log_file} 2>&1"
                        # loop until there is an available gpu
                        found = False
                        while True:
                            for gpu in available_gpus:
                                if process[gpu] is None or process[gpu].poll() is not None:
                                    process[gpu] = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} {final_command}", shell=True)
                                    found = True
                                    print(f"Running {final_command} on GPU {gpu}", flush=True)
                                    break
                            if found:
                                break
                            else:
                                time.sleep(60)

if __name__ == "__main__":
    vast()
