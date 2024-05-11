export CUDA_VISIBLE_DEVICES=7
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python val.py \
    --path best_models/epoch=36-val_loss=0.10-val_f1=0.47.ckpt \
