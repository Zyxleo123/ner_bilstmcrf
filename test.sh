export CUDA_VISIBLE_DEVICES=7
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python test.py \
    --path best_models/order/epoch=2-val_loss=0.0340-val_f1=0.8879.ckpt