export CUDA_VISIBLE_DEVICES=7
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python test.py \
    --path best_models/set_transition/epoch=26-val_loss=0.0315-val_f1=0.8999.ckpt