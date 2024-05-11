export CUDA_VISIBLE_DEVICES=7
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python test.py \
    --path best_models/chinese-roberta-wwm-ext-large_bertlr=3e-05_lstmlr=0.005_opt=adamw_dim=256_anneal=False/epoch=12-val_loss=0.0921-val_f1=0.4804.ckpt
