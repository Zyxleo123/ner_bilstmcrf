export CUDA_VISIBLE_DEVICES=6
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

# chinese-roberta-wwm-ext-large_bertlr=3e-05_lstmlr=0.005_opt=adamw_dim=256/version_2
python train.py \
    --epoch 50 \
    --bert_lr 3e-5 \
    --crf_lr 5e-3 \
    --lstm_lr 5e-3 \
    --lstm_state_dim 256 \
    --lstm_layer_num 1 \
    --batch_size 16 \
    --pretrained_model_name hfl/chinese-roberta-wwm-ext-large \
    --optimizer adamw \
    --upsample
