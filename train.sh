export CUDA_VISIBLE_DEVICES=4
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python train.py \
    --run_name set_transition \
    --epoch 50 \
    --bert_lr 0 \
    --lr 5e-3 \
    --lstm_state_dim 512 \
    --batch_size 64 \
    --pretrained_model_name 'ckiplab/bert-base-chinese-ner' \
    --optimizer adamw \
    --scheduler anneal \
