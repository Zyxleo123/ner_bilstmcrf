export CUDA_VISIBLE_DEVICES=4
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python train.py \
    --epoch 50 \
    --bert_lr 0 \
    --crf_lr 5e-3 \
    --lstm_lr 5e-3 \
    --lstm_state_dim 256 \
    --lstm_layer_num 1 \
    --batch_size 8 \
    --pretrained_model_name 'ckiplab/bert-base-chinese-ner' \
    --optimizer adamw \
