export CUDA_VISIBLE_DEVICES=4
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python train.py \
    --search \
    --run_name nolstm \
    --epoch 10 \
    --batch_size 4 \
    --bert_lr 3e-5 \
    --lr 5e-3 \
    --optimizer adamw \
    --scheduler onecycle \
    --pretrained_model_name 'ckiplab/bert-base-chinese-ner' \
    --lstm_state_dim 256 \
