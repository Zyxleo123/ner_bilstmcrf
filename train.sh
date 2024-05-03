export CUDA_VISIBLE_DEVICES=5
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python train.py \
    --epoch 100 \
    --bert_lr 1e-2 \
    --crf_lr 1e-2 \
    --lstm_lr 1e-2 \
    --lstm_state_dim 256 \
    --lstm_layer_num 1 \
    --batch_size 32 \

