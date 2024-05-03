export CUDA_VISIBLE_DEVICES=7
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

python train.py --fast_dev_run --toy
