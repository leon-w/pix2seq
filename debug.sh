rm -r results/debug

WANDB_MODE=disabled python -m debugpy --wait-for-client --listen 5678 run.py \
 --run_eagerly True \
 --config configs/config_diffusion_cifar10.py \
 --mode train \
 --model_dir results/debug \
 --config.train.checkpoint_steps 1 \
 --config.train.keep_checkpoint_max 1 \
 --config.train.batch_size 8 \
 --config.train.steps 1
