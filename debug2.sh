rm -rf results/debug

export WANDB_MODE=disabled

python -m debugpy --listen 0.0.0.0:39203 \
    run.py \
    --run_eagerly True \
    --config configs/config_diffusion_cifar10.py \
    --mode train \
    --model_dir results/debug \
    --config.train.checkpoint_steps 4 \
    --config.train.keep_checkpoint_max 1 \
    --config.train.batch_size 16 \
    --config.train.steps 1 \
    --config.model.drop_units 0.0 \
    --config.model.drop_path 0.0
