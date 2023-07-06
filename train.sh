python run.py \
 --config configs/config_diffusion_cifar10.py \
 --mode train \
 --model_dir results/cifar10 \
 --config.train.checkpoint_epochs 5 \
 --config.train.keep_checkpoint_max 2