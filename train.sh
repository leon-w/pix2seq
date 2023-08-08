python run.py \
 --config configs/config_diffusion_cifar10.py \
 --mode train \
 --model_dir results/cifar10_new \
 --config.train.checkpoint_steps 1000 \
 --config.train.keep_checkpoint_max 1
