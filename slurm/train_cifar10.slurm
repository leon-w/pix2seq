#!/bin/bash
#SBATCH --job-name=pix2seq_rin_cifar10
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --chdir=/usr/stud/wiemers/tmp/pix2seq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/stud/wiemers/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/stud/wiemers/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/stud/wiemers/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/stud/wiemers/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate pix2seq
python run.py --config configs/config_diffusion_cifar10.py --mode train --model_dir slurm/results/cifar10 --config.train.epochs=1000 --config.train.checkpoint_epochs=10