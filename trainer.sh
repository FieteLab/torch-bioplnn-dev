#!/usr/bin/bash
#SBATCH -J trainer
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --constraint=rocky8
#SBATCH --mem 30G
#SBATCH -o trainer.out

source ~/.bashrc
conda activate pytorch-3.10

python src/bioplnn/trainer.py --config config/config_random.yaml