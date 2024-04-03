#!/usr/bin/bash
#SBATCH -J trainer
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --constraint=rocky8
#SBATCH --mem 80G
#SBATCH -o trainer.out

source ~/.bashrc
conda activate pytorch

python src/bioplnn/ei_trainer.py --config config/config_ei.yaml