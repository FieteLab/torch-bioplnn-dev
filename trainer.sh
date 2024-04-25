#!/usr/bin/bash
#SBATCH -J trainer
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --constraint=rocky8
#SBATCH --mem 20G
#SBATCH -o trainer.out

source ~/.bashrc
conda activate pytorch

python -u src/bioplnn/trainers/ei.py --config config/config_ei.yaml