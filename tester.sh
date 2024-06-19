#!/usr/bin/bash
#SBATCH -J tester
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --constraint=rocky8
#SBATCH --mem 40G
#SBATCH -o outputs/tester_%j.out

source ~/.bashrc
conda activate pytorch

python -u src/bioplnn/trainers/ei_tester.py data.mode=shape data.holdout=[] model.modulation_type=ag checkpoint.run=whole-frost-113