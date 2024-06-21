#!/usr/bin/bash
#SBATCH -J trainer
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --constraint=rocky8
#SBATCH --mem 40G
#SBATCH -o outputs/trainer_%j.out

source ~/.bashrc
conda activate pytorch

python -u src/bioplnn/trainers/crnn_ei_qclevr.py data.mode=conjunction data.holdout=[cube_green,cube_blue] model.modulation_type=ag optimizer.lr=0.0004 criterion.all_timesteps=False model.immediate_inhibition=True model.exc_rectify=pos model.layer_time_delay=True