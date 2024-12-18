#!/bin/bash

#SBATCH --job-name=run_python_script
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=00:10:00
#SBATCH --partition=debug_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode02
#SBATCH --qos=debug

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G


mkdir -p slurm_logs/output slurm_logs/error

module load miniconda3

conda run -p /home/3144860/.conda/envs/ml python scripts/train_ae.py --multirun training.epochs=1 seed=1,2
