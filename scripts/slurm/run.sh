#!/bin/bash

#SBATCH --job-name=run_python_script
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode04
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=12G


mkdir -p slurm_logs/output slurm_logs/error

module load miniconda3

conda run -p /home/3144860/.conda/envs/ml python scripts/train_ae.py --multirun dataset=mnist,fashion_mnist,cifar10,cifar100 model.config.latent_dim=64,32,16,8,4 name=conv-ae seed=3
