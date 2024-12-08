#!/bin/bash

#SBATCH --job-name=run_python_script
#SBATCH --output=slurm/output/output_%x_%j.txt
#SBATCH --error=slurm/error/error_%x_%j.txt
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode03
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


# Description: This script is used to run a Python script with a conda environment in SLURM.
# It should be submitted to slurm from the root directory of the project, using sbatch.
# To handle arguments, they must be passed as environment variables using --export.
# Example: sbatch --export=PYTHON_FILE=path_to_python_file,CONDA_ENV=conda_env_name scripts/slurm_run_python_script.sh

# Check if the required environment variables are set
if [ -z "$PYTHON_FILE" ] || [ -z "$CONDA_ENV" ]; then
    echo "Error: Environment variables PYTHON_FILE and CONDA_ENV must be set."
    echo "Usage: sbatch --export=PYTHON_FILE=path_to_python_file,CONDA_ENV=conda_env_name scripts/slurm_run_python_script.sh"
    exit 1
fi

# Create the output directories
mkdir -p slurm/output slurm/error

# Load the conda module
module load miniconda3

# Activate the conda environment and run the Python script
conda run -n "$CONDA_ENV" python "$PYTHON_FILE"
