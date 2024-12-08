#!/bin/bash


# This is a wrapper script to submit a python script as a job to SLURM. It takes two arguments,
# which are passed to the SLURM script as environment variables:
# 1. Path to the python file
# 2. Conda environment name


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_python_file> <conda_env_name>"
    exit 1
fi

PYTHON_FILE=$1
CONDA_ENV=$2

echo "Submitting job to SLURM..."
echo "Python file: $PYTHON_FILE"
echo "Conda environment: $CONDA_ENV"
echo ""

sbatch --export=PYTHON_FILE="$PYTHON_FILE",CONDA_ENV="$CONDA_ENV" scripts/slurm_run_python_script.sh
