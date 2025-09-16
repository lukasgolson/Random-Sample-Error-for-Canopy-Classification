#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --job-name=NLM
#SBATCH --output=/scratch/arbmarta/NLM/NLM.out
#SBATCH --error=/scratch/arbmarta/NLM/NLM.err

# Run your Python script
source /home/arbmarta/.virtualenvs/myenv/bin/activate
python /scratch/arbmarta/NLM/NLM.py
