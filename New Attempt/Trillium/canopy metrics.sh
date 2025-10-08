#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --job-name=canopy_metrics
#SBATCH --output=/scratch/arbmarta/Canopy_Points/Outputs/canopy_metrics.out
#SBATCH --error=/scratch/arbmarta/Canopy_Points/Outputs/canopy_metrics.err

# Activate virtual environment
source /home/arbmarta/.virtualenvs/myenv/bin/activate

# Run canopy modeling script
python /scratch/arbmarta/Code/Canopy_Points/canopy_metrics.py
