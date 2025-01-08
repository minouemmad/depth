#!/bin/bash
#SBATCH --job-name=depth
#SBATCH --output=depth_output.log
#SBATCH --error=depth_error.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

module purge

singularity exec --nv \
    --overlay /scratch/mae9855/ContrastiveCrop/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /scratch/mae9855/venv/bin/activate &&
    python3.10 depth_shifts.py
    "
