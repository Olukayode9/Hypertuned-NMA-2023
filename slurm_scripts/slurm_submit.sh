#!/bin/bash
##### These lines are for Slurm
#SBATCH --output='<out_path>'
#SBATCH --error='<err_path>'
#SBATCH -c 1 # number of threads per process
#SBATCH -n 3 # number of processes
#SBATCH -A asari
#SBATCH --job-name=NMA
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=<mail>
#SBATCH --qos=normal
#SBATCH --mem=128000 # memory/RAM given in MB
#SBATCH -t 72:45:00 # walltime in hours


module purge;
module load Anaconda3/2022.10
conda activate nma

python sample_shuffled_linear_model.py
