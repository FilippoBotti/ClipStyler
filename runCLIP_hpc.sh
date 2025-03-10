#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:a100_80g:1
#SBATCH --ntasks-per-node=28
#SBATCH --time 03:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 4
##SBATCH --cpus-per-task 10

#SBATCH --job-name=logs/CLIP-mamba
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --mail-user=teresa.calzetti@studenti.unipr.it
#SBATCH --mail-type=ALL


echo "####################"
echo  "#SLURM_JOB_NODELIST      : $SLURM_JOB_NODELIST"
echo  "#SLURM_CPUS_PER_TASK     : $SLURM_CPUS_PER_TASK"
echo  "#SLURM_JOB_CPUS_PER_NODE : $SLURM_JOB_CPUS_PER_NODE"
echo  "#OMP_NUM_THREADS			: $OMP_NUM_THREADS"
echo "####################"


##ssh wn45

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate mambast

python3 train_fast.py