#!/bin/bash

#SBATCH --time=20:00:00   # walltime
#SBATCH --exclude=titan2
#SBATCH --gpus=1 # number of GPUs required for the job
#SBATCH -J "noise rehersal"   # job name
#SBATCH --output=/data/Slurm-SHARE/jsobolewski/%j.out
#SBATCH --cpus-per-task 8

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

eval "$(conda shell.bash hook)"
conda activate myenv #first you need to create conda env with all requirements (see: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

srun python run_experiment.py project="rehersal ResNet CIFAR10 Task IL" setup="taskIL" learning_rate="0.003" architecture="ResNet18" dataset="CIFAR10" rehearsal_dataset="stylegan-random" epochs="5" batch_size_rehearsal="128" batch_size="128" pretraining="True" num_classes="10" classes_per_task="2" optimizer="SGD" &
srun python run_experiment.py project="rehersal ResNet CIFAR10 Task IL" setup="taskIL" learning_rate="0.003" architecture="ResNet18" dataset="CIFAR10" rehearsal_dataset="stylegan-random" epochs="5" batch_size_rehearsal="128" batch_size="128" pretraining="True" num_classes="10" classes_per_task="2" optimizer="SGD" 
wait
