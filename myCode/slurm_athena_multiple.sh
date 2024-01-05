#!/bin/bash
#SBATCH -A plgdyplomancipw-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 2880
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH -J "noise rehersal"

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#!/bin/bash
module load Miniconda3/4.9.2

#conda activate /net/tscratch/people/plgjsobolewski/conda_env

/net/tscratch/people/plgjsobolewski/conda_env/bin/python run_experiment.py project="rehersal Alexnet MNIST Task IL CL" setup="taskIL" learning_rate="0.001" contrastive_learning_rate="0.00001" architecture="SmallAlexNet" dataset="CIFAR10" rehearsal_dataset="stat-wmm" epochs="5" contrastive_epochs="10" batch_size_rehearsal="128" batch_size="128" pretraining="true" num_classes="10" classes_per_task="2" optimizer="SGD" loss="CE" &

/net/tscratch/people/plgjsobolewski/conda_env/bin/python run_experiment.py project="rehersal Alexnet MNIST Task IL CL" setup="taskIL" learning_rate="0.001" contrastive_learning_rate="0.000001" architecture="SmallAlexNet" dataset="CIFAR10" rehearsal_dataset="stat-wmm" epochs="5" contrastive_epochs="10" batch_size_rehearsal="128" batch_size="128" pretraining="true" num_classes="10" classes_per_task="2" optimizer="SGD" loss="CE" &

/net/tscratch/people/plgjsobolewski/conda_env/bin/python run_experiment.py project="rehersal Alexnet MNIST Task IL CL" setup="taskIL" learning_rate="0.001" contrastive_learning_rate="0.0000001" architecture="SmallAlexNet" dataset="CIFAR10" rehearsal_dataset="stat-wmm" epochs="5" contrastive_epochs="10" batch_size_rehearsal="128" batch_size="128" pretraining="true" num_classes="10" classes_per_task="2" optimizer="SGD" loss="CE"
