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

/net/tscratch/people/plgjsobolewski/conda_env/bin/python run_experiment.py