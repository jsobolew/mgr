import os

def slurm_file_content(rehearsal_dataset):
    return f"""#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --exclude=titan2
#SBATCH --gpus=1 # number of GPUs required for the job
#SBATCH -J "noise rehersal"   # job name
#SBATCH --output=/data/Slurm-SHARE/jsobolewski/%j.out
#SBATCH --cpus-per-task 8

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

eval "$(conda shell.bash hook)"
conda activate myenv #first you need to create conda env with all requirements (see: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

python run_experiment.py rehearsal_dataset={rehearsal_dataset}
"""

if __name__ == "__main__":

    rehearsal_datasets = ['dead_leaves-mixed', 'dead_leaves-oriented', 'dead_leaves-squares',
    'dead_leaves-textures', 'stat-spectrum', 'stat-spectrum_color',
    'stat-spectrum_color_wmm', 'stat-wmm', 'stylegan-highfreq',
    'stylegan-random']

    repetitions = 5

    for rehearsal_dataset in rehearsal_datasets:
        for _ in range(repetitions):
            f = open("tmp_slurm_file.sh", "w")
            f.write(slurm_file_content(rehearsal_dataset))
            f.close()

            cmd = 'sbatch tmp_slurm_file.sh'
            os.system(cmd)
