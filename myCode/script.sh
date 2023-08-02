#!/bin/bash
source .env

python_ref=$PYTHON_INTERPRETER

rehearsal_datasets=('dead_leaves-mixed' 'dead_leaves-oriented' 'dead_leaves-squares' 'dead_leaves-textures' 'stat-spectrum' 'stat-spectrum_color' 'stat-spectrum_color_wmm' 'stat-wmm' 'stylegan-highfreq' 'stylegan-random')

for dataset in "${rehearsal_datasets[@]}"
do
  for i in 1 2 3 4 5
  do
    command="${python_ref} run_experiment.py rehearsal_dataset=${dataset}"
    
    # Slurm
    # bash "scheludle_slurm.sh ${command}"

    # local
    echo $command
    $command
  done
done
