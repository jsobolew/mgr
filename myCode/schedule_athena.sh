#!/bin/bash
. .env

config=$1
n_times=${2:-1}
params_extract="python3 extract_params_from_config.py ${config}"
echo "$params_extract"
out=$($params_extract)

python_command="${PYTHON_INTERPRETER} run_experiment.py ${out}"
echo -e "Scheduling $n_times times:\n"$python_command""

python_commands=""
for (( i=0; i<n_times; i++ ))
do
    if  [ $i -eq $((n_times-1)) ]
    then
        python_commands="${python_commands} ${python_command}"
    else
        python_commands="${python_commands} ${python_command} & "
    fi
done

cat << EOT > tmp.sh
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

$python_commands

wait
EOT

sbatch tmp.sh
