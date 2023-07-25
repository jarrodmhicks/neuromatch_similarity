#!/bin/bash
#SBATCH --job-name=tpjnbgpu
#SBATCH --output=outputs/tpjnbgpu%j.out
#SBATCH --error=outputs/tpjnbgpu%j.err
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1

export CONDA_ENVS_PATH=~/my-envs:/om2/user/jmhicks/conda_envs_files
source activate /om2/user/jmhicks/conda_envs_files/neuromatch_similarity

port=1108
if [ "$1" != "" ]; then
port=$1
fi

user=$(whoami)
node=$(hostname -s)
if [ $HOSTNAME == ranvier ] &&  [ -z "$SLURM_CLUSTER_NAME" ]; then
cluster="ranvier.mit.edu"
else
cluster="openmind7.mit.edu"
fi

# Print instructions to create ssh tunnel
echo -e "
>>> Command to create ssh tunnel: <<<
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}
>>> Address for jupyter-notebook: <<<
localhost:${port}
"
# Launch jupyter notebook
unset XDG_RUNTIME_DIR
jupyter notebook --no-browser --port=${port} --ip=${node}