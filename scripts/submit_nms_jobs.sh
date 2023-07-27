#!/bin/bash
#SBATCH --job-name=nms
#SBATCH --output=outputs/nms_%a.out
#SBATCH --error=outputs/nms_%a.err
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --array=1-165%33
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jmhicks@mit.edu
#SBATCH --gres=gpu:1

export CONDA_ENVS_PATH=~/my-envs:/om2/user/jmhicks/conda_envs_files
source activate /om2/user/jmhicks/conda_envs_files/neuromatch_similarity

python run.py --params "../params/model${SLURM_ARRAY_TASK_ID}_params.pt"
