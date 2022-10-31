#!/bin/bash                      
#SBATCH -t 12:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 #  number of GPUs
#SBATCH --constraint=high-capacity
#SBATCH --mem=32GB
#SBATCH -p sinha
#SBATCH --array=1
#SBATCH -D /om2/user/jangh/DeepLearning/RobustFaceRecog/logs/v5/

cd /om2/user/jangh/
hostname
date "+%y/%m/%d %H:%M:%S"

module load openmind/singularity/3.4.1
module add openmind/cuda/11.3
module add openmind/cudnn/11.5-v8.3.3.40

singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9b.simg \
python DeepLearning/RobustFaceRecog/analysis_v5_accuracy_by_blur.py \
--is_slurm=True \
--job=${SLURM_ARRAY_JOB_ID} \
--id=${SLURM_ARRAY_TASK_ID} \
