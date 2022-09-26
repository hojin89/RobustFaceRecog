#!/bin/bash                      
#SBATCH -t 72:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 #  number of GPUs
#SBATCH --constraint=high-capacity
#SBATCH --array=1-36
#SBATCH -D /om2/user/jangh/DeepLearning/RobustFaceRecog/logs/v3/

cd /om2/user/jangh/
hostname
date "+%y/%m/%d %H:%M:%S"
echo $CUDA_VISIBLE_DEVICES
module load openmind/singularity/3.4.1

singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
python DeepLearning/RobustFaceRecog/train_v3.py \
--is_slurm=True \
--job=${SLURM_ARRAY_JOB_ID} \
--id=${SLURM_ARRAY_TASK_ID} \
