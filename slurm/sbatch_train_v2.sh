#!/bin/bash                      
#SBATCH -t 60:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 8 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 #  number of GPUs
#SBATCH --constraint=high-capacity
#SBATCH --array=0-6
#SBATCH -D /om2/user/jangh/DeepLearning/RobustFaceRecog/logs/v2/

cd /om2/user/jangh/
hostname
date "+%y/%m/%d %H:%M:%S"
echo $CUDA_VISIBLE_DEVICES
module load openmind/singularity/3.4.1

singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
python DeepLearning/RobustFaceRecog/train_v2.py \
--is_slurm=True \
--id=${SLURM_ARRAY_TASK_ID} \
