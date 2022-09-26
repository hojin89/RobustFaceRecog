#!/bin/bash                      
#SBATCH -t 96:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 #  number of GPUs
##SBATCH --constraint=high-capacity
#SBATCH --constraint=24GB
#SBATCH --array=1-25
#SBATCH -D /om2/user/jangh/DeepLearning/RobustFaceRecog/logs/v4/

cd /om2/user/jangh/
hostname
date "+%y/%m/%d %H:%M:%S"
echo $CUDA_VISIBLE_DEVICES

module load openmind/singularity/3.4.1
module load openmind/cuda/11.2
module load openmind/cudnn/11.5-v8.3.3.40

#singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9b.simg \
singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
python DeepLearning/RobustFaceRecog/train_v4.py \
--is_slurm=True \
--job=${SLURM_ARRAY_JOB_ID} \
--id=${SLURM_ARRAY_TASK_ID} \
