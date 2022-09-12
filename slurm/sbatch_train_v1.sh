#!/bin/bash                      
#SBATCH -t 50:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 #  number of GPUs
#SBATCH --constraint=high-capacity
#SBATCH --array=24
#SBATCH -D /om2/user/jangh/DeepLearning/RobustFaceRecog/logs/v1/

cd /om2/user/jangh/
hostname
date "+%y/%m/%d %H:%M:%S"
echo $CUDA_VISIBLE_DEVICES
module load openmind/singularity/3.4.1

singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
python DeepLearning/RobustFaceRecog/train_v1.py \
--is_slurm=True \
--id=${SLURM_ARRAY_TASK_ID} \

#singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
#python DeepLearning/RobustFaceRecog/analysis_v1_accuracy_within_across_category.py \
#--is_slurm=True \
#--id=${SLURM_ARRAY_TASK_ID} \

#singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
#python DeepLearning/RobustFaceRecog/analysis_v1_invariance_coefficient.py \
#--is_slurm=True \
#--id=${SLURM_ARRAY_TASK_ID} \