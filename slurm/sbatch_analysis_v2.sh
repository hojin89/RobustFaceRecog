#!/bin/bash                      
#SBATCH -t 03:00:00
#SBATCH -n 1 # number of tasks to be launched (can't be smaller than -N)
#SBATCH -c 4 # number of CPU cores associated with one GPU
#SBATCH --gres=gpu:1 #  number of GPUs
#SBATCH --constraint=high-capacity
##SBATCH --mem=32GB
#SBATCH --array=0-6
#SBATCH -D /om2/user/jangh/DeepLearning/RobustFaceRecog/logs/v2/

cd /om2/user/jangh/
hostname
date "+%y/%m/%d %H:%M:%S"
echo $CUDA_VISIBLE_DEVICES
module load openmind/singularity/3.4.1

singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
python DeepLearning/RobustFaceRecog/analysis_v2_accuracy_within_across_category_by_blur.py \
--is_slurm=True \
--id=${SLURM_ARRAY_TASK_ID} \

#singularity exec --nv -B /om,/om2  /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
#python DeepLearning/RobustFaceRecog/analysis_v2_accuracy_within_across_category_by_turbulence_backup.py \
#--is_slurm=True \
#--id=${SLURM_ARRAY_TASK_ID} \