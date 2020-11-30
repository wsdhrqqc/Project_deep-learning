#!/bin/bash
#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
#SBATCH --output=results/xor_%J_stdout.txt
#SBATCH --error=results/xor_%J_stderr.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=xor_test
#SBATCH --mail-user=newqingslu@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/wsdhr/dml_hw2
#SBATCH --array=0-2
#
#################################################
source ~fagg/pythonenv/tensorflow/bin/activate
python xor_base.py --epochs 1000 --exp $SLURM_ARRAY_TASK_ID


