#!/bin/bash
#SBATCH --job-name=Recursive_Matrix_Multiplication
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:30:00
#SBATCH --mail-user=yl336@rice.edu
##SBATCH --mail-type=END,FAIL
echo "My job ran on:"
echo $SLURM_NODELIST
echo "The current job is: Recursive Matrix Multiplication"
cd /home/yl336/cmor521/Assignment2 || exit
srun --ntasks=1 ./main_matmul_recursive strong
srun --ntasks=1 ./main_matmul_recursive weak

