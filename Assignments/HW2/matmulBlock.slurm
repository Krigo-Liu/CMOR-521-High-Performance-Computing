#!/bin/bash
#SBATCH --job-name=Cach_Block_Matrix_Multiplication
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:30:00
#SBATCH --mail-user=yl336@rice.edu
##SBATCH --mail-type=ALL
echo "My job ran on:"
echo $SLURM_NODELIST
echo "The current job is: Cach Block Matrix Multiplication"
cd /home/yl336/cmor521/Assignment2 || exit
srun ./main_compare_matmul_block
srun --ntasks=1 ./main_matmul_block strong
srun --ntasks=1 ./main_matmul_block weak
