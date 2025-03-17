#!/bin/bash
#SBATCH --job-name=AXPY
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=50000m
#SBATCH --time=00:30:00
#SBATCH --mail-user=yl336@rice.edu
##SBATCH --mail-type=ALL
echo "My job ran on:"
echo $SLURM_NODELIST
echo "The current job is: AXPY"
cd /home/yl336/cmor521/Assignment2 || exit
srun ./main_axpy