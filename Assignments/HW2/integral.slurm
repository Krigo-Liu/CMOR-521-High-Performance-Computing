#!/bin/bash
#SBATCH --job-name=Integral
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
echo "The current job is: Integral"
cd /home/yl336/cmor521/Assignment2 || exit
srun ./main_integral 