#!/bin/bash
#SBATCH --job-name=CMOR421/521-HW3-CANNON
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=16               # allocate up to 16 MPI ranks
#SBATCH --cpus-per-task=1         # one core per rank
#SBATCH --mem-per-cpu=16000        # 16 GB per rank (adjust as needed)
#SBATCH --time=00:30:00
#SBATCH --mail-user=yl336@rice.edu
#SBATCH --mail-type=END,FAIL

echo "Job running on nodes: $SLURM_NODELIST"
echo "================================================="

cd /home/yl336/cmor521/CMOR-521-High-Performance-Computing/Assignments/HW3 || exit

# Load correct MPI environment
module load GCC/13.2.0
module load OpenMPI/4.1.6

mpic++ -O3 -o cannon_mpi main_CANNON.cpp ./src/matrixMul.cpp

echo "=== Running CANNON on 4 procs (2×2 grid) ==="
mpirun -n 4  ./cannon_mpi 512

echo "=== Running CANNON on 9 procs (3×3 grid) ==="
mpirun -n 9  ./cannon_mpi 768

echo "=== Running CANNON on 16 procs (4×4 grid) ==="
mpirun -np 16 ./cannon_mpi 1024

echo "All runs complete."
