#!/bin/bash
#SBATCH --job-name=SUMMA
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=16               # allocate up to 16 MPI ranks
#SBATCH --cpus-per-task=1         # one core per rank
#SBATCH --mem-per-cpu=16000        # 2 GB per rank (adjust as needed)
#SBATCH --time=00:30:00
#SBATCH --mail-user=yl336@rice.edu
#SBATCH --mail-type=END,FAIL

echo "Job running on nodes: $SLURM_NODELIST"
echo "================================================="

cd /home/yl336/cmor521/CMOR-521-High-Performance-Computing/Assignments/HW3 || exit

# Load correct MPI environment
module load GCC/13.2.0
module load OpenMPI/4.1.6

mpic++ -O3 -o summa_mpi main_SUMMA.cpp ./src/matrixMul.cpp

echo "=== Running SUMMA on 4 procs (2×2 grid) ==="
mpirun -n 4  ./summa_mpi 512 64

echo "=== Running SUMMA on 9 procs (3×3 grid) ==="
mpirun -n 9  ./summa_mpi 768 64

echo "=== Running SUMMA on 16 procs (4×4 grid) ==="
mpirun -np 16 ./summa_mpi 1024 64

echo "All runs complete."
