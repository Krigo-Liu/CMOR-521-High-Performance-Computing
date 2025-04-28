#!/bin/bash
#SBATCH --job-name=GPU_Interactive
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --reservation=cmor421
#SBATCH --ntasks=1                # 1 task
#SBATCH --cpus-per-task=1          # 1 CPU per task
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=5G                   # 5GB total memory
#SBATCH --time=00:30:00            # 30 minutes
#SBATCH --mail-user=yl336@rice.edu
#SBATCH --mail-type=END,FAIL

# Launch an interactive shell
$SHELL

echo "Job running on nodes: $SHELL"
echo "================================================="

cd /home/yl336/cmor521/CMOR-521-High-Performance-Computing/Assignments/HW4 || exit

# Load correct CUDA environment
module load GCC/12.3.0 CUDA/12.1.1

nvcc -o matmul matmul.cu


echo "=== Running Matrix Multiplication Version 1 (N = 2048) ==="
nvprof ./matmul 2048 1
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul 2048 1

echo "=== Running Matrix Multiplication Version 2 (N = 2048) ==="
nvprof ./matmul 2048 2
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul 2048 2

echo "=== Running Matrix Multiplication Version 3 (N = 2048) ==="
nvprof ./matmul 2048 3
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul 2048 3

echo "=== Running Matrix Multiplication Version 1 (N = 4096) ==="
nvprof ./matmul 4096 1
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul 4096 1

echo "=== Running Matrix Multiplication Version 2 (N = 4096) ==="
nvprof ./matmul 4096 2
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul 4096 2

echo "=== Running Matrix Multiplication Version 3 (N = 4096) ==="
nvprof ./matmul 4096 3
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul 4096 3

echo "All runs complete."