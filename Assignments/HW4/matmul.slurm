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

nvcc -o matmul_v1 matmul_v1.cu
nvcc -o matmul_v2 matmul_v2.cu
nvcc -o matmul_v3_32 matmul_v3_32.cu
nvcc -o matmul_v3_64 matmul_v3_64.cu



echo "=== BLOCKSIZE = 32 ==="

echo "=== Running Matrix Multiplication Version 1 (N = 2048) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v1 2048

echo "=== Running Matrix Multiplication Version 2 (N = 2048) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v2 2048

echo "=== Running Matrix Multiplication Version 3 (N = 2048) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v3_32 2048

echo "=== Running Matrix Multiplication Version 1 (N = 4096) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v1 4096

echo "=== Running Matrix Multiplication Version 2 (N = 4096) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v2 4096

echo "=== Running Matrix Multiplication Version 3 (N = 4096) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v3_32 4096

echo "=== BLOCKSIZE = 64 ==="

echo "=== Running Matrix Multiplication Version 1 (N = 2048) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v1 2048

echo "=== Running Matrix Multiplication Version 2 (N = 2048) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v2 2048

echo "=== Running Matrix Multiplication Version 3 (N = 2048) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v3_64 2048

echo "=== Running Matrix Multiplication Version 1 (N = 4096) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v1 4096

echo "=== Running Matrix Multiplication Version 2 (N = 4096) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v2 4096

echo "=== Running Matrix Multiplication Version 3 (N = 4096) ==="
nvprof --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./matmul_v3_64 4096

echo "All runs complete."