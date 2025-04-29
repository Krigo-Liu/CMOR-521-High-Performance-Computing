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

nvcc -o matTranspose matTranspose.cu


echo "=== Running Matrix Transpose with both naive version and shared memoryversion (N = 2048) ==="
nvprof ./matTranspose 2048
nvprof --metrics gld_throughput,gst_throughput ./matTranspose 2048

echo "=== Running Matrix Transpose with both naive version and shared memoryversion (N = 4096) ==="
nvprof ./matTranspose 4096
nvprof --metrics gld_throughput,gst_throughput ./matTranspose 4096

echo "All runs complete."