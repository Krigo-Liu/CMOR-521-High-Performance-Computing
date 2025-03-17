# Compiler and flags
CXX := g++
OMPFLAGS := -fopenmp

# Directories
SRC_DIR := src
INCLUDE_DIR := include

# Source files for AXPY
SRCS_AXPY := $(wildcard $(SRC_DIR)/axpy_omp.cpp)
MAIN_AXPY := $(wildcard ./main_axpy.cpp)

# Source file for Cache Block Matrix Multiplication
SRCS_BLOCK_MATMUL := $(wildcard $(SRC_DIR)/cache_block_matmul.cpp)
MAIN_BLOCK_MATMUL := $(wildcard ./main_matmul_block.cpp)

# Source file for Recursive Matrix Multiplication
SRCS_RECURSIVE_MATMUL := $(wildcard $(SRC_DIR)/recursive_matmul.cpp)
MAIN_RECURSIVE_MATMUL := $(wildcard ./main_matmul_recursive.cpp)

# Default target
all: main_axpy main_matmul_block main_matmul_recursive

# Build main_axpy executable
main_axpy: $(MAIN_AXPY) $(SRCS_AXPY)
	$(CXX) $(OMPFLAGS) -I$(INCLUDE_DIR) $(MAIN_AXPY) $(SRCS_AXPY) -o $@

main_matmul_block: $(MAIN_BLOCK_MATMUL) $(SRCS_BLOCK_MATMUL)
	$(CXX) $(OMPFLAGS) -I$(INCLUDE_DIR) $(MAIN_BLOCK_MATMUL) $(SRCS_BLOCK_MATMUL) -o $@

main_matmul_recursive: $(MAIN_RECURSIVE_MATMUL) $(SRCS_RECURSIVE_MATMUL)
	$(CXX) $(OMPFLAGS) -I$(INCLUDE_DIR) $(MAIN_RECURSIVE_MATMUL) $(SRCS_RECURSIVE_MATMUL) -o $@

# Clean build artifacts
clean:
	rm -f main_axpy main_matmul_block main_matmul_recursive
