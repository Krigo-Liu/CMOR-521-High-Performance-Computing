# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17
LDFLAGS := 

# Directories
SRC_DIR := src
OBJ_DIR := obj
RESULT_DIR := result

# Source files for both programs
SRCS_TRANSPOSE := $(SRC_DIR)/matrix_transpose.cpp main_transpose.cpp
SRCS_MULTIPLICATION := $(SRC_DIR)/matrix_multiplication.cpp main_multiplication.cpp

# Object files (ensure all .o files are placed in obj/)
OBJS_TRANSPOSE := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(SRCS_TRANSPOSE)))
OBJS_MULTIPLICATION := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(SRCS_MULTIPLICATION)))

# Ensure obj directory exists before compiling
$(shell mkdir -p $(OBJ_DIR))

# Default target: build all programs
all: matrix_transpose_02 matrix_transpose_03 matrix_multiplication_02 matrix_multiplication_03

# Rules for matrix_transpose
matrix_transpose_02: CXXFLAGS += -O2
matrix_transpose_02: $(OBJS_TRANSPOSE)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

matrix_transpose_03: CXXFLAGS += -O3
matrix_transpose_03: $(OBJS_TRANSPOSE)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Rules for matrix_multiplication
matrix_multiplication_02: CXXFLAGS += -O2
matrix_multiplication_02: $(OBJS_MULTIPLICATION)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

matrix_multiplication_03: CXXFLAGS += -O3
matrix_multiplication_03: $(OBJS_MULTIPLICATION)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile object files (ensuring obj directory exists)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/main_transpose.o: main_transpose.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/main_multiplication.o: main_multiplication.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure obj directory exists before compilation
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean build artifacts
clean:
	rm -rf $(RESULT_DIR) $(OBJ_DIR) matrix_transpose_02 matrix_transpose_03 matrix_multiplication_02 matrix_multiplication_03 matrix_transpose_col

.PHONY: all clean
