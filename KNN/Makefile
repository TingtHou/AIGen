
# -----------------------------------------------------------------
#   Makefile for KNN 
#   
#   Supported platforms: Unix / Linux
# ---------------------------------------------------------------------

# Directory of the target
OUTPUT = KNN

RM = rm
# Compiler
CXX =  g++
NVCC=  nvcc

# EIGEN library
EIGEN_PATH = /usr/include/Eigen

# Intel MKL library
MKLROOT = /opt/intel/mkl


## 
CUDACPP=$(SRC_DIR)/cuimnq.cpp $(SRC_DIR)/cuimnq.cpp $(SRC_DIR)/cuMINQUE1.cpp $(SRC_DIR)/cuMinqueBase.cpp $(SRC_DIR)/cuToolkit.cpp
# Compiler flags
CXXFLAGS =  -w -O3 -m64 -static -fopenmp -I $(EIGEN_PATH) -DEIGEN_NO_DEBUG -I $(MKLROOT)/include -I /usr/include/boost 
#CUDAFLAGS= -std=c++11 -c -arch=sm_30

LIB +=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/compilers_and_libraries/linux/lib/intel64 -liomp5  -lpthread -lm -ldl
LIB+=  -lboost_program_options
SRC_DIR=src
OBJ_DIR=obj

SRC =$(filter-out $(CUDACPP), $(wildcard $(SRC_DIR)/*.cpp))
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)



CPPFLAGS += -Iinclude
CFLAGS += -Wall

.PHONY: all clean

all : $(OBJ_DIR) $(OUTPUT) clean

$(OUTPUT) : $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LIB)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS) -c $^ -o $@

clean: 
	rm -rf $(OBJ_DIR)
	 
$(OBJ_DIR):
	mkdir -p $(notdir $@)
