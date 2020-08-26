
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
NVCC=  /usr/local/cuda-10.2/bin/nvcc

CPPFLAGS= -I include

# EIGEN library
EIGEN_PATH = /usr/include/eigen3

# Intel MKL library
MKLROOT = /opt/intel

# CUDA Library
CUDAROOT= /usr/local/cuda-10.2

## 
CUDACPP=$(SRC_DIR)/cuimnq.cpp $(SRC_DIR)/cuimnq.cpp $(SRC_DIR)/cuMINQUE1.cpp $(SRC_DIR)/cuMINQUE0.cpp $(SRC_DIR)/cuMinqueBase.cpp $(SRC_DIR)/cuToolkit.cpp
# Compiler flags
CXXFLAGS =  -w -O3 -m64 -fopenmp -static -I$(EIGEN_PATH) -DEIGEN_NO_DEBUG -I$(MKLROOT)/mkl/include -I/usr/include/boost  -L$(MKLROOT)/mkl/lib/intel64 -L$(MKLROOT)/compilers_and_libraries/linux/lib/intel64
CUDAFLAGS= -std=c++11 -c -arch=sm_30

LIB +=  -Wl,--start-group $(MKLROOT)/mkl/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/mkl/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/mkl/lib/intel64/libmkl_core.a -Wl,--end-group  -lpthread -lm  -ldl -liomp5 -lboost_program_options
##LIB+= -Wl,-Bstatic -liomp5 -lboost_program_options
SRC_DIR=src
INC_DIR=include
OBJ_DIR=obj


ifeq ($(stage),GPU)
	SRC_CPP = $(wildcard $(SRC_DIR)/*.cpp)
	SRC_CUDA = $(wildcard $(SRC_DIR)/*.cu)
	SRC_CPP :=$(SRC_CUDA) $(SRC_CPP)
	#OBJ_CUDA = $(SRC_CUDA:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
	OBJ =  $(filter %.o, $(SRC_CPP:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o))
	OBJ += $(filter %.o, $(SRC_CPP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o))
	LIB := $(LIB) -Wl,-Bdynamic -lcuda -lcusolver -lcudart -lcudadevrt -lcublas -Wl,--as-need
	CXXFLAGS := $(CXXFLAGS) -I $(CUDAROOT)/include  -L$(CUDAROOT)/lib64
else
	SRC_CPP = $(filter-out $(CUDACPP), $(wildcard $(SRC_DIR)/*.cpp))
	OBJ = $(SRC_CPP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
	##LIB := $(LIB) -Wl,-Bdynamic -Wl,--as-need
	CXXFLAGS := -DCPU $(CXXFLAGS)
endif





.PHONY: all clean

all : $(OBJ_DIR) $(OUTPUT) clean

$(OUTPUT) :  $(OBJ) 
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ)  $(LIB)
	
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(NVCC) $(CUDAFLAGS) $^ -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS) -c $^ -o $@



clean: 
	rm -rf $(OBJ_DIR)
	 
$(OBJ_DIR):
	mkdir -p $(notdir $@)
