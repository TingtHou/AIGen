
# -----------------------------------------------------------------
#   Makefile for KNN 
#   
#   Supported platforms: Unix / Linux
# ---------------------------------------------------------------------

# Directory of the target
OUTPUT = KNN

RM = rm
# Compiler
CXX = g++

# EIGEN library
EIGEN_PATH = /usr/include/Eigen

# Intel MKL library
MKL_PATH = /opt/intel/mkl

# Compiler flags
CXXFLAGS =  -w -O3 -m64 -static -fopenmp -I $(EIGEN_PATH) -DEIGEN_NO_DEBUG -I $(MKL_PATH)/include -I /usr/include/boost

LIB += -static -lz -Wl,--start-group  $(MKL_PATH)/lib/intel64/libmkl_intel_lp64.a $(MKL_PATH)/lib/intel64/libmkl_gnu_thread.a $(MKL_PATH)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -lboost_program_options

SRC_DIR=src
OBJ_DIR=obj

SRC = $(wildcard $(SRC_DIR)/*.cpp)
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
