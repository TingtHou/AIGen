#include<cuda_runtime.h>
#include<time.h>
#include "device_launch_parameters.h"

__global__ static void cuMatrixMult(const float* A, const float* B, float* C, int n, clock_t* time);
