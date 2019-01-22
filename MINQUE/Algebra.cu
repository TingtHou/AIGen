#include "Algebra.cuh"
//C=A*B
//A is an Dim 1 array, whose size is M*N, means M rows and N colnums;
//B is an Dim 1 array, whose size is N*S, means N rows and S colnums;
//C is an Dim 1 array, whose size is M*S, means M rows and S colnums;
__global__ static void cuMatrixMult(const float* A, const float* B, float* C, int M, int N, int S)
{
	//(blockIdx.y * gridDim.x + blockIdx.x)  means block ID;
	//(blockDim.x * blockDim.y)              means block size;
	//threadIdx.y * blockDim.y + threadIdx.x means the location (i,j) in the block
	//get thread ID
	const int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.y + threadIdx.x;
	if (tid<M*S)
	{
		int rowid = tid / S;
		int colid = tid % S;
		C[tid] = 0;
		for (int i = 0; i < N; i++)
		{
			C[tid] += A[rowid*N + i] * B[S*i + colid];
		}
	}
}