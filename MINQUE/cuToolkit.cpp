#include "pch.h"
#include "cuToolkit.h"
#include <Eigen/Dense>


int cuToolkit::cuMatrixInv(double * d_A, double *d_A_INV, int N)
{
	int sizeA = N * N * sizeof(double);
	cusolverStatus_t status;
	cudaError_t cublasstatus;
	cusolverDnHandle_t handle;
	status = cusolverDnCreate(&handle);
	if (status != CUSOLVER_STATUS_SUCCESS)
	{
		cublasstatus = cudaGetLastError();
		if (cublasstatus != cudaSuccess)
		{
			std::cout << cudaGetErrorString(cublasstatus) << std::endl <<"!!!! CUSOLVER initialization error\n" << std::endl;
		}
		return status;
	}
	double *d_B, *d_Work;
	int *dLUPivots, *dLUInfo, Lwork;
	double *B = (double*)malloc(N*N * sizeof(double));
	memset(B, 0, N*N * sizeof(double));
	for (int i = 0; i < N; i++)
	{
		B[(N + 1)*i] = 1;
	}
	cudaMalloc((void **)&d_B, N*N * sizeof(double));
	cudaMalloc((void **)& dLUPivots, N * sizeof(int));
	cudaMalloc((void **)& dLUInfo, sizeof(int));
	cudaMemcpy(d_B, B, N*N * sizeof(double), cudaMemcpyHostToDevice);
	cusolverDnDgetrf_bufferSize(handle, N, N, d_A, N, &Lwork);
	cudaMalloc((void **)& d_Work, Lwork * sizeof(double));
	cusolverDnDgetrf(handle, N, N, d_A, N, d_Work, dLUPivots, dLUInfo);
	cudaDeviceSynchronize();
	status=cusolverDnDgetrs(handle, CUBLAS_OP_N, N, N, d_A, N, dLUPivots, d_B, N, dLUInfo);
	cudaDeviceSynchronize();
	if (status != CUSOLVER_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! kernel execution error.\n");
		return status;
	}
	cublasstatus =cudaMemcpy(d_A_INV, d_B, sizeA, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	if (cublasstatus != CUSOLVER_STATUS_SUCCESS) {
		cublasstatus = cudaGetLastError();
		if (cublasstatus != cudaSuccess)
		{
			std::cout << cudaGetErrorString(cublasstatus) << "!!!! CUSOLVER initialization error\n" << std::endl;
		}
		fprintf(stderr, "!!!! Data copy between device error.\n");
		return cublasstatus;
	}
	
	free(B);
	B = nullptr;
	cudaFree(d_B);
	cudaFree(dLUInfo);
	cudaFree(dLUPivots);
	cudaFree(d_Work);
	status = cusolverDnDestroy(handle);
	if (status != CUSOLVER_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return status;
	}
	return status;
}


void cuToolkit::cuGetGPUinfo()
{
	std::cout<<"\\\\\\\\\\\\\\\\\\\\\\\\\GPU Info\\\\\\\\\\\\\\\\\\\\\\\\\\"<<std::endl;
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	printf("Device Numbers: %d\n", nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "#####################################" << std::endl;
		printf("Device ID: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}

	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
}


void InverseTest()
{
	using namespace Eigen;
	MatrixXd a = MatrixXd::Random(4, 4);
	std::cout << a.inverse() << std::endl;
	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	double *aorg, *ainv;
	cudaMalloc((double**)&aorg, 4 * 4 * sizeof(double));
	cudaMalloc((double**)&ainv, 4 * 4 * sizeof(double));
	cudaMemcpy(aorg, a.data(), 4 * 4 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(ainv, 0, 4 * 4 * sizeof(double));
	cuToolkit::cuMatrixInv(aorg, ainv, 4);
	double *ai;
	ai = (double*)malloc(4 * 4 * sizeof(double));
	cudaMemcpy(ai, ainv, 4 * 4 * sizeof(double), cudaMemcpyDeviceToHost);
	MatrixXd a2 = Eigen::Map<MatrixXd>(ai, 4, 4);
	std::cout << a2 << std::endl;
}

