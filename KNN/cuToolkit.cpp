#include "pch.h"
#include "cuToolkit.h"
#include <Eigen/Dense>
extern "C" void cuDiagMatrixinv(double *A, double *A_inv, const int nrows);

bool cuToolkit::cuLU(double * d_A, double *d_A_INV, int N)
{
	int sizeA = N * N * sizeof(double);
	cusolverStatus_t status;
	cudaError_t cublasstatus;
	cusolverDnHandle_t handle;
	status = cusolverDnCreate(&handle);
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cublasstatus) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
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
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cublasstatus) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
	}
	cudaMemcpy(d_A_INV, d_B, sizeA, cudaMemcpyDeviceToDevice);
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cublasstatus) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
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
		return false;
	}
	return true;
}

bool cuToolkit::cuSVD(double * d_A, double * d_A_INV, int N)
{
	cusolverDnHandle_t cusolverH; // cusolver handle
	cublasHandle_t cublasH; // cublas handle
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat = cudaSuccess;

	const double alpha = 1.0;
	const double beta = 1.0;
	double *d_U, *d_VT, *d_S, *d_V_d_S_T, *d_S_T; // and sing . val . matrix d_S
	int * devInfo; // on the device
	double *d_work, *d_rwork; // workspace on the device
	int lwork = 0;
	int info_gpu = 0; // info copied from device to host

	cudaStat = cudaMalloc((void **)& d_V_d_S_T, sizeof(double)* N*N);
	cudaStat = cudaMalloc((void **)& d_S, sizeof(double)*N);
	cudaStat = cudaMalloc((void **)& d_S_T, sizeof(double)* N *N);
	cudaStat = cudaMalloc((void **)& d_U, sizeof(double)* N *N);
	cudaStat = cudaMalloc((void **)& d_VT, sizeof(double)* N *N);
	cudaStat = cudaMalloc((void **)& devInfo, sizeof(int));
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudaStat) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
	}
	cudaMemset(d_V_d_S_T, 0, sizeof(double)*N*N);
	cudaMemset(d_A_INV, 0, sizeof(double)*N*N);
	cudaMemset(d_S, 0, sizeof(double)*N);
	cudaMemset(d_S_T, 0, sizeof(double)*N*N);
	cudaMemset(d_U, 0, sizeof(double)*N*N);
	cudaMemset(d_VT, 0, sizeof(double)*N*N);
	cudaMemset(devInfo, 0, sizeof(int));
	// create cusolver and cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	cublas_status = cublasCreate(&cublasH);
	// compute buffer size and prepare workspace
	cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, N, N,
		&lwork);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudaStat) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
	}
	cudaStat = cudaMalloc((void **)& d_work, sizeof(double)* lwork);
	
	// compute the singular value decomposition of d_A
	// and optionally the left and right singular vectors :
	// d_A = d_U *d_S * d_VT ; the diagonal elements of d_S
	// are the singular values of d_A in descending order
	// the first min (m,n) columns of d_U contain the left sing .vec .
	// the first min (m,n) cols of d_VT contain the right sing .vec .
	signed char jobu = 'A'; // all m columns of d_U returned
	signed char jobvt = 'A'; // all n columns of d_VT returned
	cusolver_status = cusolverDnDgesvd(cusolverH, jobu, jobvt,
		N, N, d_A, N, d_S, d_U, N, d_VT, N, d_work, lwork,
		NULL, devInfo);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudaStat) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
	}

	cuDiagMatrixinv(d_S,d_S_T, N);
	//V*S_T

	cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_VT, N, d_S_T, N, &beta, d_V_d_S_T, N);

	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudaStat) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
	}

	cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, d_V_d_S_T, N, d_U, N, &beta, d_A_INV, N);
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudaStat) << std::endl << "!!!! CUSOLVER initialization error\n" << std::endl;
		return false;
	}

	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_VT);
	cudaFree(devInfo);
	cudaFree(d_work);
//	cudaFree(d_rwork);
	cudaFree(d_V_d_S_T);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
	return true;
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
	cuToolkit::cuSVD(aorg, ainv, 4);
	double *ai;
	ai = (double*)malloc(4 * 4 * sizeof(double));
	cudaMemcpy(ai, ainv, 4 * 4 * sizeof(double), cudaMemcpyDeviceToHost);
	MatrixXd a2 = Eigen::Map<MatrixXd>(ai, 4, 4);
	std::cout << a2 << std::endl;
	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	std::cout << a2 * a << std::endl;
}

