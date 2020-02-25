#include "../include/cuToolkit.h"

bool cuToolkit::cuLU(float * d_A, float*d_A_INV, int N)
{
	int sizeA = N * N * sizeof(float);
	cusolverStatus_t status;
	cudaError_t cublasstatus;
	cusolverDnHandle_t handle;
	cusolverDnCreate(&handle);
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	float *d_Work;
	int *dLUPivots, *dLUInfo, Lwork;
	int info_gpu=0;
	cudaMalloc((void **)& dLUPivots, N * sizeof(int));
	cudaMalloc((void **)& dLUInfo, sizeof(int));
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	cusolverDnSgetrf_bufferSize(handle, N, N, d_A, N, &Lwork);
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	cudaMalloc((void **)& d_Work, Lwork * sizeof(float));
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	cusolverDnSgetrf(handle, N, N, d_A, N, d_Work, dLUPivots, dLUInfo);
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	cublasstatus = cudaMemcpy(&info_gpu, dLUInfo, sizeof(int),
		cudaMemcpyDeviceToHost); // copy devInfo -> info_gpu
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
		// check error code
	//if devInfo = 0, the LU factorization is successful. 
	//if devInfo = -i, the i - th parameter is wrong(not counting handle). 
	//if devInfo = i, the U(i, i) = 0.
	if (info_gpu)
	{
		cudaFree(dLUInfo);
		cudaFree(dLUPivots);
		cudaFree(d_Work);
		status = cusolverDnDestroy(handle);
		return false;
	}
	status=cusolverDnSgetrs(handle, CUBLAS_OP_N, N, N, d_A, N, dLUPivots, d_A_INV, N, dLUInfo);
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	cublasstatus = cudaMemcpy(&info_gpu, dLUInfo, sizeof(int),cudaMemcpyDeviceToHost); // copy devInfo -> info_gpu
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	// check error code
	//if devInfo = 0, the Cholesky factorization is successful. if devInfo = -i, the i - th parameter is wrong(not counting handle).
	//if devInfo = i, the leading minor of order i is not positive definite.
	if (info_gpu)
	{
		cudaFree(dLUInfo);
		cudaFree(dLUPivots);
		cudaFree(d_Work);
		status = cusolverDnDestroy(handle);
		cudaDeviceSynchronize();
		cublasstatus = cudaGetLastError();
		if (cublasstatus != cudaSuccess)
		{
			throw (cudaGetErrorString(cublasstatus));
		}
		return false;
	}
	cudaFree(dLUInfo);
	cudaFree(dLUPivots);
	cudaFree(d_Work);

	
	status = cusolverDnDestroy(handle);
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
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
	const double betaMinusOne = -1.0;
	const double betaZero = 0;
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
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaMemset(d_V_d_S_T, 0, sizeof(double)*N*N);
	cudaMemset(d_A_INV, 0, sizeof(double)*N*N);
	cudaMemset(d_S, 0, sizeof(double)*N);
	cudaMemset(d_S_T, 0, sizeof(double)*N*N);
	cudaMemset(d_U, 0, sizeof(double)*N*N);
	cudaMemset(d_VT, 0, sizeof(double)*N*N);
	cudaMemset(devInfo, 0, sizeof(int));
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	// create cusolver and cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cublas_status = cublasCreate(&cublasH);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	// compute buffer size and prepare workspace
	cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, N, N,
		&lwork);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaStat = cudaMalloc((void **)& d_work, sizeof(double)* lwork);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
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
		throw (cudaGetErrorString(cudaStat));
	}
	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); // copy devInfo -> info_gpu
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
		// check error code
		//if devInfo = 0, the Cholesky factorization is successful. if devInfo = -i, the i - th parameter is wrong(not counting handle).
		//if devInfo = i, the leading minor of order i is not positive definite.
	if (info_gpu)
	{
		cudaFree(d_S);
		cudaFree(d_U);
		cudaFree(d_VT);
		cudaFree(devInfo);
		cudaFree(d_work);
		//	cudaFree(d_rwork);
		cudaFree(d_V_d_S_T);
		cublasDestroy(cublasH);
		cusolverDnDestroy(cusolverH);
		return false;
	}
	cuElementInverse(d_S,d_S_T, N);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}

	//V*S_T
	cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_VT, N, d_S_T, N, &beta, d_V_d_S_T, N);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}

	cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, d_V_d_S_T, N, d_U, N, &beta, d_A_INV, N);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
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
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	return true;
}

bool cuToolkit::cuQR(double * d_A, double * d_A_INV, int N)
{
	cusolverDnHandle_t cusolverH;
	cublasHandle_t cublasH;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat = cudaSuccess;
	const int lda = N;
	// declare matrices A and Q,R on the host
	double * d_tau;// scalars defining the elementary reflectors
	int * devInfo; // info on the device
	double * d_work; // workspace on the device
	// workspace sizes
	int lwork_geqrf = 0;
	int lwork_orgqr = 0;
	int lwork = 0;
	int info_gpu = 0; // info copied from the device
	const double d_one = 1; // constants used in
	const double alpha = 1.0;
	const double betaMinusOne = - 1.0;
	const double betaZero= 0;
	// create cusolver and cublas handles
	cusolver_status = cusolverDnCreate(&cusolverH);
	cublas_status = cublasCreate(&cublasH);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaStat = cudaMalloc((void **)& d_tau, sizeof(double)*N);
	cudaStat = cudaMalloc((void **)& devInfo, sizeof(int));
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaDeviceSynchronize();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	// compute working space for geqrf and orgqr
	cusolver_status = cusolverDnDgeqrf_bufferSize(cusolverH, N, N, d_A, lda, &lwork_geqrf); // compute Sgeqrf buffer size
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cusolver_status = cusolverDnDorgqr_bufferSize(cusolverH, N, N, N, d_A, lda, d_tau, &lwork_orgqr); // and Sorgqr b. size
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
	// device memory for workspace
	cudaStat = cudaMalloc((void **)& d_work, sizeof(double)* lwork);
	// QR factorization for d_A
	cusolver_status = cusolverDnDgeqrf(cusolverH, N, N, d_A, lda, d_tau, d_work, lwork, devInfo);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); // copy devInfo -> info_gpu
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	// check error code
	//if devInfo = 0, the Cholesky factorization is successful. if devInfo = -i, the i - th parameter is wrong(not counting handle).
	//if devInfo = i, the leading minor of order i is not positive definite.
	if (info_gpu)
	{
		cudaFree(d_tau);
		cudaFree(devInfo);
		cudaFree(d_work);
		cublasDestroy(cublasH);
		cusolverDnDestroy(cusolverH);
		return false;
	}

/*
/////////////////
	double *d_R;
	cudaMalloc((void **)&d_R, N*N * sizeof(double));
	cudaMemset(d_R, 0, N*N * sizeof(double));
	cuGetupperTriangular(d_A, d_R, N);
	cudaDeviceSynchronize();
/ *	cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_R, N, d_U, N, &beta, d_A_INV, N);* /
	double *R = (double*)malloc(N*N * sizeof(double));
	cudaMemcpy(R, d_R, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
	Eigen::MatrixXd a1 = Eigen::Map<Eigen::MatrixXd>(R, N, N);
	std::cout << "R:\n" << a1 << std::endl;
							
							
	cublas_status = cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
		N, N, &d_one, d_R, lda, d_A_INV, N);
	cudaDeviceSynchronize();
	double *R_inv = (double*)malloc(N*N * sizeof(double));
	cudaMemcpy(R_inv, d_A_INV, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
	Eigen::MatrixXd a1_inv = Eigen::Map<Eigen::MatrixXd>(R_inv, N, N);
	std::cout << "R inverse:\n" << a1_inv << std::endl;
							
	cusolver_status = cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, N, N, N, d_A, lda,
		d_tau, d_A_INV, N, d_work, lwork, devInfo);
	cudaDeviceSynchronize();
							
	cusolver_status = cusolverDnDorgqr(
		cusolverH,
		N,
		N,
		N,
		d_A,
		lda,
		d_tau,
		d_work,
		lwork,
		devInfo);
		cudaDeviceSynchronize();
		double *Q = (double*)malloc(N*N * sizeof(double));
		cudaMemcpy(Q, d_A, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
							
	Eigen::MatrixXd a2 = Eigen::Map<Eigen::MatrixXd>(Q, N, N);
	std::cout << "Q:\n" << a2 << std::endl;
								*/

	// step 5: compute Q^T*B, B is Idenitity
	cusolver_status = cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT,	CUBLAS_OP_T, N,	N, N, d_A, lda,
		d_tau, d_A_INV, N, d_work, lwork, devInfo);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); // copy devInfo -> info_gpu
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	// check error code
	//if devInfo = 0, the Cholesky factorization is successful. if devInfo = -i, the i - th parameter is wrong(not counting handle).
	//if devInfo = i, the leading minor of order i is not positive definite.
	if (info_gpu)
	{
		cudaFree(d_tau);
		cudaFree(devInfo);
		cudaFree(d_work);
		cublasDestroy(cublasH);
		cusolverDnDestroy(cusolverH);
		return false;
	}

	cublas_status = cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
		N, N, &d_one, d_A, lda, d_A_INV,N);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	//check inversed 
	
	cudaFree(d_tau);
	cudaFree(devInfo);
	cudaFree(d_work);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	return true;
}

bool cuToolkit::cuCholesky(float* d_A, int N)
{
	cudaError cudaStatus;
	cusolverStatus_t cusolverStatus;

	cusolverDnHandle_t handle; // device versions of
	float  *Work, *d_A_INV; //worksp & inverse result.
	cudaStatus = cudaMalloc((void**)&d_A_INV, N* N*sizeof(float));
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	cuAsDiag(d_A_INV, N);
	int *d_info, Lwork; // device version of info , worksp . size
	int info_gpu = 0; // device info copied to host
	cusolverStatus = cusolverDnCreate(&handle); // create handle
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	// prepare memory on the device
	cudaStatus = cudaMalloc((void **)& d_info, sizeof(int));
	// compute workspace size and prepare workspace
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	cusolverStatus = cusolverDnSpotrf_bufferSize(handle, uplo, N, d_A, N, &Lwork);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMalloc((void **)& Work, Lwork * sizeof(float));
	// Cholesky decomposition d_A =L*L^T, lower triangle of d_A is
	// replaced by the factor L
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	cusolverStatus = cusolverDnSpotrf(handle, uplo, N, d_A, N, Work,
		Lwork, d_info);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy (& info_gpu , d_info , sizeof (int), cudaMemcpyDeviceToHost ); // copy devInfo -> info_gpu
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	// check error code
	//if devInfo = 0, the Cholesky factorization is successful. if devInfo = -i, the i - th parameter is wrong(not counting handle).
	//if devInfo = i, the leading minor of order i is not positive definite.
	if (info_gpu)
	{
		cudaStatus = cudaFree(d_info);
		cudaStatus = cudaFree(Work);
		cusolverStatus = cusolverDnDestroy(handle);
		return false;
	}
	// solve d_A *X=d_B , where d_A is factorized by potrf function
	// d_B is overwritten by the solution
	cusolverStatus = cusolverDnSpotrs(handle, uplo, N, N, d_A, N,
		d_A_INV, N, d_info);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost); // copy devInfo -> info_gpu
	//if devInfo = 0, the Cholesky factorization is successful. if devInfo = -i, the i-th parameter is wrong (not counting handle).
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
	if (info_gpu)
	{
		cudaStatus = cudaFree(d_info);
		cudaStatus = cudaFree(Work);
		cusolverStatus = cusolverDnDestroy(handle);
		return false;
	}
	cudaMemcpy(d_A, d_A_INV, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(d_A_INV);
	cudaFree(d_info);
	cudaFree(Work);
	cusolverDnDestroy(handle);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStatus));
	}
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
	int nrank = 10;
	using namespace Eigen;
	MatrixXf a = MatrixXf::Random(nrank, nrank);
	a = a.transpose() * a;
	MatrixXf a_inv = a.inverse();
	Eigen::MatrixXf IdentityMatrix(nrank, nrank);
	IdentityMatrix.setIdentity();
// 	VectorXd b(nrank);
// 	b.setOnes();
// 	a.col(0) << b;
// 	a.col(5) << b;
	
//	Eigen::MatrixXd ain(nrank, nrank);
//	ain.setZero();
//	bool status=ToolKit::Inv_LU(a, ain);
//	std::cout << "LU inverse Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
 //	std::cout << "Original :\n" << a << std::endl;
// 	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\"<< std::endl;
// 	std::cout << "Inverse :\n" << a.inverse() << std::endl;
// 	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	{
		float *aorg;
		cudaMalloc((float**)&aorg, nrank * nrank * sizeof(float));
		cudaMemcpy(aorg, a.data(), nrank * nrank * sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		bool inverstatus=cuToolkit::cuCholesky(aorg, nrank);
		if (inverstatus)
		{
			float*ai;
			ai = (float*)malloc(nrank * nrank * sizeof(float));
			cudaMemcpy(ai, aorg, nrank * nrank * sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			MatrixXf a2 = Eigen::Map<MatrixXf>(ai, nrank, nrank);
				std::cout << a2 << std::endl;
				std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
				std::cout << a_inv << std::endl;
		/*
			if (a_solution_exists)
			{
				std::cout << "Cholesky Passed." << std::endl;
			}
			else
			{
				std::cout << "Cholesky Failed." << std::endl;
			}
			*/
			free(ai);
		}
		else
		{
			std::cout << "Cholesky Failed." << std::endl;
		}
		
		cudaFree(aorg);
	
	}
	/*
	{
		double *aorg, *ainv;
		cudaMalloc((double**)&aorg, nrank * nrank * sizeof(double));
		cudaMalloc((double**)&ainv, nrank * nrank * sizeof(double));
		cudaMemcpy(aorg, a.data(), nrank * nrank * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ainv, h_ainv, nrank * nrank * sizeof(double), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		clock_t t1 = clock();
		bool status= cuToolkit::cuQR(aorg, ainv, nrank);
		std::cout << "cuQR inverse Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		if (status)
		{
			double *ai;
			ai = (double*)malloc(nrank * nrank * sizeof(double));
			cudaMemcpy(ai, ainv, nrank * nrank * sizeof(double), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			MatrixXd a2 = Eigen::Map<MatrixXd>(ai, nrank, nrank);
	//		std::cout << a2 << std::endl;
			bool a_solution_exists = (a*a2*a).isApprox(a, 1e-10);
			if (a_solution_exists)
			{
				std::cout << "QR Passed." << std::endl;
			}
			else
			{
				std::cout << "QR Failed." << std::endl;
			}
			free(ai);
		}
		else
		{
			std::cout << "QR Failed." << std::endl;
		}
		cudaFree(aorg);
		cudaFree(ainv);
	
	}
	{
		double *aorg, *ainv;
		cudaMalloc((double**)&aorg, nrank * nrank * sizeof(double));
		cudaMalloc((double**)&ainv, nrank * nrank * sizeof(double));
		cudaMemcpy(aorg, a.data(), nrank * nrank * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ainv, h_ainv, nrank * nrank * sizeof(double), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		clock_t t1 = clock();
		bool status=cuToolkit::cuSVD(aorg, ainv, nrank);
		std::cout << "cuSVD inverse Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		if (status)
		{
			double *ai;
			ai = (double*)malloc(nrank * nrank * sizeof(double));
			cudaMemcpy(ai, ainv, nrank * nrank * sizeof(double), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			MatrixXd a2 = Eigen::Map<MatrixXd>(ai, nrank, nrank);
			bool a_solution_exists = (a*a2*a).isApprox(a, 1e-10);
			if (a_solution_exists)
			{
				std::cout << "SVD Passed." << std::endl;
			}
			else
			{
				std::cout << "SVD Failed." << std::endl;
			}
			
			free(ai);
		}
		else
		{
			std::cout << "SVD Failed." << std::endl;
		}
		cudaFree(aorg);
		cudaFree(ainv);

		
		
	}

	{
		double *aorg, *ainv;
		cudaMalloc((double**)&aorg, nrank * nrank * sizeof(double));
		cudaMalloc((double**)&ainv, nrank * nrank * sizeof(double));
		cudaMemcpy(aorg, a.data(), nrank * nrank * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ainv, h_ainv, nrank * nrank * sizeof(double), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		clock_t t1 = clock();
		bool status=cuToolkit::cuLU(aorg, ainv, nrank);
		std::cout << "cuLU inverse Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		if (status)
		{
			double *ai;
			ai = (double*)malloc(nrank * nrank * sizeof(double));
			cudaMemcpy(ai, ainv, nrank * nrank * sizeof(double), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			MatrixXd a2 = Eigen::Map<MatrixXd>(ai, nrank, nrank);
			bool a_solution_exists = (a*a2*a).isApprox(a, 1e-10);
			if (a_solution_exists)
			{
				std::cout << "LU Passed." << std::endl;
			}
			else
			{
				std::cout << "LU Failed." << std::endl;
			}
			free(ai);
		}
		else
		{
			std::cout << "LU Failed." << std::endl;
		}
		
		cudaFree(aorg);
		cudaFree(ainv);
	
		
	}
	*/
}
