#include "../include/cuToolkit.h"



//@brief:	Inverse a float matrix, and rewrite the inversed matrix into original matrix;
//@param:	Ori_Matrix				The matrix to be inversed;
//@param:	DecompositionMode		The mode to decompose the matrix;Cholesky = 0, LU = 1, QR = 2,SVD = 3;
//@param:	AltDecompositionMode	Another mode to decompose if the first mode is failed; Only QR = 2 and SVD = 3 are allowed;
//@param:	allowPseudoInverse		Allow to use pseudo inverse matrix; If True, AltDecompositionMode is avaiable; otherwise AltDecompositionMode is unavaiable.
//@ret:		int						The info of statue of inverse method; 
//									0 means successed; 1 means calculating inverse matrix is failed, and  using pseudo inverse matrix instead if allowPseudoInverse is true; 
//									2 means calculating inverse matrix is failed, and pseudo inverse matrix is also failed.
int cuInverse(float * Ori_Matrix, int N, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse)
{
	int status = 0;
	bool statusInverse;
	switch (DecompositionMode)
	{
	case 0:
	{
		float det;
		statusInverse = cuToolkit::cuCholesky(Ori_Matrix,N);
		//	statusInverse = ToolKit::Inv_Cholesky(Ori_Matrix);
		status += !statusInverse;
		if (!statusInverse && allowPseudoInverse)
		{
			if (AltDecompositionMode == 3)
			{
				statusInverse = cuToolkit::cuSVD(Ori_Matrix,N);
			}
			if (AltDecompositionMode == 2)
			{
				statusInverse = cuToolkit::cuQR(Ori_Matrix,N);
			}
			status += !statusInverse;
		}
	}
	break;
	case 1:
	{
		float det;
		statusInverse = cuToolkit::cuLU(Ori_Matrix,N);
		status += !statusInverse;
		if (!statusInverse && allowPseudoInverse)
		{
			if (AltDecompositionMode == 3)
			{
				statusInverse = cuToolkit::cuSVD(Ori_Matrix, N);
			}
			if (AltDecompositionMode == 2)
			{
				statusInverse = cuToolkit::cuQR(Ori_Matrix, N);
			}
			status += !statusInverse;
		}
	}
	break;
	case 2:
	{
		statusInverse = cuToolkit::cuQR(Ori_Matrix, N);
		status += !statusInverse;
	}
	break;
	case 3:
	{
		statusInverse = cuToolkit::cuSVD(Ori_Matrix, N);
		status += !statusInverse;
	}
	break;
	}
	return status;
}


bool cuToolkit::cuLU(float * d_A, int N)
{
	int sizeA = N * N * sizeof(float);
	float* d_A_INV;
	cudaMalloc((void**)&d_A_INV, sizeA);
	cuAsDiag(d_A_INV, N);
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
	//This function solves a linear system of multiple right-hand sides
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
		cudaFree(d_A_INV);
		status = cusolverDnDestroy(handle);
		cudaDeviceSynchronize();
		cublasstatus = cudaGetLastError();
		if (cublasstatus != cudaSuccess)
		{
			throw (cudaGetErrorString(cublasstatus));
		}
		return false;
	}
	cudaMemcpy(d_A, d_A_INV, sizeA, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	cudaFree(dLUInfo);
	cudaFree(dLUPivots);
	cudaFree(d_Work);
	cudaFree(d_A_INV);
	
	status = cusolverDnDestroy(handle);
	cudaDeviceSynchronize();
	cublasstatus = cudaGetLastError();
	if (cublasstatus != cudaSuccess)
	{
		throw (cudaGetErrorString(cublasstatus));
	}
	return true;
}

bool cuToolkit::cuSVD(float * d_A, int N)
{
	cusolverDnHandle_t cusolverH; // cusolver handle
	cublasHandle_t cublasH; // cublas handle
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat = cudaSuccess;
	float* d_A_INV;
	

	const float alpha = 1.0;
	const float beta = 1.0;
	const float betaMinusOne = -1.0;
	const float betaZero = 0;
	float*d_U, *d_VT, *d_S, *d_V_d_S_T, *d_S_T; // and sing . val . matrix d_S
	int * devInfo; // on the device
	float*d_work, *d_rwork; // workspace on the device
	int lwork = 0;
	int info_gpu = 0; // info copied from device to host

	cudaStat = cudaMalloc((void **)&d_A_INV, sizeof(float) * N * N);
	cudaStat = cudaMalloc((void **)& d_V_d_S_T, sizeof(float)* N*N);
	cudaStat = cudaMalloc((void **)& d_S, sizeof(float)*N);
	cudaStat = cudaMalloc((void **)& d_S_T, sizeof(float)* N *N);
	cudaStat = cudaMalloc((void **)& d_U, sizeof(float)* N *N);
	cudaStat = cudaMalloc((void **)& d_VT, sizeof(float)* N *N);
	cudaStat = cudaMalloc((void **)& devInfo, sizeof(int));
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaMemset(d_A_INV, 0, sizeof(float) * N * N);
	cudaMemset(d_V_d_S_T, 0, sizeof(float)*N*N);
	cudaMemset(d_S, 0, sizeof(float)*N);
	cudaMemset(d_S_T, 0, sizeof(float)*N*N);
	cudaMemset(d_U, 0, sizeof(float)*N*N);
	cudaMemset(d_VT, 0, sizeof(float)*N*N);
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
	cudaStat = cudaMalloc((void **)& d_work, sizeof(float)* lwork);
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
	cusolver_status = cusolverDnSgesvd(cusolverH, jobu, jobvt,
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
		cudaFree(d_A_INV);
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
	cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_VT, N, d_S_T, N, &beta, d_V_d_S_T, N);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}

	cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, d_V_d_S_T, N, d_U, N, &beta, d_A_INV, N);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaMemcpy(d_A, d_A_INV, N*N * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_VT);
	cudaFree(devInfo);
	cudaFree(d_work);
//	cudaFree(d_rwork);
	cudaFree(d_V_d_S_T);
	cudaFree(d_A_INV);
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

bool cuToolkit::cuQR(float * d_A, int N)
{
	cusolverDnHandle_t cusolverH;
	cublasHandle_t cublasH;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat = cudaSuccess;
	const int lda = N;
	float* d_A_INV;
	cudaStat = cudaMalloc((void **)&d_A_INV, N*N*sizeof(float));
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cuAsDiag(d_A_INV, N);
	// declare matrices A and Q,R on the host
	float* d_tau;// scalars defining the elementary reflectors
	int * devInfo; // info on the device
	float* d_work; // workspace on the device
	// workspace sizes
	int lwork_geqrf = 0;
	int lwork_orgqr = 0;
	int lwork = 0;
	int info_gpu = 0; // info copied from the device
	const float d_one = 1; // constants used in
	const float alpha = 1.0;
	const float betaMinusOne = - 1.0;
	const float betaZero= 0;
	// create cusolver and cublas handles
	cusolver_status = cusolverDnCreate(&cusolverH);
	cublas_status = cublasCreate(&cublasH);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaStat = cudaMalloc((void **)& d_tau, sizeof(float)*N);
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
	cusolver_status = cusolverDnSgeqrf_bufferSize(cusolverH, N, N, d_A, lda, &lwork_geqrf); // compute Sgeqrf buffer size
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cusolver_status = cusolverDnSorgqr_bufferSize(cusolverH, N, N, N, d_A, lda, d_tau, &lwork_orgqr); // and Sorgqr b. size
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
	// device memory for workspace
	cudaStat = cudaMalloc((void **)& d_work, sizeof(float)* lwork);
	// QR factorization for d_A
	cusolver_status = cusolverDnSgeqrf(cusolverH, N, N, d_A, lda, d_tau, d_work, lwork, devInfo);
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

	// step 5: compute Q^T*B, B is Idenitity
	cusolver_status = cusolverDnSormqr(cusolverH, CUBLAS_SIDE_LEFT,	CUBLAS_OP_T, N,	N, N, d_A, lda,
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
		cudaFree(d_A_INV);
		cublasDestroy(cublasH);
		cusolverDnDestroy(cusolverH);
		return false;
	}
	//This function solves the triangular linear system with multiple right - hand - sides
	cublas_status = cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
		N, N, &d_one, d_A, lda, d_A_INV,N);
	cudaDeviceSynchronize();
	cudaStat = cudaGetLastError();
	if (cudaStat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudaStat));
	}
	cudaMemcpy(d_A, d_A_INV, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
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
	cudaFree(d_A_INV);
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

/*
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
*/
/*
void InverseTest()
{
	int nrank = 10;
	using namespace Eigen;
	MatrixXf a = MatrixXf::Random(nrank, nrank);
	a = a.transpose() * a;
	//std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	//std::cout << "Orginal Matrix:" << std::endl;
	//std::cout << a.inverse() << std::endl;
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
			MatrixXf newMatrix = a * a2 * a;
//			std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
//			std::cout << "Orginal Matrix * Inversed Matrix * Orginal Matrix:" << std::endl;
//			std::cout << a2 << std::endl;
			
			bool a_solution_exists = newMatrix.isApprox(a);
			if (a_solution_exists)
			{
				std::cout << "Cholesky Passed." << std::endl;
			}
			else
			{
				std::cout << "Cholesky Failed." << std::endl;
			}
			
			free(ai);
		}
		else
		{
			std::cout << "Cholesky Failed." << std::endl;
		}
		
		cudaFree(aorg);
	
	}

	{
		float* aorg;
		cudaMalloc((float**)&aorg, nrank * nrank * sizeof(float));
		cudaMemcpy(aorg, a.data(), nrank * nrank * sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		bool inverstatus = cuToolkit::cuLU(aorg, nrank);
		if (inverstatus)
		{
			float* ai;
			ai = (float*)malloc(nrank * nrank * sizeof(float));
			cudaMemcpy(ai, aorg, nrank * nrank * sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			MatrixXf a2 = Eigen::Map<MatrixXf>(ai, nrank, nrank);
			//		std::cout << a2 << std::endl;
			//		std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
			//		std::cout << a_inv << std::endl;
			bool a_solution_exists = (a * a2 * a).isApprox(a);
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

	}
	

	{
		float* aorg;
		cudaMalloc((float**)&aorg, nrank * nrank * sizeof(float));
		cudaMemcpy(aorg, a.data(), nrank * nrank * sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		bool inverstatus = cuToolkit::cuQR(aorg, nrank);
		if (inverstatus)
		{
			float* ai;
			ai = (float*)malloc(nrank * nrank * sizeof(float));
			cudaMemcpy(ai, aorg, nrank * nrank * sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			MatrixXf a2 = Eigen::Map<MatrixXf>(ai, nrank, nrank);
			//		std::cout << a2 << std::endl;
			//		std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
			//		std::cout << a_inv << std::endl;
			bool a_solution_exists = (a * a2 * a).isApprox(a);
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

	}
	{
		float* aorg;
		cudaMalloc((float**)&aorg, nrank * nrank * sizeof(float));
		cudaMemcpy(aorg, a.data(), nrank * nrank * sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		bool inverstatus = cuToolkit::cuSVD(aorg, nrank);
		if (inverstatus)
		{
			float* ai;
			ai = (float*)malloc(nrank * nrank * sizeof(float));
			cudaMemcpy(ai, aorg, nrank * nrank * sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			MatrixXf a2 = Eigen::Map<MatrixXf>(ai, nrank, nrank);
	//		std::cout << a2 << std::endl;
		//	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	//		std::cout << a.inverse() << std::endl;
			bool a_solution_exists = (a * a2 * a).isApprox(a);
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

	}
}
*/