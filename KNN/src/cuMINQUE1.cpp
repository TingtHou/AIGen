#include "../include/cuMINQUE1.h"

cuMINQUE1::cuMINQUE1(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomposition = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomposition = altDecompositionMode;
	CheckGPU();
}

cuMINQUE1::~cuMINQUE1()
{
}

void cuMINQUE1::importY(Eigen::VectorXf& Y)
{
	this->h_Y = Y.data();
	this->nind = Y.size();
	cudaMalloc((float**)&d_Y, nind * sizeof(float));
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}
	cudaMemcpy(d_Y, h_Y, nind * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}
	

}

void cuMINQUE1::pushback_Vi(Eigen::MatrixXf& vi)
{
	h_Vi.push_back(vi.data());
	nVi++;
}

void cuMINQUE1::pushback_X(Eigen::MatrixXf& X, bool intercept)
{
	int nrows = X.rows();
	int ncols = X.cols();
	assert(nrows == nind);
	if (ncov == 0)
	{
		if (intercept)
		{
			this->X.resize(nind, ncols + 1);
			Eigen::VectorXf IDt(nind);
			IDt.setOnes();
			this->X << IDt, X;
		}
		else
		{
			this->X = X;
		}

	}
	else
	{
		Eigen::MatrixXf tmpX = this->X;
		if (intercept)
		{
			this->X.resize(nind, ncov + ncols - 1);
			this->X << tmpX, X.block(0, 1, nrows, ncols - 1);
		}
		else
		{
			this->X.resize(nind, ncov + ncols);
			this->X << tmpX, X;
		}

	}
	ncov = this->X.cols();

}

void cuMINQUE1::pushback_W(Eigen::VectorXf &W)
{
	this->h_W = W.data();
	Isweight = true;
}

void cuMINQUE1::estimateVCs()
{
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	init();
	float* d_VW, *d_Ry;
	float* h_Ry; Eigen::VectorXf Ry(nind);
	h_Ry = Ry.data();
	//h_Ry = (float*)malloc(nind * sizeof(float));
	cudaMalloc((float**)&d_Ry, nind * sizeof(float));
	cudaMalloc((float**)&d_VW, nind * nind * sizeof(float));
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}
	cudaMemset(d_VW, 0, nind * nind * sizeof(float));
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}

	if (!Isweight)
	{
		#pragma omp parallel for
		for (int i = 0; i < nVi; i++)
		{
			h_W[i]=1;
		}
	}
	const float one = 1; 
	const float zero = 0;
	const float minus_one = -1;
	for (int i = 0; i < nVi; i++)
	{
		cublasStatus_t cublasstatus=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &h_W[i], d_Vi[i], nind, d_Identity, nind, &one, d_VW, nind);
		if (cublasstatus!= CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(cublasstatus));
		}
	}
	cuToolkit::cuCholesky(d_VW, nind);
//	std::vector<Eigen::MatrixXf> RV(nVi);
	std::vector<float*> d_RV(nVi);
	std::vector<float*> d_Vi_y(nVi);
	std::vector<float*> d_yVi_y(nVi);
//	std::vector<float*> h_RV(nVi);
	for (int i = 0; i < nVi; i++)
	{
//		RV[i].resize(nind, nind);
//		RV[i].setZero();
		cudaMalloc((float**)&d_RV[i], nind * nind * sizeof(float));
		cudaMalloc((float**)&d_Vi_y[i], nind * 1 * sizeof(float));
		cudaMalloc((float**)&d_yVi_y[i], nind * sizeof(float));
		cudaDeviceSynchronize();
		cudastat = cudaGetLastError();
		if (cudastat != cudaSuccess)
		{
			throw (cudaGetErrorString(cudastat));
		}
		cudaMemset(d_RV[i], 0, nind * nind * sizeof(float));
		cudaMemset(d_Vi_y[i], 0, nind * 1 * sizeof(float));
		cudaMemset(d_yVi_y[i], 0, nind * sizeof(float));
		//h_RV[i] = (float*)malloc(nind * nind * sizeof(float));
//		h_RV[i]= RV[i].data();
	}
	if (ncov == 0)
	{
		cublasSgemv(handle, CUBLAS_OP_N, nind, nind, &one, d_VW, nind, d_Y, 1, 0, d_Ry, 1);
	}
	else
	{
		float* d_B, *d_Xt_B, *d_inv_XtB_Bt;
		cudaMalloc((float**)&d_B, nind * ncov * sizeof(float));
		cudaMalloc((float**)&d_Xt_B, ncov * ncov * sizeof(float));
		cudaMalloc((float**)&d_inv_XtB_Bt, ncov * nind * sizeof(float));
		cudaMemset(d_B, 0, nind * ncov * sizeof(float));
		cudaMemset(d_Xt_B, 0, ncov * ncov * sizeof(float));
		cudaMemset(d_inv_XtB_Bt, 0, ncov * nind * sizeof(float));
		cudaDeviceSynchronize();
		//B=inv(VW)*X
		status=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, ncov, nind, &one, d_VW, nind, d_X, nind, &zero, d_B, nind);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		
		//XtB=Xt*B
		status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ncov, ncov, nind, &one, d_X, nind, d_B, nind, &zero, d_Xt_B, ncov);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		//inv(XtB)
		cuToolkit::cuCholesky(d_Xt_B, ncov);
		//inv_XtB_Bt=inv(XtB)*Bt
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,ncov, nind,ncov, &one, d_Xt_B, ncov, d_B, nind, &zero, d_inv_XtB_Bt, ncov);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		//inv_VM=inv_VM-B*inv(XtB)*Bt
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, X.cols(), &minus_one, d_B, nind, d_inv_XtB_Bt, ncov, &one , d_VW, nind);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		//Ry=(inv_VM-B*inv(XtB)*Bt)*Y
		status = cublasSgemv(handle, CUBLAS_OP_N, nind, nind, &one, d_VW, nind, d_Y, 1, &zero, d_Ry, 1);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		
		cudaFree(d_B);
		cudaFree(d_Xt_B);
		cudaFree(d_inv_XtB_Bt);
	}

	for (int i = 0; i < nVi; i++)
	{
		status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, nind, nind, &one, d_VW, nind, d_Vi[i], nind, &zero, d_RV[i], nind);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		status = cublasSgemv(handle, CUBLAS_OP_N, nind, nind, &one, d_Vi[i], nind, d_Ry, 1, &zero, d_Vi_y[i], 1);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
	}
	/*
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::ratio<1, 1>> duration_s(end - start);
	std::cout << "GPU completed" << std::endl;
	std::cout << "Computational time: " << duration_s.count() << " second(s)." << std::endl;
	*/
	Eigen::VectorXf h_u(nVi);
	Eigen::MatrixXf h_F(nVi, nVi);
	float* V_identity;
	cudaMalloc((float**)&V_identity, nind * sizeof(float));
	cuFillValue(V_identity, nind,1);
	cudaDeviceSynchronize();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

	for (int i = 0; i < nVi; i++)
	{
		cuVectorCwiseProduct(d_Ry, d_Vi_y[i],d_yVi_y[i],nind);
		status = cublasSdot(handle, nind, d_yVi_y[i], 1, V_identity, 1, h_u.data()+i);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
	}
	cudaFree(V_identity);
	cudaMalloc((float**)&V_identity, nind*nind * sizeof(float));
	cuFillValue(V_identity, nind* nind, 1);
	Eigen::MatrixXf tmp(nind, nind);
	for (int i = 0; i < nVi; i++)
	{
		float* d_RVi_t;
		cudaMalloc((float**)&d_RVi_t, nind * nind * sizeof(float));
		cudaMemcpy(d_RVi_t, d_RV[i], nind * nind * sizeof(float), cudaMemcpyDeviceToDevice);
		cuTranspose(d_RV[i], d_RVi_t, nind, nind);
		for (int j = i; j < nVi; j++)
		{
			float* RVij;
			cudaMalloc((float**)&RVij, nind* nind * sizeof(float));
			cudaMemset(RVij, 0, nind* nind * sizeof(float));
			cuVectorCwiseProduct(d_RVi_t, d_RV[j], RVij, nind*nind);
			status = cublasSdot(handle, nind * nind, RVij, 1, V_identity, 1, h_F.data()+i + nVi * j);
			cudaDeviceSynchronize();
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				throw (_cudaGetErrorEnum(status));
			}
			h_F(j, i) = h_F(i, j);
		}
	}
	cudaFree(V_identity);
	ToolKit::comput_inverse_logdet_LU_mkl(h_F);
	vcs = h_F * h_u;
	cudaFree(d_VW);
	cudaFree(d_Ry);
	for (int i = 0; i < nVi; i++)
	{
		cudaFree(d_RV[i]);
		cudaFree(d_Vi_y[i]);
		cudaFree(d_yVi_y[i]);
	}
}



void cuMINQUE1::init()
{
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

	//alloc a mem for d_X on device
	cudaMalloc((float**)&d_X, nind * ncov * sizeof(float));
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}
	//copy X from host to device
	h_X = X.data();
	cudaMemcpy(d_X, h_X, nind * ncov * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}

	//generate a identity matrix on device
	h_Identity = (float*)malloc(nind * nind * sizeof(float));
	memset(h_Identity, 0, nind * nind * sizeof(float));
	for (int id = 0; id < nind; id++) h_Identity[id * nind + id] = 1;
	cudaMalloc((float**)&d_Identity, nind * nind * sizeof(float));
	cudastat = cudaMemcpy(d_Identity, h_Identity, nind * nind * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}
	d_Vi.resize(nVi);
	//copy Vi from host to device
	for (int i = 0; i < nVi; i++)
	{
		cudaMalloc((float**)&d_Vi[i], nind * nind * sizeof(float));
		cudaDeviceSynchronize();
		cudastat = cudaGetLastError();
		if (cudastat != cudaSuccess)
		{
			throw (cudaGetErrorString(cudastat));
		}
		cudaMemcpy(d_Vi[i], h_Vi[i], nind * nind * sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		cudastat = cudaGetLastError();
		if (cudastat != cudaSuccess)
		{
			throw (cudaGetErrorString(cudastat));
		}
	}


}

void cuMINQUE1::CheckGPU()
{
	CUdevice dev;
	int deviceCount;
	char deviceName[256];
	cuInit(0);
	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0)
    {
        throw ("Error: There are no available device(s) that support CUDA\n");
    }
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}
	if (deviceID>=deviceCount)
	{
		throw ("Error: illegal device ID.\n");
	}
	cuDeviceGetName(deviceName, 256, deviceID);
	printf("Using Device %d: \"%s\"\n", deviceID, deviceName);
}
