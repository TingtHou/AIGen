#include "../include/cuMINQUE0.h"




cuMINQUE0::cuMINQUE0(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomposition = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomposition = altDecompositionMode;
	CheckGPU();
}

cuMINQUE0::~cuMINQUE0()
{
}

void cuMINQUE0::estimateVCs()
{
	
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

	float* d_Ry;
	float* h_Ry; Eigen::VectorXf Ry(nind);
	h_Ry = Ry.data();
	//h_Ry = (float*)malloc(nind * sizeof(float));
	cudaMalloc((float**)&d_Ry, nind * sizeof(float));
	
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		throw (cudaGetErrorString(cudastat));
	}
	const float one = 1;
	const float zero = 0;
	const float minus_one = -1;

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
		cudaMemcpy(d_Ry, d_Y, nind*sizeof(float),cudaMemcpyDeviceToDevice);
	}
	else
	{
		float* d_Xt_X, * d_X_inv_XtX, * d_X_inv_XtX_Xt;
		
		cudaMalloc((float**)&d_Xt_X, ncov * ncov * sizeof(float));
		cudaMalloc((float**)&d_X_inv_XtX, nind * ncov * sizeof(float));
		cudaMalloc((float**)&d_X_inv_XtX_Xt, nind * nind * sizeof(float));
		cudaMemset(d_Xt_X, 0, ncov * ncov * sizeof(float));
		cudaMemset(d_X_inv_XtX, 0, ncov * nind * sizeof(float));
		cudaMemset(d_X_inv_XtX_Xt, 0, nind * nind * sizeof(float));
		cudaDeviceSynchronize();
		//Xt_X=Xt*X
		status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ncov, ncov, nind, &one, d_X, nind, d_X, nind, &zero, d_Xt_X, ncov);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		//inv(XtX)
		cuInverse(d_Xt_X, ncov, Decomposition, altDecomposition, allowPseudoInverse);
	//	cuToolkit::cuCholesky(d_Xt_X, ncov);
		//X_inv_XtX=X*inv(XtX)
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, ncov, ncov, &one, d_X, nind, d_Xt_X, ncov, &zero, d_X_inv_XtX, nind);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		//X_inv_XtX=X*inv(XtX)
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nind, nind, ncov, &one, d_X_inv_XtX, nind, d_X, nind, &zero, d_X_inv_XtX_Xt, nind);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		status = cublasSgemv(handle, CUBLAS_OP_N, nind, nind, &one, d_X_inv_XtX_Xt, nind, d_Y, 1, &zero, d_Ry, 1);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
		cuVectorCwiseMinus(d_Y, d_Ry, d_Ry,nind);
		float* d_tmp;
		cudaMalloc((float**)&d_tmp, nind * X.cols() * sizeof(float));
		for (int i = 0; i < nVi; i++)
		{	
			cudaMemset(d_tmp, 0, nind * X.cols() * sizeof(float));
			status = cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, nind, ncov, &one, d_Vi[i], nind, d_X_inv_XtX, nind, &zero, d_tmp, nind);
			cudaDeviceSynchronize();
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				throw (_cudaGetErrorEnum(status));
			}
			status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nind, nind, ncov, &one, d_tmp, nind, d_X, nind, &zero, d_RV[i], nind);
			cudaDeviceSynchronize();
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				throw (_cudaGetErrorEnum(status));
			}
			cuVectorCwiseMinus(d_Vi[i], d_RV[i], d_RV[i], nind * nind);
			status = cublasSgemv(handle, CUBLAS_OP_N, nind, nind, &one, d_Vi[i], nind, d_Ry, 1, &zero, d_Vi_y[i], 1);
			cudaDeviceSynchronize();
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				throw (_cudaGetErrorEnum(status));
			}
		}
		cudaFree(d_Xt_X);
		cudaFree(d_X_inv_XtX);
		cudaFree(d_X_inv_XtX_Xt);
	}

	float* d_Identity;
	Eigen::VectorXf h_u(nVi);
	Eigen::MatrixXf h_F(nVi, nVi);
	cudaMalloc((float**)&d_Identity, nind * sizeof(float));
	cuFillValue(d_Identity, nind, 1);
	cudaDeviceSynchronize();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

	for (int i = 0; i < nVi; i++)
	{
		cuVectorCwiseProduct(d_Ry, d_Vi_y[i], d_yVi_y[i], nind);
		status = cublasSdot(handle, nind, d_yVi_y[i], 1, d_Identity, 1, h_u.data() + i);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
	}
	cudaFree(d_Identity);
	cudaMalloc((float**)&d_Identity, nind * nind * sizeof(float));
	cuFillValue(d_Identity, nind * nind, 1);
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
			cudaMalloc((float**)&RVij, nind * nind * sizeof(float));
			cudaMemset(RVij, 0, nind * nind * sizeof(float));
			cuVectorCwiseProduct(d_RVi_t, d_RV[j], RVij, nind * nind);
			status = cublasSdot(handle, nind * nind, RVij, 1, d_Identity, 1, h_F.data() + i + nVi * j);
			cudaDeviceSynchronize();
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				throw (_cudaGetErrorEnum(status));
			}
			h_F(j, i) = h_F(i, j);
		}
	}
	cudaFree(d_Identity);
	Inverse(h_F, Decomposition, altDecomposition, allowPseudoInverse);
	vcs = h_F * h_u;
	cudaFree(d_Ry);
	for (int i = 0; i < nVi; i++)
	{
		cudaFree(d_RV[i]);
		cudaFree(d_Vi_y[i]);
		cudaFree(d_yVi_y[i]);
	}
}
