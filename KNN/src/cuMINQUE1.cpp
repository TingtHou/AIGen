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

void cuMINQUE1::estimateVCs()
{
	float* d_Identity;
	cudaMalloc((float**)&d_Identity, nind * nind * sizeof(float));
	cuAsDiag(d_Identity, nind);
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

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
	cudaFree(d_Identity);
	cuInverse(d_VW, nind, Decomposition, altDecomposition, allowPseudoInverse);
	//cuToolkit::cuCholesky(d_VW, nind);
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
		cuInverse(d_Xt_B, ncov, Decomposition, altDecomposition, allowPseudoInverse);
		//cuToolkit::cuCholesky(d_Xt_B, ncov);
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
	Eigen::VectorXf h_u(nVi);
	Eigen::MatrixXf h_F(nVi, nVi);
	cudaMalloc((float**)&d_Identity, nind * sizeof(float));
	cuFillValue(d_Identity, nind,1);
	cudaDeviceSynchronize();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

	for (int i = 0; i < nVi; i++)
	{
		cuVectorCwiseProduct(d_Ry, d_Vi_y[i],d_yVi_y[i],nind);
		status = cublasSdot(handle, nind, d_yVi_y[i], 1, d_Identity, 1, h_u.data()+i);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(status));
		}
	}
	cudaFree(d_Identity);
	cudaMalloc((float**)&d_Identity, nind*nind * sizeof(float));
	cuFillValue(d_Identity, nind* nind, 1);
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
			status = cublasSdot(handle, nind * nind, RVij, 1, d_Identity, 1, h_F.data()+i + nVi * j);
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
	//ToolKit::comput_inverse_logdet_LU_mkl(h_F);
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
