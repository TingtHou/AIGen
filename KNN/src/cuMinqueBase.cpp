#include "../include/cuMinqueBase.h"


cuMinqueBase::cuMinqueBase()
{

}

cuMinqueBase::~cuMinqueBase()
{
	cudaFree(d_Y);
	cudaFree(d_X);
	for (int i = 0; i < nVi; i++)
	{
		cudaFree(d_Vi[i]);
	}

}

void cuMinqueBase::importY(Eigen::VectorXf& Y)
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

void cuMinqueBase::pushback_Vi(Eigen::MatrixXf& vi)
{
	h_Vi.push_back(vi.data());
	nVi++;
}

void cuMinqueBase::pushback_X(Eigen::MatrixXf& X, bool intercept)
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

void cuMinqueBase::pushback_W(Eigen::VectorXf& W)
{
	this->h_W = W.data();
	Isweight = true;
}

void cuMinqueBase::importY(float* d_Y, int nind)
{
	this->d_Y = d_Y;
	this->nind = nind;
}

void cuMinqueBase::pushback_Vi(std::vector<float*> &d_Vi)
{
	this->d_Vi = d_Vi;
	nVi = d_Vi.size();
}

void cuMinqueBase::pushback_X(float* d_X, int ncov)
{
	this->d_X = d_X;
	this->ncov = ncov;
}

void cuMinqueBase::pushback_W(float* h_W)
{
	this->h_W = h_W;
}

void cuMinqueBase::estimateFix()
{
	if (ncov == 0)
	{
		return;
	}
	float* fixed;
	fix.resize(ncov);
	cudaMalloc((float**)&fixed,  ncov * sizeof(float));
	float* d_Identity;
	cudaMalloc((float**)&d_Identity, nind * nind * sizeof(float));
	cuAsDiag(d_Identity, nind);
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

	float* d_VW;
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

	const float one = 1;
	const float zero = 0;
	const float minus_one = -1;
	for (int i = 0; i < nVi; i++)
	{
		cublasStatus_t cublasstatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &vcs[i], d_Vi[i], nind, d_Identity, nind, &one, d_VW, nind);
		if (cublasstatus != CUBLAS_STATUS_SUCCESS)
		{
			throw (_cudaGetErrorEnum(cublasstatus));
		}
	}
	cudaFree(d_Identity);

	cuInverse(d_VW, nind, Decomposition, altDecomposition, allowPseudoInverse);

	float* d_B, * d_Xt_B, * d_inv_XtB_Bt;
	cudaMalloc((float**)&d_B, nind * ncov * sizeof(float));
	cudaMalloc((float**)&d_Xt_B, ncov * ncov * sizeof(float));
	cudaMalloc((float**)&d_inv_XtB_Bt, ncov * nind * sizeof(float));
	cudaMemset(d_B, 0, nind * ncov * sizeof(float));
	cudaMemset(d_Xt_B, 0, ncov * ncov * sizeof(float));
	cudaMemset(d_inv_XtB_Bt, 0, ncov * nind * sizeof(float));
	cudaDeviceSynchronize();
	//B=inv(VW)*X
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, ncov, nind, &one, d_VW, nind, d_X, nind, &zero, d_B, nind);
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
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ncov, nind, ncov, &one, d_Xt_B, ncov, d_B, nind, &zero, d_inv_XtB_Bt, ncov);
	cudaDeviceSynchronize();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}

	status = cublasSgemv(handle, CUBLAS_OP_N, ncov, nind, &one, d_inv_XtB_Bt, ncov, d_Y, 1, &zero, fixed, 1);
	cudaDeviceSynchronize();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(status));
	}
	cudastat=cudaMemcpy(fix.data(), fixed, ncov * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	if (cudastat != CUBLAS_STATUS_SUCCESS)
	{
		throw (_cudaGetErrorEnum(cudastat));
	}

	cudaFree(fixed);
	cudaFree(d_B);
	cudaFree(d_Xt_B);
	cudaFree(d_inv_XtB_Bt);
}

void cuMinqueBase::init()
{
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


void cuMinqueBase::CheckGPU()
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
	if (deviceID >= deviceCount)
	{
		throw ("Error: illegal device ID.\n");
	}
	cuDeviceGetName(deviceName, 256, deviceID);
	printf("Using Device %d: \"%s\"\n", deviceID, deviceName);
}


void cuMinqueBase::CheckInverseStatus(int status)
{

	switch (status)
	{
	case 0:
		break;
	case 1:
		if (allowPseudoInverse)
		{
			stringstream ss;
			ss << "Calculating inverse matrix is failed, using pseudo inverse matrix instead\n";
			printf("%s", ss.str().c_str());
			//			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
			LOG(WARNING) << "Calculating inverse matrix is failed, using pseudo inverse matrix instead";
		}
		else
		{
			stringstream ss;
			ss << "[Error]: calculating inverse matrix is failed, and pseudo inverse matrix is not allowed\n";
			throw std::exception(logic_error(ss.str().c_str()));
		}
		break;
	case 2:
	{
		stringstream ss;
		ss << "[Error]: calculating inverse matrix is failed, and pseudo inverse matrix is also failed\n";
		throw std::exception(logic_error(ss.str().c_str()));
	}
	break;
	default:
		stringstream ss;
		ss << "[Error]: unknown code [" << std::to_string(status) << "] from calculating inverse matrix.\n";
		throw std::exception(logic_error(ss.str().c_str()));
	}

}