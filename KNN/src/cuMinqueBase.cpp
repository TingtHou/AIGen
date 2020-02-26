#include "..\include\cuMinqueBase.h"

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
