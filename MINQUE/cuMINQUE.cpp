#include "pch.h"
#include "cuMINQUE.h"
#include <Eigen/Dense>
#include <fstream>
#include "ThreadPool.h"
cuMINQUE::cuMINQUE()
{
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) 
	{
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		return;
	}
}
cuMINQUE::~cuMINQUE()
{
	cudaFree(D_Y);
	cudaFree(D_V);
	cudaFree(D_Vsum_INV);
	cudaFree(D_Vsum);
	cudaFree(D_Identity);
	cudaFree(D_Gamma);
	cudaFree(D_Gamma_INV);
	cudaFree(D_Ai);
//	free(H_Y);
	free(H_V);
	free(H_Identity);
	free(H_Gamma);
	free(H_Ai);
	free(H_Eta);
	free(theta);
	D_Y = nullptr;
	D_V = nullptr;
	D_Vsum_INV = nullptr;
	D_Vsum = nullptr;
	D_Identity = nullptr;
	D_Gamma_INV = nullptr;
	D_Gamma = nullptr;
	D_Ai = nullptr;
	H_Y = nullptr;
	H_V = nullptr;
	H_Identity = nullptr;
	H_Gamma = nullptr;
	H_Ai = nullptr;
	H_Eta = nullptr;
	theta = nullptr;
	for (int i = 0; i < V.size(); i++)
	{
		cudaFree(V.at(i));
		V.at(i) = nullptr;
	}
	cublasDestroy(handle);

}

void cuMINQUE::import_Y(double * Y, int nind)
{
	const double alpha = 1.0;
	const double beta = 1.0;
	H_Y = Y;
	cudaMalloc((double**)&D_Y, nind * sizeof(double));
	cudaMemcpy(D_Y, H_Y, nind * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudastat) << std::endl;
	}
	this->nind = nind;
	H_Identity = (double*)malloc(nind*nind * sizeof(double));
	memset(H_Identity, 0, nind*nind * sizeof(double));
	for (int id = 0; id < nind; id++) H_Identity[id*nind + id] = 1;
	cudaMalloc((double**)&D_Identity, nind*nind * sizeof(double));
	cudastat = cudaMemcpy(D_Identity, H_Identity, nind*nind * sizeof(double), cudaMemcpyHostToDevice);
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudastat) << std::endl;
	}
	cudaDeviceSynchronize();
	V.clear();
	cudaMalloc((void**)&D_Vsum, nind*nind * sizeof(double));
	cudaMemset(D_Vsum, 0, nind*nind * sizeof(double));
	cudaDeviceSynchronize();
}

void cuMINQUE::push_back_Vi(double * Vi, int n)
{
	const double alpha = 1.0;
	const double beta = 1.0;
	if (n != nind) throw("The dimension of covariance matrix does not match the sample size");
	D_V = nullptr;
	cudaMalloc((double**)&D_V, nind*nind * sizeof(double));
	cudaMemcpy(D_V, Vi, nind*nind * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
//	cublasSetMatrix(nind, nind, sizeof(double), Vi, nind, D_V, nind);
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudaGetErrorString(cudastat) << std::endl;
	}

	V.push_back(D_V);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Identity, nind, D_V, nind, &beta, D_Vsum, nind);
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudastat << std::endl;
	}
	nVi++;
}

void cuMINQUE::init()
{
	cudaMalloc((void **)&D_Vsum_INV, nind*nind * sizeof(double));
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudastat << std::endl;
	}
	cudaMalloc((double**)&D_Gamma_INV, nVi*nVi * sizeof(double));
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudastat << std::endl;
	}
	cudaMemset(D_Gamma_INV, 0, nVi*nVi * sizeof(double));
	
	cudaMalloc((double**)&D_Gamma, nVi*nVi * sizeof(double));
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudastat << std::endl;
	}
	cudaMemset(D_Gamma, 0, nVi*nVi * sizeof(double));
	
	theta = (double*)malloc(nVi * sizeof(double));
	memset(theta, 0, nVi * sizeof(double));
	H_Gamma = (double*)malloc(nVi*nVi * sizeof(double));
	memset(H_Gamma, 0, nVi*nVi * sizeof(double));
	H_Eta = (double*)malloc(nVi * sizeof(double));
	memset(H_Eta, 0, nVi * sizeof(double));
}

int cuMINQUE::estimate()
{
	init();
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudastat << std::endl;
	}
	//////Calculate sum of Variables matrix////////
	cuToolkit::cuMatrixInv(D_Vsum, D_Vsum_INV, nind);
	cudaDeviceSynchronize();
	cudastat = cudaGetLastError();
	if (cudastat != cudaSuccess)
	{
		std::cout << cudastat << std::endl;
	}
	std::cout << "Calculate inversed matrix of SumV: Done" << std::endl;
// 	double *H_Vsum = (double*)malloc(nind*nind * sizeof(double));
// 	cudaMemcpy(H_Vsum, D_Vsum_INV, nind*nind * sizeof(double), cudaMemcpyDeviceToHost);
// 	Eigen::MatrixXd test = Eigen::Map<Eigen::MatrixXd>(H_Vsum, nind, nind);
// 	std::ofstream out;
// 	out.open("inverse.txt", std::ios::out);
// 	out << test << std::endl;
// 	out.close();
	cudaFree(D_Vsum);
	D_Vsum = nullptr;
	///////////Calculate Gamma Matrix////////////////////
	
	Calc_Gamma();
	std::cout << "Calculate  matrix of Gamma: Done" << std::endl;
	/////////////////////////////////////
	
	const double alpha = 1.0;
	const double beta = 0.0;
	double *D_AiY;
	cudaMalloc((double**)&D_AiY, nind * sizeof(double));
	double *H_AiY;
	H_AiY = (double*)malloc(nind * sizeof(double));
	cudaMalloc((double**)&D_Ai, nind*nind * sizeof(double));
	for (int i=0;i<nVi;i++)
	{
		Calc_Ai(i);
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, 1, nind, &alpha, D_Ai, nind, D_Y, nind, &beta, D_AiY, nind);
	//	cublasDgbmv(handle, CUBLAS_OP_N, nind, nind, 2, 1, &alpha, D_Ai, nind, D_Y, 1,&beta, D_AiY, 1);
		cudaMemcpy(H_AiY, D_AiY, nind * sizeof(double), cudaMemcpyDeviceToHost);
		for (int j=0;j<nind;j++)
		{
			theta[i] += H_AiY[j] * H_Y[j];
		}
	}
	cudaFree(D_AiY);
	free(H_AiY);
	H_AiY = nullptr;
	D_AiY = nullptr;
	/////////////////
	return 0;
}

std::vector<double> cuMINQUE::GetTheta()
{
	std::vector<double> retheta;
	retheta.clear();
	for (int i=0;i<nVi;i++)
	{
		retheta.push_back(theta[i]);
	}
	return retheta;
}


void cuMINQUE::Calc_Gamma()
{
	
// 	double *D_Ksum_inv_Ki_Ksum_inv_Kj=nullptr, *D_Ksum_inv_Ki=nullptr,*D_Ksum_inv_Kj=nullptr;
// 	double *H_Ksum_inv_Ki_Ksum_inv_Kj = nullptr;
// 	cudaMalloc((double**)&D_Ksum_inv_Ki_Ksum_inv_Kj, nind*nind * sizeof(double));
// 	cudaMalloc((double**)&D_Ksum_inv_Ki, nind*nind * sizeof(double));
// 	cudaMalloc((double**)&D_Ksum_inv_Kj, nind*nind * sizeof(double));
// 	cudaDeviceSynchronize();
//	H_Ksum_inv_Ki_Ksum_inv_Kj = (double*)malloc(nind*nind * sizeof(double));
// 	const double alpha = 1.0;
// 	const double beta = 0.0;
	ThreadPool tp(10);
	int k = 0;
	for (int i=0;i<nVi;i++)
	{
		for (int j=i;j<nVi;j++)
		{
			tp.enqueue([k](double *D_Vsum_INV, double *Vi, double *Vj, int nind, double* D_Gamma,int i, int j, int nVi) {
				double *D_Ksum_inv_Ki_Ksum_inv_Kj = nullptr, *D_Ksum_inv_Ki = nullptr, *D_Ksum_inv_Kj = nullptr;
				cudaMalloc((double**)&D_Ksum_inv_Ki_Ksum_inv_Kj, nind*nind * sizeof(double));
				cudaMalloc((double**)&D_Ksum_inv_Ki, nind*nind * sizeof(double));
				cudaMalloc((double**)&D_Ksum_inv_Kj, nind*nind * sizeof(double));
				const double alpha = 1.0;
				const double beta = 0.0;
				cublasHandle_t handle;
				cublasStatus_t status= cublasCreate(&handle);
				if (status != CUBLAS_STATUS_SUCCESS)
				{
					fprintf(stderr, "!!!! CUBLAS initialization error\n");
					return;
				}
				cudaDeviceSynchronize();
				cudaMemset(D_Ksum_inv_Ki_Ksum_inv_Kj, 0, nind*nind * sizeof(double));
				cudaDeviceSynchronize();
				status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Vsum_INV, nind, Vi, nind, &beta, D_Ksum_inv_Ki, nind);
				cudaDeviceSynchronize();
				cudaError_t cudastat = cudaGetLastError();
				if (cudastat != cudaSuccess)
				{
					std::cout << cudaGetErrorString(cudastat) << std::endl;
				}
				status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Vsum_INV, nind, Vj, nind, &beta, D_Ksum_inv_Kj, nind);
				cudaDeviceSynchronize();


				cudastat = cudaGetLastError();
				if (cudastat != cudaSuccess)
				{
					std::cout << cudaGetErrorString(cudastat) << std::endl;
				}
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Ksum_inv_Ki, nind, D_Ksum_inv_Kj, nind, &beta, D_Ksum_inv_Ki_Ksum_inv_Kj, nind);
				cudaDeviceSynchronize();
				cudastat = cudaGetLastError();
				if (cudastat != cudaSuccess)
				{
					std::cout << cudaGetErrorString(cudastat) << std::endl;
				}
				// 			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Vsum_INV, nind, init_V.at(i), nind, &beta, D_Ksum_inv_Ki_Ksum_inv_Kj, nind);
				// 
				// 			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Ksum_inv_Ki_Ksum_inv_Kj, nind, D_Vsum_INV, nind, &beta, D_Ksum_inv_Ki_Ksum_inv_Kj, nind);
				// 			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Ksum_inv_Ki_Ksum_inv_Kj, nind, init_V.at(j), nind, &beta, D_Ksum_inv_Ki_Ksum_inv_Kj, nind);
				int ijID = j * nVi + i;
				int jiID = i * nVi + j;
				//		cudaMemcpy(H_Ksum_inv_Ki_Ksum_inv_Kj, D_Ksum_inv_Ki_Ksum_inv_Kj, nind*nind * sizeof(double), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				cuMatrixTrace(D_Ksum_inv_Ki_Ksum_inv_Kj, nind, D_Gamma + ijID);
				// 			double d = -1;
				// 			cudaMemcpy(&d,D_Gamma + ijID, sizeof(double), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				cudaMemcpy(D_Gamma + jiID, D_Gamma + ijID, sizeof(double), cudaMemcpyDeviceToDevice);
				cudaDeviceSynchronize();
				cudaFree(D_Ksum_inv_Ki_Ksum_inv_Kj);
				cudaFree(D_Ksum_inv_Ki);
				cudaFree(D_Ksum_inv_Kj);
				D_Ksum_inv_Kj = nullptr;
				D_Ksum_inv_Ki = nullptr;
				D_Ksum_inv_Ki_Ksum_inv_Kj = nullptr;
			}, D_Vsum_INV, V.at(i), V.at(j), nind,D_Gamma, i, j, nVi);
			k++;
/*			cudastat = cudaGetLastError();
			if (cudastat != cudaSuccess)
			{
				std::cout << cudaGetErrorString(cudastat) << std::endl;
			}
			for (int id = 0; id < nind; id++)
			{
				H_Gamma[ijID] += H_Ksum_inv_Ki_Ksum_inv_Kj[id*nind + id];
			}
			int jiID = i * nVi + j;
			H_Gamma[jiID] = H_Gamma[ijID];
			*/
		}
	}
	tp.~ThreadPool();

//	free(H_Ksum_inv_Ki_Ksum_inv_Kj);
// 	Eigen::MatrixXd test = Eigen::Map<Eigen::MatrixXd>(H_Gamma, nVi, nVi);
// 	std::cout << test << std::endl;
//	cublasSetMatrix(nVi, nVi, sizeof(double), H_Gamma, nVi, D_Gamma, nVi);
//	cudaDeviceSynchronize();
	cuToolkit::cuMatrixInv(D_Gamma, D_Gamma_INV, nVi);	
	cudaDeviceSynchronize();
// 	double *H_Vsum = (double*)malloc(nVi*nVi * sizeof(double));
// 	cudaMemcpy(H_Vsum, D_Gamma_INV, nVi*nVi * sizeof(double), cudaMemcpyDeviceToHost);
// 	Eigen::MatrixXd test = Eigen::Map<Eigen::MatrixXd>(H_Vsum, nVi, nVi);
// 	std::cout << test << std::endl;

}

void cuMINQUE::Calc_Ai(int i)
{
	
	cudaMemset(D_Ai, 0, nind*nind * sizeof(double));
	Cal_Eta(i);
	const double alpha = 1.0;
	const double beta_add = 1.0;
	const double beta_multi = 0.0;
	double *SumK_INV_Ki_SumK_INV, *SumK_INV_Ki;
	cudaMalloc((double**)&SumK_INV_Ki_SumK_INV, nind*nind * sizeof(double));
	cudaMalloc((double**)&SumK_INV_Ki, nind*nind * sizeof(double));
	cudaMemset(SumK_INV_Ki_SumK_INV, 0, nind*nind * sizeof(double));
	cudaDeviceSynchronize();
	for (int id = 0; id < nVi; id++)
	{
		const double ai = H_Eta[id];
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, D_Vsum_INV, nind, V.at(id), nind, &beta_multi, SumK_INV_Ki, nind);
		cudaDeviceSynchronize();
	///////////Calculate Gamma Matrix////////////////////
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &alpha, SumK_INV_Ki, nind, D_Vsum_INV, nind, &beta_multi, SumK_INV_Ki_SumK_INV, nind);
		cudaDeviceSynchronize();
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nind, nind, nind, &ai, SumK_INV_Ki_SumK_INV, nind, D_Identity, nind, &beta_add, D_Ai, nind);
		cudaDeviceSynchronize();
	}
// 	double *H_Vsum = (double*)malloc(nind*nind * sizeof(double));
// 	cudaMemcpy(H_Vsum, D_Ai, nind*nind * sizeof(double), cudaMemcpyDeviceToHost);
// 	Eigen::MatrixXd test = Eigen::Map<Eigen::MatrixXd>(H_Vsum, nind, nind);
// 	std::ofstream out;
// 	out.open("Ai.txt", std::ios::out);
// 	out << test << std::endl;
// 	out.close();
	cudaFree(SumK_INV_Ki_SumK_INV);
	cudaFree(SumK_INV_Ki);
	SumK_INV_Ki = nullptr;
	SumK_INV_Ki_SumK_INV = nullptr;
}

void cuMINQUE::Cal_Eta(int i)
{
// 	free(H_Eta);
// 	
	memset(H_Eta, 0, nVi * sizeof(double));
	int cuPoint = i * nVi;
	cudaMemcpy(H_Eta, D_Gamma_INV + cuPoint, nVi * sizeof(double),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

}


