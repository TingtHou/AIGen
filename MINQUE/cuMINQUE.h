#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <cusolverDn.h>
#include <vector>
#include "cuToolkit.h"
extern "C" void cuMatrixTrace(const double *A, const int nrows, double *Result);
class cuMINQUE
{
public:
	cuMINQUE();
	~cuMINQUE();
	bool importY(double *Y, int nind);
	bool pushback_Vi(double *Vi,int n);
	bool estimate();
	std::vector<double> GetTheta();
private:
	cublasStatus_t status;
	cudaError_t cudastat;
	cublasHandle_t handle;
	double nVi = 0;;
// 	const double alpha = 1.0;
// 	const double beta = 0.0;
	/////////////

	double *H_Y = nullptr;
	double *H_V = nullptr;
	double *H_Identity = nullptr;
	double *H_Gamma = nullptr;
	double *H_Ai = nullptr;
	double *H_Eta = nullptr;
	std::vector<double *> V;
	int nind=0;
	
	double *D_V = nullptr;
	double *D_Y = nullptr;
	double *D_Vsum_INV = nullptr;
	double *D_Vsum = nullptr;
	double *D_Identity = nullptr;
	double *D_Gamma = nullptr;
	double *D_Gamma_INV = nullptr;
	double *theta = nullptr;
	double *D_Ai = nullptr;
	
private:
	bool init();
	bool Calc_Gamma();
	bool Calc_Ai(int i);
	bool Cal_Eta(int i);
};

