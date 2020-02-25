#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <mkl.h>
#include <chrono>
#include "helper_cuda.h"
#include "cuToolkit.h"
#include "ToolKit.h"
#include "cuMinqueBase.h"
class cuMINQUE1 :
	public cuMinqueBase
{
public:
	cuMINQUE1(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse);
	~cuMINQUE1();
	void importY(Eigen::VectorXf &Y);
	void pushback_Vi(Eigen::MatrixXf &vi);
	void pushback_X(Eigen::MatrixXf &X, bool intercept);
	void pushback_W(Eigen::VectorXf& W);
	void estimateVCs();
	Eigen::VectorXf getfix() { return fix; };
	Eigen::VectorXf getvcs() { return vcs; };
private:
	int deviceID = 0;

	//pointer of host
	float* h_Y;
	float* h_X;
	float* h_W;
	float* f;
	float* h_Identity;
	std::vector<float*> h_Vi;
	float* h_tmp;
	Eigen::MatrixXf X;
	Eigen::MatrixXf fix;
	Eigen::MatrixXf vcs;
	//pointer of device
	float* d_Y;
	float* d_X;
	//float* d_vcs;
	//float* d_fix;
	float* d_Identity;
	std::vector<float*> d_Vi;

	cublasStatus_t status;
	cudaError_t cudastat;
	cublasHandle_t handle;
	int nVi;
	
	int nind = 0;
	int ncov = 0;
	int Decomposition = 0;
	int altDecomposition = 2;
	bool allowPseudoInverse = true;
	int ThreadId = 0;
	bool iscancel = false;
	void init();
	void CheckGPU();
	bool Iscovariate = false;
	bool Isweight = false;
};


