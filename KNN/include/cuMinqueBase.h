#pragma once
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
class cuMinqueBase
{
public:
	cuMinqueBase();
	~cuMinqueBase();
	void importY(Eigen::VectorXf& Y);
	void pushback_Vi(Eigen::MatrixXf& vi);
	void pushback_X(Eigen::MatrixXf& X, bool intercept);
	void pushback_W(Eigen::VectorXf& W);
	void importY(float* d_Y, int nind);
	void pushback_Vi(std::vector<float*> &d_Vi);
	void pushback_X(float* d_X, int ncov);
	void pushback_W(float* h_W);
	virtual void estimateVCs() = 0;
	Eigen::VectorXf getfix() { return fix; };
	Eigen::VectorXf getvcs() { return vcs; };
	void init();
protected:
	int deviceID = 0;

	//pointer of host
	float* h_Y=NULL;
	float* h_X = NULL;
	float* h_W = NULL;
	std::vector<float*> h_Vi;
	Eigen::MatrixXf X;
	Eigen::MatrixXf fix;
	Eigen::MatrixXf vcs;

	//pointer of device
	float* d_Y = NULL;
	float* d_X = NULL;
	//float* d_vcs;
	//float* d_fix;
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
	
	void CheckGPU();
	bool Iscovariate = false;
	bool Isweight = false;
};
