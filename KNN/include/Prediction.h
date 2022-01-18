#pragma once
//#define MKL_INT size_t
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <vector>
#include <mkl.h>
#include "Random.h"
#include <fstream>
#include "CommonFunc.h"
#include "Toolkit.h"
#include "easylogging++.h"
class Prediction
{
public:
	Prediction(Eigen::VectorXf& Real_Y, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& vcs, Eigen::MatrixXf& X, Eigen::VectorXf& fixed, bool isbinary, int mode,float ratio=1);
	Eigen::VectorXf getPredictY() { return Predict_Y; };
private:
	Eigen::VectorXf Real_Y;
	Eigen::VectorXf Predict_Y;
	std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels;
	Eigen::VectorXf vcs;
	Eigen::MatrixXf X;
	Eigen::VectorXf fixed;
	bool isbinary;
	int nind=0;
	int ncvs=0;
	float mse=0;
	float cor=0;
	float auc = 0;
	int mode=0;
	float ratio=1;
private:
	void predictWithKernel();
	void PredictYLOO();
	void predictBLUP();
};