#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <vector>
#include "Random.h"
#include <fstream>
#include "CommonFunc.h"
#include "Toolkit.h"
class Prediction
{
public:
	Prediction(Eigen::VectorXf& Real_Y, std::vector<Eigen::MatrixXf>& Kernels, Eigen::VectorXf& vcs, Eigen::MatrixXf& X, Eigen::VectorXf& fixed, bool isbinary);
	float getMSE();
	float getCor();
	float getAUC();
private:
	Eigen::VectorXf Real_Y;
	Eigen::VectorXf Predict_Y;
	std::vector<Eigen::MatrixXf> Kernels;
	Eigen::VectorXf vcs;
	Eigen::MatrixXf X;
	Eigen::VectorXf fixed;
	bool isbinary;
	int nind=0;
	int ncvs=0;
	float mse=0;
	float cor=0;
	float auc = 0;
private:
	void GetpredictY();
	void calc_mse();
	void calc();
};