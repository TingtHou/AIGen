#pragma once
#include <Eigen/Dense>
#include <vector>
#include "Random.h"
#include <fstream>
#include "ToolKit.h"
class Prediction
{
public:
	Prediction(Eigen::VectorXf& Real_Y, std::vector<Eigen::MatrixXf>& Kernels, Eigen::VectorXf& vcs, Eigen::MatrixXf& X, Eigen::VectorXf& fixed);
	float getMSE();
	float getCor();
private:
	Eigen::VectorXf Real_Y;
	Eigen::VectorXf Predict_Y;
	std::vector<Eigen::MatrixXf> Kernels;
	Eigen::VectorXf vcs;
	Eigen::MatrixXf X;
	Eigen::VectorXf fixed;
	int nind;
	int ncvs;
	float mse;
	float cor;
private:
	void GetpredictY();
	void calc_mse();
	void calc_cor();
	void calc();
};