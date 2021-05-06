#pragma once
#include <Eigen/Dense>
#include <torch/torch.h>
class Basis
{
public:
	virtual void evaluate(Eigen::VectorXd & points) = 0;
	virtual torch::Tensor pen_1d(torch::Tensor &param, double lamd1 = 1) = 0;
	virtual torch::Tensor pen_2d(torch::Tensor &param, Basis &basis, double lamda0 = 1, double lamda1 = 1) = 0;
	int64_t n_basis;
	double length;
	torch::Tensor pen0;
	torch::Tensor pen2;
	torch::Tensor mat;
	bool linear;

};
