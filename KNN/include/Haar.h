#pragma once
#include <torch/torch.h>
#include <Eigen/Dense>
#include <cmath>
#include "CommonFunc.h"
#include "Basis.h"

class Haar : public Basis
{
public:
	Haar(int64_t n_basis, bool linear = true);
	void evaluate(Eigen::VectorXd &points);
	torch::Tensor pen_1d(torch::Tensor &param, double lamd1 = 1);
	torch::Tensor pen_2d(torch::Tensor &param, Basis &basis, double lamda0 = 1, double lamda1 = 1);

//	torch::Tensor pen2;
	
//	torch::Tensor mat;
private:
	int64_t level;
	

	Eigen::VectorXd psi(Eigen::VectorXd &x, int64_t j, int64_t k);
};