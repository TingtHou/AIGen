#pragma once
#include "Basis.h"
#include "Bspline.h"
#include "CommonFunc.h"
#include <torch/torch.h>
#include <Eigen/Dense>

class SplineBasis : public Basis
{
public:
	SplineBasis(int64_t n_basis, bool linear = true); //here linear is for being consistence with Haar
	void evaluate(Eigen::VectorXd& points);
	torch::Tensor pen_1d(torch::Tensor& param, double lamd1 = 1);
	torch::Tensor pen_2d(torch::Tensor& param, Basis& basis, double lamda0 = 1, double lamda1 = 1);
private:
	std::shared_ptr<Bspline> bss;
};