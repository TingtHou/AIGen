#include "../include/SplineBasis.h"

SplineBasis::SplineBasis(int64_t n_basis, bool linear)
{
	this->n_basis = n_basis;
	bss = std::make_shared<Bspline>(4, n_basis);
	Bspline bss2 = bss->derivative(2);
	Eigen::MatrixXd coefficients = bss2.get_deriv_coeff();
	Eigen::MatrixXd pen2_eigen= bss2.Gram_matrix();
	Eigen::MatrixXd pen0_eigen = bss->Gram_matrix();
	pen2_eigen = coefficients * pen2_eigen * coefficients.transpose();
	pen0 = dtt::eigen2libtorch(pen0_eigen);
	pen2 = dtt::eigen2libtorch(pen2_eigen);
//	std::cout << pen0 << std::endl;
//	std::cout << pen2 << std::endl;
}

void SplineBasis::evaluate(Eigen::VectorXd& points)
{
	Eigen::MatrixXd mat_Eigen(n_basis, points.size());
	mat_Eigen.setZero();
	mat_Eigen = bss->evaluate(points);
	mat_Eigen.transposeInPlace();
	mat = dtt::eigen2libtorch(mat_Eigen);
}

torch::Tensor SplineBasis::pen_1d(torch::Tensor& param, double lamd1)
{
	torch::Tensor pen = param.matmul(pen2).matmul(param) * lamd1;
	return pen;
}

torch::Tensor SplineBasis::pen_2d(torch::Tensor& param, Basis& basis, double lamda0, double lamda1)
{
	torch::Tensor pen = torch::trace(basis.pen0.matmul((param.t().matmul(pen2)).matmul(param))) * lamda1 +
		torch::trace(pen0.matmul((param.matmul(basis.pen2)).matmul(param.t()))) * lamda0;
	return pen;
}
