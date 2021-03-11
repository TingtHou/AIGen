#include "../include/haar.h"

Haar::Haar(int64_t n_basis, bool linear)
{
	level = (int64_t)std::log2(n_basis);
	Eigen::MatrixXd pen2_Eigen(n_basis, n_basis);
	this->n_basis = n_basis + linear - 1;
	this->linear = linear;
	pen2_Eigen.setZero();
	pen2_Eigen(1, 1) = 4;
	for (int64_t level1 = 1; level1 < level; level1++)
	{
		int64_t base1 = std::pow(2, level1);
		for (int64_t k = 1; k < base1; k++)
		{
			pen2_Eigen(base1 + k - 1, base1 + k) = base1;
			pen2_Eigen(base1 + k, base1 + k) = 6 * base1;
		}
		pen2_Eigen(base1, base1) = 5 * base1;
		pen2_Eigen(2 * base1 - 1, 2 * base1 - 1) = 5 * base1;
		for (int64_t level2 = 0; level2 < level1; level2++)
		{
			int64_t base2 = std::pow(2, level2);
			int64_t interval = std::pow(2, level1 - level2 - 1);
			double val = std::pow(2, (double)level2 / 2 + (double)level1 / 2);
			for (int64_t m = 0; m < base2; m++)
			{
				int64_t row = base2 + m;
				if (m > 0)
				{
					pen2_Eigen(row, base1 + 2 * m * interval - 1) += val;
					pen2_Eigen(row, base1 + 2 * m * interval) += val;
				}
				pen2_Eigen(row, base1 + (2 * m + 1) * interval - 1) -= 2 * val;
				pen2_Eigen(row, base1 + (2 * m + 1) * interval) -= 2 * val;
				if (m < base2 - 1)
				{
					pen2_Eigen(row, base1 + 2 * (m + 1) * interval - 1) += val;
					pen2_Eigen(row, base1 + 2 * (m + 1) * interval) += val;
				}
			}

		}
	}
	Eigen::MatrixXd pen2_Eigen_f = pen2_Eigen + pen2_Eigen.transpose() - pen2_Eigen.diagonal().asDiagonal().toDenseMatrix();

	pen2_Eigen.resize(0, 0);
	pen2 = dtt::eigen2libtorch(pen2_Eigen_f);
	//auto type = pen2.dtype();
	pen2_Eigen_f.resize(0, 0);
	if (!linear)
	{
		using namespace  torch::indexing;
		pen2 = pen2.index({ Slice(1, None), Slice(1, None) });
	}
	
}

void Haar::evaluate(Eigen::VectorXd &points)
{
	Eigen::MatrixXd mat_Eigen(n_basis, points.size());
	mat_Eigen.setZero();
	int64_t index = 0;
	if (linear)
	{
		for (int64_t i = 0; i < mat_Eigen.cols(); i++)
		{
			mat_Eigen(0, i) = points[i] >= 0 && points[i] < 1 ? 1 : 0;
		}
		index = 1;
	}
	for (int64_t j = 0; j < level; j++)
	{
		int64_t l = std::pow(2, j);
		for (size_t k = 0; k < l; k++)
		{
			mat_Eigen.row(index++) << psi(points, j, k).transpose();
		}
	}
	mat_Eigen.transposeInPlace();
	mat = dtt::eigen2libtorch(mat_Eigen);
	mat_Eigen.resize(0, 0);
}

torch::Tensor Haar::pen_1d(torch::Tensor &param, double lamd1)
{
	torch::Tensor pen = param.matmul(pen2).matmul(param) * lamd1;
	return pen;
}

torch::Tensor Haar::pen_2d(torch::Tensor &param, Basis &basis, double lamda0, double lamda1)
{
	torch::Tensor pen = torch::trace((param.t()).matmul(pen2).matmul(param)) * lamda1 / std::pow((double)n_basis, 0.5) +
		torch::trace(param.matmul(basis.pen2).matmul(param.t())) * lamda0 / std::pow((double)basis.n_basis, 0.5);
	return pen;
}

Eigen::VectorXd Haar::psi(Eigen::VectorXd &x, int64_t j, int64_t k)
{
	Eigen::VectorXd y(x.size());
	for (int64_t i = 0; i < y.size(); i++)
	{
		y[i] = std::pow(2, (double)j / 2) * ((std::pow(2, j) * x[i] - k >= 0) && (std::pow(2, j) * x[i] - k) < 0.5 ? 1 : 0) -
			((std::pow(2, j) * x[i] - k >= 0.5) && (std::pow(2, j) * x[i] - k < 1) ? 1 : 0);
	}
	return y;
}
