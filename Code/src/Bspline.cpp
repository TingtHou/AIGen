#include "../include/Bspline.h"

Bspline::Bspline(int order, int n_basis)
{
	this->order = order;
	this->degree = order - 1;
	this->n_basis = n_basis;
	this->nknots = n_basis - order;
	coefficients.resize(n_basis, n_basis);
	coefficients.setIdentity();
	inknots.resize(nknots);
	knots.resize(nknots+2*order);
	knots.setZero();
	inknots.resize(nknots);
	int i = order;
	int j = 0;
	double step = 1.0 / (double)(nknots+1);
	for (; i < order+nknots; i++)
	{
		knots[i] = step * (double)(i - order + 1);
		inknots[j++] = knots[i];
	}
	for (; i < knots.size(); i++)
	{
		knots[i] = 1;
	}

//	std::cout << "order: " << order << "\t n_basis: " << n_basis << "\n internal knots: \n" << inknots.transpose() << "\n knots: \n" << knots.transpose() << std::endl;
}

Bspline::Bspline(int order, Eigen::VectorXd knots)
{
	this->order = order;
	this->degree = order - 1;
	this->inknots = knots.block(1,0, knots.size()-2,1);
	this->nknots = inknots.size();
	this->n_basis = this->nknots +order;
	coefficients.resize(n_basis, n_basis);
	coefficients.setIdentity();
	int i = 0;
	this->knots.resize(nknots + 2 * order);
	for (; i < order; i++)
	{
		this->knots[i] = knots[0];
	}
	int j = 0;
	for (; i < order+nknots; i++)
	{
		this->knots[i] = inknots[j++];
	}
	for (; i < nknots + 2 * order; i++)
	{
		this->knots[i] = knots[knots.size()-1];
	}
	//std::cout << "order: " << order << "\t n_basis: " << n_basis << "\n internal knots: \n" << inknots.transpose() << "\n knots: \n" << this->knots.transpose() << std::endl;
}

Eigen::MatrixXd Bspline::operator()(Eigen::VectorXd points)
{
	return evaluate(points);
}

Eigen::MatrixXd Bspline::operator()(double points)
{
	return evaluate(points);
}

Eigen::MatrixXd Bspline::evaluate(Eigen::VectorXd points)
{
	Eigen::MatrixXd BasisMatrix(n_basis, points.size());
	for (int i = 0; i < n_basis; i++)
	{
		for (int j = 0; j < points.size(); j++)
		{
			BasisMatrix(i, j) = basis(points[j], order - 1, i);
		}
	}

	return BasisMatrix;
}

Eigen::MatrixXd Bspline::evaluate(double points)
{
	Eigen::MatrixXd BasisMatrix(n_basis, 1);
	for (int i = 0; i < n_basis; i++)
	{

		BasisMatrix(i, 0) = basis(points, order - 1, i);

	}

	return BasisMatrix;
}

Eigen::MatrixXd Bspline::Gram_matrix()
{
//	using boost::math::quadrature::trapezoidal;
	using boost::math::quadrature::gauss_kronrod;
	Eigen::MatrixXd Gram(n_basis, n_basis);
	for (long long k = 0; k < (n_basis + 1) * n_basis / 2; k++)
	{
		int tmp_K = k;
		int i = tmp_K / n_basis, j = tmp_K % n_basis;
		if (j < i) i = n_basis - i, j = n_basis - j - 1;
		int degree=order-1;
		auto f = [i, j, degree, this](double x) {
			return basis(x, degree, i) * basis(x, degree, j);
		};
		//double tol =;
	//	int max_refinements = 20;
	//	double integ = trapezoidal(f, 0.0, 1.0, 1e-20);
		double error;
		double integ = gauss_kronrod<double, 61>::integrate(f, 0,1,5, 1e-9, &error);
		Gram(i, j) = Gram(j, i) = integ;
	}
	return Gram;
}

Bspline Bspline::derivative(int order)
{
	Eigen::VectorXd innerknots_w_boundary=knots.block(this->order-1, 0, nknots + 2, 1);
	Bspline deriv(this->order - order, innerknots_w_boundary);
	Eigen::MatrixXd coeff(n_basis, n_basis);
	coeff.setIdentity();
	for (size_t i = 0; i < order; i++)
	{
		Bspline deriv_tmp(this->order - i, innerknots_w_boundary);
		coeff = coeff * deriv_tmp.Calc_deriv_coeff();
	}
	deriv.setCoefficients(coeff);
	return deriv;
}



void Bspline::setCoefficients(Eigen::MatrixXd coeff)
{
	this->coefficients = coeff;
}

double Bspline::basis(double x, int degree, int i)
{
	double y;
	if (degree == 0)
	{
		if (x < knots[i + 1] & x >= knots[i])
			y = 1;
		else
			y = 0;
	}
	else
	{
		double temp1, temp2;
		if (abs(knots[degree+i]-knots[i])<1e-6)
		{
			temp1 = 0;
		}
		else
		{
			temp1 = (x - knots[i]) / (knots[degree + i] - knots[i]);
		}
		if (abs(knots[i + degree + 1] - knots[i + 1]) < 1e-6)
		{
			temp2 = 0;
		}
		else
		{
			temp2 = (knots[i + degree + 1] - x) / (knots[i + degree + 1] - knots[i + 1]);
		}
		y = temp1 * basis(x, degree - 1, i) + temp2 * basis(x, degree - 1, i + 1);
	}
	return(y);
}

Eigen::MatrixXd Bspline::Calc_deriv_coeff()
{
	Eigen::MatrixXd temp(n_basis, n_basis + 1);
	temp.setZero();
	for (size_t i = 0; i < n_basis; i++)
	{
		if (abs(knots[i + degree] - knots[i]) < 1e-6)
		{
			temp(i, i) = 0;
		}
		else
		{
			temp(i, i) = (double)(degree) / (knots[i + degree] - knots[i]);
		}
		if (abs(knots[i + degree + 1] - knots[i + 1]) < 1e-6)
		{
			temp(i, i + 1) = 0;
		}
		else
		{
			temp(i, i + 1) = -(double)(degree) / (knots[i + degree + 1] - knots[i + 1]);
		}

	}
	Eigen::MatrixXd coeff = temp.block(0, 1, n_basis, n_basis - 1);

	return coeff;
}
