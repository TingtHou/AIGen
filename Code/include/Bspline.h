#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
class Bspline
{
public:
	Bspline(int order, int n_basis);
	Bspline(int order, Eigen::VectorXd knots);
	Eigen::MatrixXd operator() (Eigen::VectorXd points);
	Eigen::MatrixXd operator() (double points);
	Eigen::MatrixXd evaluate(Eigen::VectorXd points);
	Eigen::MatrixXd evaluate(double points);
	Eigen::MatrixXd Gram_matrix();
	Bspline derivative(int order);
	void setCoefficients(Eigen::MatrixXd coeff);
	Eigen::MatrixXd get_deriv_coeff() { return coefficients; };

private:
	int n_basis;
	int order;
	int nknots;
	int degree;
	Eigen::VectorXd inknots;
	Eigen::VectorXd knots;
	Eigen::MatrixXd coefficients;

	double basis(double x, int degree, int i);
	Eigen::MatrixXd Calc_deriv_coeff();
};