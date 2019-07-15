#include "pch.h"
#include "ToolKit.h"
#include <stdio.h>
#include <iostream>
void ToolKit::ArraytoVector(double ** a, int n, int m, vector<vector<double>>& v, bool Transpose)
{
	v.clear();
	if (Transpose)
	{
		v.resize(m);
		for (int i = 0; i < m; i++)
		{
			v[i].resize(n);
			for (int j = 0; j < n; j++)
			{
				v[i][j] = a[j][i];
			}
		}
	}
	else
	{
		v.resize(n);
		for (int i = 0; i < n; i++)
		{
			v[i].resize(m);
			for (int j = 0; j < m; j++)
			{
				v[i][j] = a[i][j];
			}
		}

	}
}


void ToolKit::Array2toArrat1(double **a, int n, int m, double *b, bool Colfirst)
{
	int k = 0;
	if (Colfirst)
	{
		for (int i=0;i<m;i++)
		{
			for (int j=0;j<n;j++)
			{
				b[k++] = a[j][i];
			}
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				b[k++] = a[i][j];
			}
		}
	}
}

void ToolKit::Vector2toArray1(vector<vector<double>>& v, double * b, bool Colfirst)
{
	int n = v.size();
	int m = v[0].size();
	int k = 0;
	if (Colfirst)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				b[k++] = v[j][i];
			}
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				b[k++] = v[i][j];
			}
		}
	}
}

void ToolKit::Stringsplit(string & org, vector<string>& splited, string delim)
{
	splited.clear();
	size_t current;
	size_t next = -1;
	do
	{
		current = next + 1;
		next = org.find_first_of(delim, current);
		splited.push_back(org.substr(current, next - current));
	} while (next != string::npos);
}

void ToolKit::dec2bin(int num, int *bin)
{
	for (int i = 8; i >= 0; i--) {
		if (num & (1 << i))
			bin[7 - i] = 1;
		else
			bin[7 - i] = 0;
	}
}

bool ToolKit::Inv_Cholesky(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix)
{
 	Eigen::LDLT<Eigen::MatrixXd> LDLT;
	Eigen::MatrixXd IdentityMatrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	IdentityMatrix.setIdentity();
	LDLT.compute(Ori_Matrix);
	if (!Ori_Matrix.isApprox(Ori_Matrix.transpose()) || LDLT.info() == Eigen::NumericalIssue)
	{
		return false;
	}
 	Inv_Matrix = LDLT.solve(IdentityMatrix);
//	std::cout << IdentityMatrix.isApprox(Ori_Matrix*Inv_Matrix) << std::endl;  // false
	bool a_solution_exists = (Ori_Matrix*Inv_Matrix).isApprox(IdentityMatrix,1e-10);
//	Inv_Matrix = Ori_Matrix.ldlt().solve(IdentityMatrix);
	return a_solution_exists;
}

bool ToolKit::Inv_LU(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix)
{
	Eigen::PartialPivLU<Eigen::MatrixXd> LU(Ori_Matrix);
	Eigen::MatrixXd IdentityMatrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	IdentityMatrix.setIdentity();
	Inv_Matrix = LU.inverse();
	bool a_solution_exists = (Ori_Matrix*Inv_Matrix).isApprox(IdentityMatrix, 1e-10);
	return a_solution_exists;
}

bool ToolKit::Inv_SVD(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix, bool allowPseudoInverse)
{

	auto svd = Ori_Matrix.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	const auto &singularValues = svd.singularValues();
	Eigen::MatrixXd singularValuesInv(Ori_Matrix.cols(), Ori_Matrix.rows());
	singularValuesInv.setZero();
	double  pinvtoler = 1.e-20; // choose your tolerance wisely
	bool singlar = false;

	for (unsigned int i = 0; i < singularValues.size(); ++i)
	{
		if (abs(singularValues(i)) > pinvtoler)
			singularValuesInv(i, i) = (double)1.0 / singularValues(i);
		else
		{
			if (!allowPseudoInverse)
			{
				return false;
			}
			singularValuesInv(i, i) = (double)0.0;
//			singlar = true;
		}
	}
	Inv_Matrix = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
	bool a_solution_exists = (Ori_Matrix*Inv_Matrix*Ori_Matrix).isApprox(Ori_Matrix, 1e-10);
	return a_solution_exists;
}

bool ToolKit::Inv_QR(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix, bool allowPseudoInverse)
{
	Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> QR;
	Eigen::MatrixXd IdentityMatrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	IdentityMatrix.setIdentity();
	QR.compute(Ori_Matrix);
	if (!QR.isInvertible()&&!allowPseudoInverse)
	{
		return false;
	}
	if (QR.isInvertible())
	{
		Inv_Matrix = QR.solve(IdentityMatrix);
	}
	else
	{
		Inv_Matrix = QR.pseudoInverse();
	}
	bool a_solution_exists = (Ori_Matrix*Inv_Matrix*Ori_Matrix).isApprox(Ori_Matrix, 1e-10);
	return a_solution_exists;
}

