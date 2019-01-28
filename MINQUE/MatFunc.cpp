#include "pch.h"
#include "MatFunc.h"

void MatFunc::LUinversion(std::vector<std::vector<double>>& M, double limit)
{
	int rows = M.size();
	int cols = M[0].size();
	if (rows != cols) throw("Matrix is not a square matrix");
	Eigen::MatrixXd ori(rows, cols);
	for (int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
		{
			ori(i, j) = M[i][j];
		}
	}
	Eigen::MatrixXd inversion = ori.inverse();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (ori(i, j)<limit)
			{
				M[i][j] = 0;
			}
			else
			{
				M[i][j]= ori(i, j);
			} 
		}
	}
}
