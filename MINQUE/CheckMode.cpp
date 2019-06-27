#include "pch.h"
#include "CheckMode.h"
#include <Eigen/Dense>
#include "ToolKit.h"
#include <time.h>
#include <iostream>
void CheckMatrixInverseMode()
{
	Eigen::MatrixXd Ori = Eigen::MatrixXd::Random(2000,2000);
	Ori = Ori * Ori.transpose();
//Chocky
	Eigen::MatrixXd Inv(Ori.rows(), Ori.cols());
	Eigen::MatrixXd IdentityMatrix(Ori.rows(), Ori.cols());
	IdentityMatrix.setIdentity();
	Inv.setZero();
	std::cout << "Checking Cholesky decomposition: ";

	bool status=ToolKit::Inv_Cholesky(Ori, Inv);
	if (!status)
	{
		std::cout << "Failed" << std::endl;
	}
	else
	{
		std::cout << "Passed" << std::endl;
	}
	/////////////////////////////////////////////
	Inv.setZero();
	std::cout << "Checking LU decomposition: ";
	status=ToolKit::Inv_LU(Ori, Inv);
	if (!status)
	{
		std::cout << "Failed" << std::endl;
	}
	else
	{
		std::cout << "Passed" << std::endl;
	}

	//////////////////////////////////////////////
	Inv.setZero();
	std::cout << "Checking QR decomposition: ";
	status=ToolKit::Inv_QR(Ori, Inv,true);
	if (!status)
	{
		std::cout << "Failed" << std::endl;
	}
	else
	{
		std::cout << "Passed" << std::endl;
	}
	

	Inv.setZero();
	std::cout << "Checking SVD decomposition: ";
	status=ToolKit::Inv_SVD(Ori, Inv, true);
	if (!status)
	{
		std::cout << "Failed" << std::endl;
	}
	else
	{
		std::cout << "Passed" << std::endl;
	}
	
}

void CheckkernelIOStream()
{

}