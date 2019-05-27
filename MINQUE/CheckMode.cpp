#include "pch.h"
#include "CheckMode.h"
#include <Eigen/Dense>
#include "ToolKit.h"
#include <time.h>
#include <iostream>
void CheckMatrixInverseMode()
{
	Eigen::MatrixXd Ori = Eigen::MatrixXd::Random(10,10);
	Ori = Ori * Ori.transpose();
//Chocky
	Eigen::MatrixXd Inv(10, 10);
	Eigen::MatrixXd IdentityMatrix(10, 10);
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
		bool is_identity = (Ori*Inv).isIdentity();
		if (is_identity)
		{
			std::cout << "Passed" << std::endl;
		}
		else
		{
			std::cout << "Failed" << std::endl;
			std::cout << "Original matrix:\n"<<Ori << std::endl;
			std::cout << "Inversed matrix:\n" << Inv << std::endl;
			std::cout << "Original Matrix * Inversed Matrix:\n"<< Ori * Inv << std::endl;
		}
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
		bool is_identity = (Ori*Inv).isIdentity();
		if (is_identity)
		{
			std::cout << "Passed" << std::endl;
		}
		else
		{
			std::cout << "Failed" << std::endl;
			std::cout << "Failed" << std::endl;
			std::cout << "Original matrix:\n" << Ori << std::endl;
			std::cout << "Inversed matrix:\n" << Inv << std::endl;
			std::cout << "Original Matrix * Inversed Matrix:\n" << Ori * Inv << std::endl;

		}
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
		bool is_identity = (Ori*Inv).isIdentity();
		if (is_identity)
		{
			std::cout << "Passed" << std::endl;
		}
		else
		{
			std::cout << "Failed" << std::endl;

			std::cout << "Original matrix:\n" << Ori << std::endl;
			std::cout << "Inversed matrix:\n" << Inv << std::endl;
			std::cout << "Original Matrix * Inversed Matrix:\n" << Ori * Inv << std::endl;

		}
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
		bool is_identity = (Ori*Inv).isIdentity();
		if (is_identity)
		{
			std::cout << "Passed" << std::endl;
		}
		else
		{
			std::cout << "Failed" << std::endl;
			std::cout << "Original matrix:\n" << Ori << std::endl;
			std::cout << "Inversed matrix:\n" << Inv << std::endl;
			std::cout << "Original Matrix * Inversed Matrix:\n" << Ori * Inv << std::endl;

		}
	}
}
