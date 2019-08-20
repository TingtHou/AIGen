#include "pch.h"
#include "CheckMode.h"
#include <Eigen/Dense>
#include "ToolKit.h"
#include <time.h>
#include "KernelManage.h"
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

void checkkernel()
{
	KernelReader kreader("../sample/sub500_4098");
	kreader.read();
	KernelData kd = kreader.getKernel();
	std::vector<int> id;
	for (int i = 0; i < kd.fid_iid.size(); i++)
	{
		id.push_back(i);
	}
	std::random_shuffle(id.begin(), id.end());
	KernelData newkd;
	newkd.kernelMatrix.resize(id.size(), id.size());
	newkd.VariantCountMatrix.resize(id.size(), id.size());
	for (int i = 0; i < id.size(); i++)
	{
		for (int j = 0; j < id.size(); j++)
		{
			newkd.kernelMatrix(i, j) = kd.kernelMatrix(id[i], id[j]);
			newkd.VariantCountMatrix(i, j) = kd.VariantCountMatrix(id[i], id[j]);
		}
		auto it = kd.fid_iid.left.find(id[i]);
		newkd.fid_iid.insert({ i,  it->second});
	}
	KernelWriter kw(newkd);
	kw.write("../sample/RandomS");
}

void CheckkernelIOStream()
{

}