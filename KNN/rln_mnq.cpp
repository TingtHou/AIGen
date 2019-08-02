#include "pch.h"
#include "rln_mnq.h"
#include "ToolKit.h"
#include <iostream>
#include <fstream>
#include <iomanip>

rln_mnq::rln_mnq(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomp = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomp = altDecompositionMode;
}

void rln_mnq::estimate()
{
	vcs.resize(nVi);
	if (W.size()==0)
	{
		W = Eigen::VectorXd(nVi);
		#pragma omp parallel for
		for (int i=0;i<nVi;i++)
		{
			W[i] = 1 / (double)nVi;
		}
	}
	for (int i = 0; i < nVi; i++)
	{
		VW += W[i] * Vi[i];
	}
	Eigen::MatrixXd Inv_VW(nind,nind);
	Inv_VW.setZero();
	int status=Inverse(VW, Inv_VW, Decomp, altDecomp, allowPseudoInverse);
	//ToolKit::Inv_SVD(VW, Inv_VW,true);
	switch (status)
	{
	case 0:
		break;
	case 1:
		if (allowPseudoInverse)
		{
			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
		}
		else
		{
			throw ("Error: calculating inverse matrix is failed, and pseudo inverse matrix is not allowed");
		}
		break;
	case 2:
		throw("Error: calculating inverse matrix is failed, and pseudo inverse matrix is also failed");
		break;
	default:
		throw("Error: unknown code [" + std::to_string(status) + "] from calculating inverse matrix.");
	}
	std::vector<Eigen::MatrixXd> RV;
	Eigen::VectorXd Ry;
	Eigen::MatrixXd C;
	if (ncov == 0)
	{
		for (int i = 0; i < nVi; i++)
		{
			Eigen::MatrixXd Inv_VW_Vi = Inv_VW * Vi[i];
			RV.push_back(Inv_VW_Vi);
		}
		Ry = Inv_VW * Y;
	}
	else
	{
		Eigen::MatrixXd B = Inv_VW * X;
		Eigen::MatrixXd Xt_B = X.transpose()*B;
		Eigen::MatrixXd INV_Xt_B(Xt_B.rows(), Xt_B.cols());
		int status = Inverse(Xt_B, INV_Xt_B, Decomp, altDecomp, allowPseudoInverse);
		switch (status)
		{
		case 0:
			break;
		case 1:
			if (allowPseudoInverse)
			{
				logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
			}
			else
			{
				throw ("Error: calculating inverse matrix is failed, and pseudo inverse matrix is not allowed");
			}
			break;
		case 2:
			throw("Error: calculating inverse matrix is failed, and pseudo inverse matrix is also failed");
			break;
		default:
			throw("Error: unknown code [" + std::to_string(status) + "] from calculating inverse matrix.");
		}
	//	ToolKit::Inv_SVD(Xt_B, INV_Xt_B,true);
		C = INV_Xt_B * B.transpose();
		Eigen::MatrixXd P = X * C;
		for (int i=0;i<nVi;i++)
		{
			RV.push_back(Inv_VW*(Vi[i] - P * Vi[i]));
		}
		Ry = Inv_VW * (Y - P * Y);
	}

	Eigen::VectorXd u(nVi);
	u.setZero();
	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		Eigen::VectorXd Vi_Ry = Vi[i]*Ry;
		u[i] = Ry.cwiseProduct(Vi_Ry).sum();
	}

	Eigen::MatrixXd F(nVi, nVi);
	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		for (int j=i;j<nVi;j++)
		{
			F(i, j) = RV[i].cwiseProduct(RV[j]).sum();
			F(j, i) = F(i, j);
		}
	}
	Eigen::MatrixXd INV_F(nVi, nVi);
	int status2 = Inverse(F, INV_F, Decomp, altDecomp, allowPseudoInverse);
	switch (status2)
	{
	case 0:
		break;
	case 1:
		if (allowPseudoInverse)
		{
			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
		}
		else
		{
			throw ("Error: calculating inverse matrix is failed, and pseudo inverse matrix is not allowed");
		}
		break;
	case 2:
		throw("Error: calculating inverse matrix is failed, and pseudo inverse matrix is also failed");
		break;
	default:
		throw("Error: unknown code [" + std::to_string(status) + "] from calculating inverse matrix.");
	}

//	ToolKit::Inv_SVD(F, INV_F,true);
	
	vcs = INV_F * u;
	if (ncov)
	{
		fix = C * Y;
	}
}

rln_mnq::~rln_mnq()
{
}



