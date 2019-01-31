#include "pch.h"
#include "minque.h"
#include <iostream>


void MINQUE::importY(Eigen::VectorXd Y)
{
	this->Y = Y;
	nind = Y.size();
	Eigen::MatrixXd Identity = Eigen::MatrixXd(nind, nind);
	Identity.setZero();
	for (int i = 0; i < nind; i++) Identity(i, i) = 1.0;
	V.clear();
	V.insert(V.end(), Identity);
	nVi++;
}

void MINQUE::Vi_pushback(Eigen::MatrixXd Vi)
{
	V.insert(V.end(), Vi);
	nVi++;
}

void MINQUE::setGPU(bool isGPU)
{
	GPU = isGPU;
}


MINQUE::~MINQUE()
{
}

void MINQUE::Cal_Eta(int i)
{
	Eta = Eigen::VectorXd(nVi);
	Eigen::VectorXd ei = Eigen::VectorXd(nVi);
	ei.setZero();
	ei[i] = 1;
	if (GPU)
	{
		double *eta = (double *)malloc(nVi * sizeof(double));
		double *gammainv = Gamma_INV.data();
		double *vei = ei.data();
		cuMatrixMult(gammainv, vei, eta, nVi, nVi, 1);
		Eta = Eigen::Map<Eigen::MatrixXd>(eta, nVi, nVi);
	}
	else
	{
		Eta = Gamma_INV * ei;
	}
}

void MINQUE::Calc_Vsum_Inv()
{
	Eigen::MatrixXd ViSum = V.at(0);
	for (int i=0;i<nVi;i++)
	{
		ViSum += ViSum + V.at(i);
	}
	if (GPU)
	{
		double *p = ViSum.data();
		double *inv = (double *)malloc(nind*nind * sizeof(double));;
		int Rows = ViSum.rows();
		cuMatrixInv(p, inv, nind);
		Vsum_INV= Eigen::Map<Eigen::MatrixXd>(inv, Rows, Rows);
		free(inv);
	}
	else
	{
		Vsum_INV = ViSum.inverse();
	}
}

void MINQUE::Calc_Gamma()
{
	Gamma = Eigen::MatrixXd(nVi, nVi);
	Gamma.setZero();
	for (int i=0;i<nVi;i++)
	{
		for (int j = i; j<nVi;j++)
		{
			if (GPU)
			{
				double *Sumkernal = Vsum_INV.data();
				double *Ki = V.at(i).data();
				double *Kj = V.at(j).data();
				double *Ksum_inv_Ki = (double *)malloc(nind*nind * sizeof(double));
				double *Ksum_inv_Kj = (double *)malloc(nind*nind * sizeof(double));
				double *Ksum_inv_Ki_Ksum_inv_Kj = (double *)malloc(nind*nind * sizeof(double));
				cuMatrixMult(Sumkernal, Ki, Ksum_inv_Ki, nind, nind, nind);
				cuMatrixMult(Sumkernal, Kj, Ksum_inv_Kj, nind, nind, nind);
				cuMatrixMult(Ksum_inv_Ki, Ksum_inv_Kj, Ksum_inv_Ki_Ksum_inv_Kj, nind, nind, nind);
				for (int id=0;id<nind;id++)
				{
					Gamma(i, j) += Ksum_inv_Ki_Ksum_inv_Kj[id*nind + id];
				}
				free(Ksum_inv_Ki);
				free(Ksum_inv_Kj);
				free(Ksum_inv_Ki_Ksum_inv_Kj);
			}
			else
			{
				Gamma(i, j) = (Vsum_INV*V.at(i)*Vsum_INV*V.at(j)).trace();
			}
			Gamma(j, i) = Gamma(i, j);
		}
	}
	if (GPU)
	{
		double *inv = (double*)malloc(nVi*nVi * sizeof(double));
		double *org = Gamma.data();
		cuMatrixInv(org, inv, nVi);
		Gamma_INV = Eigen::Map<Eigen::MatrixXd>(inv, nVi, nVi);
		free(inv);
	}
	else
	{
		Gamma_INV = Gamma.inverse();
	}
}

void MINQUE::Calc_Ai(int i)
{
	Ai = Eigen::MatrixXd(nind, nind);
	Ai.setZero();
	Cal_Eta(i);
	double *inv=nullptr;
	double *ai=nullptr;
	if (GPU)
	{
		ai = (double *)malloc(nind*nind * sizeof(double));
		memset(ai, 0, nind*nind * sizeof(double));
		inv = Vsum_INV.data();
	}
	for (int id=0;id<nVi;id++)
	{
		if (GPU)
		{
			double *Ki = V.at(id).data();
			double *SumK_INV_Ki = (double *)malloc(nind*nind * sizeof(double));
			double *SumK_INV_Ki_SumK_INV = (double *)malloc(nind*nind * sizeof(double));
			memset(SumK_INV_Ki_SumK_INV, 0, nind*nind * sizeof(double));
			memset(SumK_INV_Ki, 0, nind*nind * sizeof(double));
			cuMatrixMult(inv, Ki, SumK_INV_Ki, nind, nind, nind);
			cuMatrixMult(SumK_INV_Ki, inv, SumK_INV_Ki_SumK_INV, nind, nind, nind);
			for (int j=0;j<nind*nind;j++)
			{
				ai[j] += Eta(id)*SumK_INV_Ki_SumK_INV[j];
			}
			free(SumK_INV_Ki_SumK_INV);
			free(SumK_INV_Ki);

		}
		else
		{
			Ai += Eta(id)*Vsum_INV*V.at(id)*Vsum_INV;
		}
	}
	if (GPU)
	{
		Ai = Eigen::Map<Eigen::MatrixXd>(ai, nind, nind);
		free(ai);
	}
}


void MINQUE::start()
{
	std::cout << "Start Sum of Kernel calculation" << std::endl;
	Calc_Vsum_Inv();
	std::cout << "Sum of Kernel was completed" << std::endl;
	std::cout << "Start Gamma Matrix calculation" << std::endl;
	Calc_Gamma();
	std::cout << "Start Gamma Matrix calculation completed" << std::endl;
	theta = Eigen::VectorXd(nVi);
	for (int i=0;i<nVi;i++)
	{
		std::cout << "Start Aj calculation" << std::endl;
		Calc_Ai(i);
		std::cout << "Start Aj calculation completed" << std::endl;
		theta[i] = Y.transpose()*Ai*Y;
		std::cout << "Variance of i: "<<theta[i] << std::endl;
	}
	std::cout << theta << std::endl;
}

