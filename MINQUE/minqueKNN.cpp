#include "pch.h"
#include "minqueKNN.h"


void MINQUEKNN::importY(Eigen::VectorXd Y)
{
	this->Y = Y;
	nind = Y.size();
}

void MINQUEKNN::importU_pushback(Eigen::MatrixXd Ui)
{
	U.insert(U.end(), Ui);
	nKernel++;
}

void MINQUEKNN::setGPU(bool isGPU)
{
	GPU = isGPU;
}

MINQUEKNN::MINQUEKNN()
{
	U = std::vector<Eigen::MatrixXd>();
}


MINQUEKNN::~MINQUEKNN()
{
}

void MINQUEKNN::Cal_Eta(int i)
{
	Eta = Eigen::VectorXd(nKernel);
	Eigen::VectorXd ei = Eigen::VectorXd(nKernel);
	ei.setZero();
	ei[i] = 1;
	if (GPU)
	{
		double *eta = (double *)malloc(nKernel * sizeof(double));
		double *gammainv = Gamma_INV.data();
		double *vei = ei.data();
		cuMatrixMult(gammainv, vei, eta, nKernel, nKernel, 1);
		Eta = Eigen::Map<Eigen::MatrixXd>(eta, nKernel, nKernel);
	}
	else
	{
		Eta = Gamma_INV * ei;
	}
}

void MINQUEKNN::Calc_KernelSum_Inv()
{
	Eigen::MatrixXd Kernelsum = U.at(0);
	for (int i=0;i<nKernel;i++)
	{
		Kernelsum += Kernelsum + U.at(i);
	}
	if (GPU)
	{
		double *p = Kernelsum.data();
		double *inv = (double *)malloc(nind*nind * sizeof(double));;
		int Rows = Kernelsum.rows();
		cuMatrixInv(p, inv, Rows);
		Kernelsum_INV= Eigen::Map<Eigen::MatrixXd>(inv, Rows, Rows);
		free(inv);
	}
	else
	{
		Kernelsum_INV = Kernelsum.inverse();
	}
}

void MINQUEKNN::Calc_Gamma()
{
	Gamma = Eigen::MatrixXd(nKernel, nKernel);
	Gamma.setZero();
	for (int i=0;i<nKernel;i++)
	{
		for (int j = i; j<nKernel;j++)
		{
			if (GPU)
			{
				double *Sumkernal = Kernelsum_INV.data();
				double *Ki = U.at(i).data();
				double *Kj = U.at(j).data();
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
				Gamma(i, j) = (Kernelsum_INV*U.at(i)*Kernelsum_INV*U.at(j)).trace();
			}
			Gamma(j, i) = Gamma(i, j);

		}
	}
	if (GPU)
	{
		double *inv = (double*)malloc(nKernel*nKernel * sizeof(double));
		double *org = Gamma.data();
		cuMatrixInv(org, inv, nind);
		Gamma_INV = Eigen::Map<Eigen::MatrixXd>(inv, nKernel, nKernel);
		free(inv);
	}
	else
	{
		Gamma_INV = Gamma.inverse();
	}
}

void MINQUEKNN::Calc_Ai(int i)
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
		inv = Kernelsum_INV.data();
	}
	for (int id=0;id<nKernel;id++)
	{
		if (GPU)
		{
			double *Ki = U.at(id).data();
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
			Ai += Eta(id)*Kernelsum_INV*U.at(id)*Kernelsum_INV;
		}
	}
	if (GPU)
	{
		Ai = Eigen::Map<Eigen::MatrixXd>(ai, nind, nind);
		free(ai);
	}
}


void MINQUEKNN::start()
{
	Calc_KernelSum_Inv();
	Calc_Gamma();
	theta = Eigen::VectorXd(nKernel);
	for (int i=0;i<nKernel;i++)
	{
		Calc_Ai(i);
		theta[i] = Y.transpose()*Ai*Y;
	}
}

