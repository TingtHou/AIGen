#include "pch.h"
#include "minque.h"
#include <iostream>
#include <fstream>
void MINQUE::importY(Eigen::VectorXd Y)
{
	this->Y = Y;
	nind = Y.size();
	V.clear();
}

void MINQUE::pushback_Vi(Eigen::MatrixXd Vi)
{
	V.insert(V.end(), Vi);
	nVi++;
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
	Eta = Gamma_INV * ei;

}

void MINQUE::Calc_Vsum_Inv()
{
	Eigen::MatrixXd ViSum = V.at(0);
	for (int i=1;i<nVi;i++)
	{
		ViSum +=  V.at(i);
	}
	Vsum_INV = ViSum.inverse();

}

void MINQUE::Calc_Gamma()
{
	Gamma = Eigen::MatrixXd(nVi, nVi);
	Gamma.setZero();
	for (int i=0;i<nVi;i++)
	{
		for (int j = i; j<nVi;j++)
		{
			Gamma(i, j) = (Vsum_INV*V.at(i)*Vsum_INV*V.at(j)).trace();
			Gamma(j, i) = Gamma(i, j);
		}
	}
	Gamma_INV = Gamma.inverse();

}

void MINQUE::Calc_Ai(int i)
{
	Ai = Eigen::MatrixXd(nind, nind);
	Ai.setZero();
	Cal_Eta(i);
	for (int id=0;id<nVi;id++)
	{
		Ai += Eta(id)*Vsum_INV*V.at(id)*Vsum_INV;
	}

}


void MINQUE::estimate()
{
	Calc_Vsum_Inv();
	Calc_Gamma();
	theta = Eigen::VectorXd(nVi);
	for (int i=0;i<nVi;i++)
	{
		Calc_Ai(i);
		theta[i] = Y.transpose()*Ai*Y;
	}
}

Eigen::VectorXd  MINQUE::Gettheta()
{
	return theta;
}

