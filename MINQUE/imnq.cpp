#include "pch.h"
#include "imnq.h"
#include "LinearRegression.h"
#include "rln_mnq.h"
#include <iomanip>
void imnq::estimate()
{
	
	Iterate();
}

void imnq::setOptions(MinqueOptions mnqoptions)
{
	this->itr = mnqoptions.iterate;
	this->tol = mnqoptions.tolerance;
	this->Decomposition = mnqoptions.MatrixDecomposition;
	this->altDecomposition = mnqoptions.altMatrixDecomposition;
	this->allowpseudoinverse = mnqoptions.allowPseudoInverse;
}



Eigen::VectorXd imnq::initVCS()
{
	if (ncov == 0)
	{
		X.resize(nind, 1);
		X.setOnes();
		ncov++;
	}
	LinearRegression lr(Y, X, false);
	lr.MLE();
	double mse = lr.GetMSE();
	Eigen::VectorXd vc0(nVi);
	vc0.setZero();
	if (W.size()==0)
	{
		
		for (int i=0;i<nVi;i++)
		{
			vc0(i) = mse / (double)nVi;
		}
	}
	else
	{
		int j = 0;
		for (int i=ncov+1;i<=W.size();i++)
		{
			vc0(j++) = W(i);
		}
	}
	Eigen::VectorXd vc1(nVi);
	vc1.setZero();
	vc1(0) = mse;
	return vc0;
}

void imnq::Iterate()
{
	Eigen::VectorXd vc0,vc1(nVi);
	vc0  = initVCS();
	double diff = 0;
	rln_mnq *mnq=nullptr;
	while (itr>0)
	{
		mnq = new rln_mnq(Decomposition,altDecomposition,allowpseudoinverse);
		mnq->importY(Y);
		for (int i=0;i<nVi;i++)
		{
			mnq->pushback_Vi(Vi[i]);
		}
		if (ncov!=0)
		{
			mnq->puskback_X(X,false);
		}
		mnq->pushback_W(vc0);
		mnq->estimate();
		vc1=mnq->getvcs();
		Eigen::VectorXd tmp = vc1 - vc0;
		diff = (vc1 - vc0).cwiseAbs().maxCoeff();
		std::cout << std::fixed << std::setprecision(3) << "It: " << itr << "\t" << vc1.transpose() << "\tdiff: ";
		std::cout<< std::scientific <<diff << std::endl;
		if (diff<tol)
		{

			break;
		}
		vc0 = vc1;
		itr--;
	}
	vcs = mnq->getvcs();
	fix = mnq->getfix();
	delete mnq;
}
