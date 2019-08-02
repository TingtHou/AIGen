#include "pch.h"
#include "imnq.h"
#include "LinearRegression.h"
#include "rln_mnq.h"
#include <sstream>
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

int imnq::getIterateTimes()
{
	return initIterate;
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
	logfile->write("Starting Iterate MINQUE Algorithm",true);
	while (initIterate <itr)
	{
	//	clock_t t1 = clock();
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
		mnq->setLogfile(logfile);
		mnq->estimate();
		vc1=mnq->getvcs();
		diff = (vc1 - vc0).squaredNorm() / vc0.squaredNorm();
//		std::stringstream ss;
// 		ss<<std::fixed << std::setprecision(3) << "It: " << initIterate << "\t" << vc1.transpose() << "\tdiff: ";
// 		ss << std::scientific << diff;
// 		logfile->write(ss.str(),true);
		if (diff<tol)
		{

			break;
		}
		vc0 = vc1;
		initIterate++;
	//	std::cout << "Iterate Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	}
	vcs = mnq->getvcs();
	fix = mnq->getfix();
	delete mnq;
	
}
