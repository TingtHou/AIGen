#include "../include/imnq.h"

void imnq::estimateVCs()
{
	Iterate();
}

void imnq::setOptions(MinqueOptions mnqoptions)
{
	this->itr = mnqoptions.iterate;
	this->tol = mnqoptions.tolerance;
	this->Decomposition = mnqoptions.MatrixDecomposition;
	this->altDecomposition = mnqoptions.altMatrixDecomposition;
	this->allowPseudoInverse = mnqoptions.allowPseudoInverse;
}

int imnq::getIterateTimes()
{
	return initIterate;
}

void imnq::isEcho(bool isecho)
{
	this->isecho = isecho;

}


Eigen::VectorXf imnq::initVCS()
{
	if (ncov == 0)
	{
		X.resize(nind, 1);
		X.setOnes();
		ncov++;
	}
	LinearRegression lr(Y, X, false);
	lr.MLE();
	float mse = lr.GetMSE();
	Eigen::VectorXf vc0(nVi);
	vc0.setZero();
	if (W.size()==0)
	{
		
		for (int i=0;i<nVi;i++)
		{
			vc0(i) = mse / (float)nVi;
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
	Eigen::VectorXf vc1(nVi);
	vc1.setZero();
	vc1(0) = mse;
	return vc0;
}

void imnq::Iterate()
{
	Eigen::VectorXf vc0(nVi),vc1(nVi);
	if (W.size()!=0)
	{
		vc0 = W;
	}
	else
	{
	
		vc0.setOnes();
	}
	float diff = 0;
	minque1 *mnq=nullptr;
	if (isecho)
	{
		printf("Starting Iterate MINQUE Algorithm at thread %d \n", ThreadId);
	}
	LOG(INFO) << "Starting Iterate MINQUE Algorithm at thread "<<ThreadId;
	bool isConverge = false;
	while (initIterate <itr)
	{
		mnq = new minque1(Decomposition,altDecomposition, allowPseudoInverse);
		mnq->importY(Y);
		for (int i=0;i<nVi;i++)
		{
			mnq->pushback_Vi(Vi[i]);
		}
		if (ncov!=0)
		{
			mnq->pushback_X(X,false);
		}
		mnq->pushback_W(vc0);
		mnq->setThreadId(ThreadId);
		mnq->estimateVCs();
		vc1=mnq->getvcs();
	    diff = (vc1 - vc0).squaredNorm() / vc0.squaredNorm();
		std::stringstream ss;
		ss << std::fixed << "Thread ID: " << ThreadId << std::setprecision(3) << "\tIt: " << initIterate << "\t" << vc1.transpose() << "\tdiff: ";
		ss << std::scientific << diff;
		vc0 = vc1;
		initIterate++;
		if (isecho)
		{
			printf("%s\n", ss.str().c_str());
		}
		LOG(INFO) << ss.str();
		if (diff<tol)
		{
			isConverge = true;
			break;
		}

	}
	if (!isConverge)
	{
		std::stringstream ss;
		ss << "[Error]: iteration not converged (stop after " << itr << " iterations). You can specify the option --iter to allow for more iterations\n";
		throw std::string(ss.str().c_str());
	}
	vcs = mnq->getvcs();
//	mnq->estimateFix();
//	fix = mnq->getfix();
	delete mnq;
	
}
