#include "../include/imnq.h"

void imnq::estimateVCs()
{
	IterateTimes=Iterate();
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
	return IterateTimes;
}

void imnq::isEcho(bool isecho)
{
	this->isecho = isecho;

}

/*
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
*/

int  imnq::Iterate()
{
	int initIterate = 0;
	Eigen::VectorXf vc0(nVi),vc1(nVi);
	vcs.resize(nVi);
	if (W.size()!=0)
	{
		vc0 = W;
	}
	else
	{
	
		vc0.setOnes();
	}
	LOG(WARNING) << " inital weight" << vc0;
	float diff = 0;
	//minque1 *mnq=nullptr;
	if (isecho)
	{
		printf("Starting Iterate MINQUE Algorithm at thread %d \n", ThreadId);
	}
	LOG(INFO) << "Starting Iterate MINQUE Algorithm at thread "<<ThreadId;
	bool isConverge = false;
	while (initIterate <itr)
	{
		LOG(WARNING) << "new mnq class for "<< initIterate;
		auto mnq = minque1(Decomposition,altDecomposition, allowPseudoInverse);
		mnq.importY(Y);
		for (int i=0;i<nVi;i++)
		{
			mnq.pushback_Vi(Vi[i]);
		}
		if (ncov!=0)
		{
			mnq.pushback_X(X,false);
		}
		mnq.pushback_W(vc0);
		mnq.setThreadId(ThreadId);
		mnq.estimateVCs();
		vc1=mnq.getvcs();
	    diff = (vc1 - vc0).squaredNorm() / vc0.squaredNorm();
		std::stringstream ss;
		ss << std::fixed << "Thread ID: " << ThreadId << std::setprecision(3) << "\tIt: " << initIterate << "\t" << vc1.transpose() << "\tdiff: ";
		ss << std::scientific << diff;
		LOG(INFO) << " vc0=vc1";
		vc0 = vc1;
		LOG(INFO) << "initIterate++";
		initIterate++;
		if (isecho)
		{
			std::cout << ss.str()+"\n";
		}
		LOG(INFO) << ss.str();
		if (diff<tol)
		{
			isConverge = true;
			LOG(INFO) << "Iterate Converged";
			break;
		}
		LOG(INFO) << "Finish iterate " << initIterate;
	}
	LOG(INFO) << "Iterate done" << initIterate;
	if (!isConverge)
	{
		std::stringstream ss;
		ss << "[Warning]: iteration not converged (stop after " << itr << " iterations). You can specify the option --iter to allow for more iterations\n";
		std::cout << ss.str();
		LOG(WARNING) << ss.str();
	}
	LOG(WARNING) << "vcs = vc0";
	vcs = vc0;
	return initIterate;
}
