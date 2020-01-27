#include "../include/imnq.h"

void imnq::estimate()
{
// 	int nProcessors = omp_get_max_threads();
// 
// 	std::cout << nProcessors << std::endl;
// 	omp_set_num_threads(nProcessors);
	
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

void imnq::isEcho(bool isecho)
{
	this->isecho = isecho;

}

void imnq::SetMINQUE1(bool MINQUE1)
{
	this->MINQUE1 = MINQUE1;
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
	Eigen::VectorXf vc0,vc1(nVi);
	vc0  = initVCS();
	float diff = 0;
	rln_mnq *mnq=nullptr;
	LOG(INFO) << "Starting Iterate MINQUE Algorithm at thread "<<ThreadId;
	while (initIterate <itr)
	{
		mnq = new rln_mnq(Decomposition,altDecomposition,allowpseudoinverse);
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
		mnq->estimate();
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

			break;
		}

	}
	vcs = mnq->getvcs();
	fix = mnq->getfix();
	delete mnq;
	
}

void imnq::UseMINQUE1()
{
	Eigen::VectorXf vc0, vc1(nVi);
	vc0 = initVCS();
	rln_mnq mnq = rln_mnq(Decomposition, altDecomposition, allowpseudoinverse);

	LOG(INFO) << "Starting Iterate MINQUE Algorithm at thread " << ThreadId;
	mnq.importY(Y);
	for (int i = 0; i < nVi; i++)
	{
		mnq.pushback_Vi(Vi[i]);
	}
	if (ncov != 0)
	{
		mnq.pushback_X(X, false);
	}
	mnq.pushback_W(vc0);
	mnq.setThreadId(ThreadId);
	mnq.estimate();
	vc1 = mnq.getvcs();
	std::stringstream ss;
	ss << std::fixed << "Thread ID: " << ThreadId << std::setprecision(3) << "\tIt: " << initIterate << "\t" << vc1.transpose() << std::endl;
	if (isecho)
	{
		printf("%s\n", ss.str().c_str());
	}
	LOG(INFO) << ss.str();
	vcs = mnq.getvcs();
	fix = mnq.getfix();
}
