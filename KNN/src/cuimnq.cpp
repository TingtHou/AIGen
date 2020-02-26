#include "..\include\cuimnq.h"

void cuimnq::setOptions(MinqueOptions mnqoptions)
{
	this->itr = mnqoptions.iterate;
	this->tol = mnqoptions.tolerance;
	this->Decomposition = mnqoptions.MatrixDecomposition;
	this->altDecomposition = mnqoptions.altMatrixDecomposition;
	this->allowPseudoInverse = mnqoptions.allowPseudoInverse;
}

int cuimnq::getIterateTimes()
{
	return initIterate;
}

void cuimnq::isEcho(bool isecho)
{
	this->isecho = isecho;
}

void cuimnq::Iterate()
{
	Eigen::VectorXf vc0(nVi), vc1(nVi);
	if (Isweight)
	{
		vc0= Eigen::Map<Eigen::VectorXf>(h_W,nVi);
	}
	else
	{
		vc0.setOnes();
	}
	float diff = 0;
	cuMINQUE1* mnq = nullptr;
//	LOG(INFO) << "Starting Iterate MINQUE Algorithm at thread " << ThreadId;
	while (initIterate < itr)
	{
		mnq = new cuMINQUE1(Decomposition, altDecomposition, allowPseudoInverse);
		mnq->importY(d_Y,nind);
		mnq->pushback_Vi(d_Vi);
		if (ncov != 0)
		{
			mnq->pushback_X(d_X,ncov);
		}
		mnq->pushback_W(vc0);
		mnq->estimateVCs();
		vc1 = mnq->getvcs();
		diff = (vc1 - vc0).squaredNorm() / vc0.squaredNorm();
		std::stringstream ss;
		ss << std::fixed <</* "Thread ID: " << ThreadId <<*/ std::setprecision(3) << "\tIt: " << initIterate << "\t" << vc1.transpose() << "\tdiff: ";
		ss << std::scientific << diff;
		vc0 = vc1;
		initIterate++;
		if (isecho)
		{
			printf("%s\n", ss.str().c_str());
		}
		LOG(INFO) << ss.str();
		if (diff < tol)
		{

			break;
		}

	}
	vcs = mnq->getvcs();
	//	mnq->estimateFix();
	//	fix = mnq->getfix();
	delete mnq;
}
