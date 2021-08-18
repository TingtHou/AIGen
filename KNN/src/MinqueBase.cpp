#include "../include/MinqueBase.h"


void MinqueBase::importY(Eigen::VectorXf &Y)
{
	this->Y = Y;
	nind = Y.size();
	Vi.clear();
}

void MinqueBase::pushback_Vi(Eigen::MatrixXf *vi)
{
	Vi.insert(Vi.end(), vi);
	nVi++;
}

void MinqueBase::pushback_X(Eigen::MatrixXf &X, bool intercept)
{
	int nrows = X.rows();
	int ncols = X.cols();
	assert(nrows == nind);
	if (ncov == 0)
	{
		if (intercept)
		{
			this->X.resize(nind, ncols + 1);
			Eigen::VectorXf IDt(nind);
			IDt.setOnes();
			this->X << IDt, X;
		}
		else
		{
			this->X = X;
		}
		
	}
	else
	{
		Eigen::MatrixXf tmpX = this->X;
		if (intercept)
		{
			this->X.resize(nind, ncov + ncols - 1);
			this->X << tmpX, X.block(0, 1, nrows, ncols - 1);
		}
		else
		{
			this->X.resize(nind, ncov + ncols);
			this->X << tmpX, X;
		}
		
	}
	ncov = this->X.cols();
}


void MinqueBase::pushback_W(Eigen::VectorXf &W)
{
	int nelmt = W.size();
	assert(nVi == nelmt);
	this->W = W;
}

void MinqueBase::setThreadId(int Thread_id)
{
	ThreadId = Thread_id;
}

void MinqueBase::estimateFix(Eigen::VectorXf VCs_hat)
{
	if (ncov==0)
	{
		return;
	}
	if (VW.size()==0)
	{
		VW = Eigen::MatrixXf(nind, nind);
		VW.setZero();
	}
	Eigen::MatrixXf B(nind, X.cols());
	Eigen::MatrixXf Xt_B(X.cols(), X.cols());
	Eigen::MatrixXf inv_XtB_Bt(X.cols(), nind);
	Eigen::MatrixXf Identity(nind, nind);
	Identity.setIdentity();
	fix.resize(ncov);
	float* pr_X = X.data();
	float* pr_Y = Y.data();
	float* pr_B = B.data();
	float* pr_Xt_B = Xt_B.data();
	float* pr_VW = VW.data();
	float* pr_Identity = Identity.data();
	float* pr_inv_XtB_Bt = inv_XtB_Bt.data();
	//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	LOG(INFO) << "Clac VW in fix effect estimation.";
	for (int i = 0; i < nVi; i++)
	{
		float* pr_Vi = (*Vi[i]).data();
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, VCs_hat[i], pr_Vi, nind, pr_Identity, nind, 1, pr_VW, nind);
	}
	Identity.resize(0, 0);
	LOG(INFO) << "Inverse VW in fix effect estimation.";
	int status = Inverse(VW, Decomposition, altDecomposition, allowPseudoInverse);
	CheckInverseStatus("V matrix", status, allowPseudoInverse);
	LOG(INFO) << "calc B=inv(VW)*X";
	//B=inv(VW)*X
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), nind, 1, pr_VW, nind, pr_X, nind, 0, pr_B, nind);
	//XtB=Xt*B
	LOG(INFO) << "calc XtB=Xt*B";
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_B, nind, 0, pr_Xt_B, X.cols());
	//inv(XtB)
	LOG(INFO) << "inverse XtB";
	status = Inverse(Xt_B, Decomposition, SVD, false);
	LOG(WARNING) << "Check inverse status";
	CheckInverseStatus("P matrix", status, false);
	//inv_XtB_Bt=inv(XtB)*Bt
	LOG(INFO) << "calc inv_XtB_Bt=inv(XtB)*Bt";
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, X.cols(), nind, X.cols(), 1, pr_Xt_B, X.cols(), pr_B, nind, 0, pr_inv_XtB_Bt, X.cols());
	fix = inv_XtB_Bt * Y;
}

void MinqueBase::CheckInverseStatus(std::string MatrixType, int status, bool allowPseudoInverse)
{
	
	switch (status)
	{
	case 0:
		break;
	case 1:
		if (allowPseudoInverse)
		{
			LOG(WARNING) << 1;
			std::stringstream ss;
			ss << "[Warning]: Thread ID: " << ThreadId
				<< "\t"<< MatrixType<<": Calculating inverse matrix fails, using pseudo inverse matrix instead\n";
			std::cout << ss.str();
		//	printf("%s", ss.str().c_str());
			//			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
			LOG(WARNING) << ss.str();
		}
		else
		{
			LOG(WARNING) << 10;
			std::stringstream ss;
			ss << "[Error]: Thread ID: " << ThreadId
				<< "\t" << MatrixType << ": calculating inverse matrix fails, and pseudo inverse matrix is not allowed\n";
			throw  std::string(ss.str());
		}
		break;
	case 2:
	{
		LOG(WARNING) << 2;
		std::stringstream ss;
		ss << "[Error]: Thread ID: " << ThreadId
			<< "\t" << MatrixType << ": calculating inverse matrix fails, and pseudo inverse matrix is also failed\n";
		throw  std::string(ss.str());
	}
	break;
	default:
		LOG(WARNING) << 3;
		std::stringstream ss;
		ss << "[Error]: Thread ID: " << ThreadId
			<< "\t" << MatrixType << ": unknown code [" << std::to_string(status) << "] from calculating inverse matrix.\n";
		throw  std::string(ss.str());
	break;
	}

}

// void MinqueBase::setLogfile(LOG * logfile)
// {
// 	this->logfile = logfile;
// }

