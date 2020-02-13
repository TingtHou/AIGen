#include "../include/MinqueBase.h"


void MinqueBase::importY(Eigen::VectorXf Y)
{
	this->Y = Y;
	nind = Y.size();
	Vi.clear();
	VW = Eigen::MatrixXf(nind, nind);
	VW.setZero();
}

void MinqueBase::pushback_Vi(Eigen::MatrixXf *vi)
{
	Vi.insert(Vi.end(), vi);
	nVi++;
}

void MinqueBase::pushback_X(Eigen::MatrixXf X, bool intercept)
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


void MinqueBase::pushback_W(Eigen::VectorXf W)
{
	int nelmt = W.size();
	assert(nVi == nelmt);
	this->W = W;
}

void MinqueBase::setThreadId(int Thread_id)
{
	ThreadId = Thread_id;
}

void MinqueBase::estimateFix()
{
	if (ncov==0)
	{
		return;
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
	for (int i = 0; i < nVi; i++)
	{
		float* pr_Vi = (*Vi[i]).data();
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, vcs[i], pr_Vi, nind, pr_Identity, nind, 1, pr_VW, nind);
	}
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), nind, 1, pr_VW, nind, pr_X, nind, 0, pr_B, nind);
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_B, nind, 0, pr_Xt_B, X.cols());
	//inv(XtB)
	int status = Inverse(Xt_B, Decomposition, altDecomposition, allowPseudoInverse);
	CheckInverseStatus(status);
	//inv_XtB_Bt=inv(XtB)*Bt
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, X.cols(), nind, X.cols(), 1, pr_Xt_B, X.cols(), pr_B, nind, 0, pr_inv_XtB_Bt, X.cols());
	fix = inv_XtB_Bt * Y;
}

void MinqueBase::CheckInverseStatus(int status)
{
	
	switch (status)
	{
	case 0:
		break;
	case 1:
		if (allowPseudoInverse)
		{
			stringstream ss;
			ss << "[Warning]: Thread ID: " << ThreadId
				<< "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead\n";
			printf("%s", ss.str().c_str());
			//			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
			LOG(WARNING) << "Thread ID: " << ThreadId << "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead";
		}
		else
		{
			stringstream ss;
			ss << "[Error]: Thread ID: " << ThreadId
				<< "\tcalculating inverse matrix is failed, and pseudo inverse matrix is not allowed\n";
			throw std::exception(logic_error(ss.str().c_str()));
		}
		break;
	case 2:
	{
		stringstream ss;
		ss << "[Error]: Thread ID: " << ThreadId
			<< "\tcalculating inverse matrix is failed, and pseudo inverse matrix is also failed\n";
		throw std::exception(logic_error(ss.str().c_str()));
	}
	break;
	default:
		stringstream ss;
		ss << "[Error]: Thread ID: " << ThreadId
			<< "\tunknown code [" << std::to_string(status) << "] from calculating inverse matrix.\n";
		throw std::exception(logic_error(ss.str().c_str()));
	}

}

// void MinqueBase::setLogfile(LOG * logfile)
// {
// 	this->logfile = logfile;
// }

