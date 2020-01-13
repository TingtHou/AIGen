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

// void MinqueBase::setLogfile(LOG * logfile)
// {
// 	this->logfile = logfile;
// }

