#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <vector>
#include "CommonFunc.h"

class MinqueBase
{
public:
	void importY(Eigen::VectorXf &Y);
	void pushback_Vi(Eigen::MatrixXf *vi);
	void pushback_X(Eigen::MatrixXf &X,bool intercept);
	void pushback_W(Eigen::VectorXf &W);
	void setThreadId(int Thread_id);
	virtual void estimateVCs()=0;
	void estimateFix(Eigen::VectorXf VCs_hat);
	Eigen::VectorXf getfix() { return fix; };
	Eigen::VectorXf getvcs() { return vcs; };
protected:
	int nind = 0;
	int nVi = 0;
	int ncov = 0;
	Eigen::VectorXf Y;
	Eigen::MatrixXf X;
	Eigen::VectorXf W;
	Eigen::MatrixXf VW;
	Eigen::VectorXf vcs;
	Eigen::VectorXf fix;
	std::vector<Eigen::MatrixXf*> Vi;

	int Decomposition = Cholesky;
	int altDecomposition = QR;
	bool allowPseudoInverse = true;
	int ThreadId = 0;
	bool iscancel=false;
	void CheckInverseStatus(std::string MatrixType,int status, bool allowPseudoInverse);
};

