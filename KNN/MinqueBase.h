#pragma once
#include <Eigen/Dense>
#include <vector>
#include "cLog.h"
class MinqueBase
{
public:
	void importY(Eigen::VectorXd Y);
	void pushback_Vi(Eigen::MatrixXd vi);
	void puskback_X(Eigen::MatrixXd X,bool intercept);
	void pushback_W(Eigen::VectorXd W);
	virtual void estimate()=0;
	Eigen::VectorXd getfix() { return fix; };
	Eigen::VectorXd getvcs() { return vcs; };
	void setLogfile(LOG *logfile);
protected:
	int nind = 0;
	int nVi = 0;
	int ncov = 0;
	Eigen::VectorXd Y;
	Eigen::MatrixXd X;
	Eigen::VectorXd W;
	Eigen::MatrixXd VW;
	Eigen::VectorXd vcs;
	Eigen::VectorXd fix;
	std::vector<Eigen::MatrixXd> Vi;
	LOG *logfile;

};

