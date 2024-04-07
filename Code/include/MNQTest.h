#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <vector>
#include "CommonFunc.h"
#include "MinqueBase.h"
#include <mkl.h>
#include <thread>
#include <boost/math/distributions.hpp>
#include "mkl_types.h"
#include "mkl_cblas.h"
#include "ToolKit.h"
#include "Davies.h"
class MINQUETest
{
public:
	MINQUETest(std::shared_ptr<MinqueBase> MNQobj);
	float MINQUE_Normal(Eigen::VectorXf weights, std::vector<int> TestingIndex);
	float MINQUE0_overall(Eigen::VectorXf weights, std::vector<int> TestingIndex);
	float MINQUE0_singleparameter(Eigen::VectorXf est_null, int TestingIndex);
protected:

	Eigen::VectorXf vcs;
	std::vector<std::shared_ptr<Eigen::MatrixXf>> Vi;
	Eigen::MatrixXf F;
	Eigen::VectorXf pvalue;
	int ThreadId = 0;

};