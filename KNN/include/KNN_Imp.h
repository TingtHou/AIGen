#pragma once
#include <Eigen/Dense>
#include "../include/KernelGenerator.h"
#include "../include/KernelExpansion.h"
#include "../include/Batch.h"
#include "../include/easylogging++.h"
#include "../include/ToolKit.h"
#include "../include/LinearRegression.h"
#include "../include/Random.h"
#include "../include/PlinkReader.h"
#include "../include/Options.h"
#include "../include/CommonFunc.h"
#include "../include/KernelManage.h"
#include "../include/DataManager.h"
#include "../include/imnq.h"
#include "../include/Prediction.h"
#include "../include/MINQUE0.h"
#include "../include/MNQTest.h"



void BatchMINQUE1(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, 
	Eigen::VectorXf& coefs, float& iterateTimes, int nsplit, int seed, int nthread, bool isecho, bool dataType);

std::shared_ptr<MinqueBase> cMINQUE1(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho);


void BatchMINQUE0(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, 
	Eigen::VectorXf& coefs, int nsplit, int seed, int nthread, bool dataType);

std::shared_ptr< MinqueBase>	 cMINQUE0(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs);
void Fixed_estimator(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho);