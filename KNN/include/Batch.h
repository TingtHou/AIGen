#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include "CommonFunc.h"

//Generate subset data for each batch
class Batch
{
public:
	Batch(std::vector<Eigen::MatrixXf *> &kernels, Eigen::VectorXf &phe, Eigen::MatrixXf &Covs, int splitnum, int seed, bool isclear);
	void start(int dataType);
	void GetBatchKernels(std::vector< std::vector<Eigen::MatrixXf*>> &BatchedKernel);
	void GetBatchPhe(std::vector< Eigen::VectorXf>  &phe);
	void GetBatchCov(std::vector< Eigen::MatrixXf> & cov);
	Eigen::VectorXi getSizesofBatch() {	return nindInEachBatch;	};
	~Batch();
private:
	std::vector<Eigen::MatrixXf *> kernels;
	std::vector< std::vector<Eigen::MatrixXf>>  KernelBatched;
	std::vector< Eigen::VectorXf>  pheBatched;
	std::vector< Eigen::MatrixXf>  CovsBatched;
	Eigen::VectorXf phe;
	Eigen::MatrixXf Covs;
	int nInd = 0;
	int splitnum;
	int nkernels;
	unsigned int seed = 0;
	bool isclear = true;
	void shuffle();
	void shuffle_binary();
	Eigen::VectorXi nindInEachBatch;
};

