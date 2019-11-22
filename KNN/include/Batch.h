#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include "CommonFunc.h"
class Batch
{
public:
	Batch(std::vector<Eigen::MatrixXf> &kernels, Eigen::VectorXf &phe, int splitnum, int seed, bool isclear);
	void start();
	void GetBatchKernels(std::vector< std::vector<Eigen::MatrixXf>> &BatchedKernel);
	void GetBatchPhe(std::vector< Eigen::VectorXf>  &phe);
	~Batch();
private:
	std::vector<Eigen::MatrixXf> kernels;
	std::vector< std::vector<Eigen::MatrixXf>>  KernelBatched;
	std::vector< Eigen::VectorXf>  pheBatched;
	Eigen::VectorXf phe;
	int nInd = 0;
	int splitnum;
	int nkernels;
	unsigned int seed = 0;
	bool isclear = true;
	void shuffle();
	
};

