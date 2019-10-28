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
	Batch(std::vector<Eigen::MatrixXd> &kernels, Eigen::VectorXd &phe, int splitnum, int seed, bool isclear);
	void start();
	void GetBatchKernels(std::vector< std::vector<Eigen::MatrixXd>> &BatchedKernel);
	void GetBatchPhe(std::vector< Eigen::VectorXd>  &phe);
	~Batch();
private:
	std::vector<Eigen::MatrixXd> kernels;
	std::vector< std::vector<Eigen::MatrixXd>>  KernelBatched;
	std::vector< Eigen::VectorXd>  pheBatched;
	Eigen::VectorXd phe;
	int nInd = 0;
	int splitnum;
	int nkernels;
	unsigned int seed = 0;
	bool isclear = true;
	void shuffle();
	
};

