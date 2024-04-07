#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <algorithm>
#include "CommonFunc.h"
class KernelExpansion
{
public:
	//KernelExpansion(std::vector<Eigen::MatrixXf> &MatrixHList, int dimension);
	KernelExpansion(std::vector<KernelData> *kernels, int dimension);
	KernelExpansion(std::vector<std::shared_ptr<KernelData>> kernels, int dimension);
	~KernelExpansion();
	std::vector<std::shared_ptr<Eigen::MatrixXf>> GetExtendMatrix();
	void test();
private:
	int dimension;
	int KernelCount;
	int nrow = 0, ncol = 0;
	std::vector<std::shared_ptr<Eigen::MatrixXf>> OriKernelList;
	std::vector<std::shared_ptr<Eigen::MatrixXf>> ExtendedMatrix;
	void Expanse(int degree, std::vector<std::shared_ptr<Eigen::MatrixXf>>EMatrix);
	void GetFullIndex(int degree, int M, int *Index, std::vector<std::vector<int>> &Comb);
	void GetUniqueIndex(std::vector<std::vector<int>> &Comb, std::vector<std::vector<int>> &UniqueComb);
};

