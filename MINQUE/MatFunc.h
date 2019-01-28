#pragma once
#include <vector>
#include<Eigen/dense>

class MatFunc
{
public:
	static void LUinversion(std::vector< std::vector<double>> &M, double limit);
};

