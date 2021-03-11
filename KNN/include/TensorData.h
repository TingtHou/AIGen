#pragma once
#include "CommonFunc.h"
#include <torch/torch.h>

class TensorData
{
public:
	TensorData(PhenoData& phe, GenoData& gene, CovData& cov);
	TensorData(torch::Tensor phe, torch::Tensor gene, torch::Tensor  cov, Eigen::VectorXd pos, Eigen::VectorXd loc);
	void setPos(Eigen::VectorXd pos);
	void setLoc(Eigen::VectorXd loc);
//	TensorData(PhenoData& phe, GenoData& gene, CovData& cov, bool isBalance);
	std::shared_ptr<TensorData> getSample(int64_t index);
	torch::Tensor getY();
	torch::Tensor getX();
	torch::Tensor getZ();
	Eigen::VectorXd getPos()
	{
		return pos;
	}
	Eigen::VectorXd getLoc()
	{
		return loc;
	}
	void setMean_STD(torch::Tensor mean, torch::Tensor std);
	double pos0 = -9;
	double pos1 = -9;
	double loc0 = -9;
	double loc1 = -9;
	torch::Tensor std_y;
	torch::Tensor mean_y;
	int64_t nind;
	bool isBalanced = true; // define balance if the knots of y are same, imbalance if the lenght of knots of y are different or the value of knots are different.
	int dataType=0;
private:
	torch::Tensor y;
	torch::Tensor x;
	torch::Tensor z;
	Eigen::VectorXd pos;
	Eigen::VectorXd loc;
	std::shared_ptr<PhenoData> phe;
	std::shared_ptr<GenoData> gene;
	std::shared_ptr<CovData> cov;

};
