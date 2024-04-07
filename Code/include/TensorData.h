#pragma once
#include "CommonFunc.h"
#include <torch/torch.h>
#include <tuple>
class TensorData
{
public:
	TensorData(PhenoData &phe, GenoData &gene, CovData &cov);
	TensorData( std::shared_ptr<PhenoData> phe, std::shared_ptr<GenoData> gene, std::shared_ptr<CovData> cov);
	TensorData(torch::Tensor phe, torch::Tensor gene, torch::Tensor  cov, Eigen::VectorXd pos, Eigen::VectorXd loc, double loc0,double loc1);
//	TensorData(std::shared_ptr<PhenoData> phe, torch::Tensor gene, torch::Tensor  cov, Eigen::VectorXd pos, double loc0, double loc1);
	void setPos(Eigen::VectorXd pos);
	void setLoc(Eigen::VectorXd loc);
	void setBatchNum(int64_t Batch_Num);
//	void wide2long();
	//int64_t getNind(int64_t index) { return nknots[index]; };
//	TensorData(PhenoData& phe, GenoData& gene, CovData& cov, bool isBalance);
	std::shared_ptr<TensorData> getSample(int64_t index);
//	std::tuple<std::shared_ptr<TensorData>, std::shared_ptr<TensorData>> GetsubTrain(double ratio);
	torch::Tensor getY();
	torch::Tensor getX();
	torch::Tensor getZ();
	Eigen::VectorXd getPos()
	{
		return pos;
	}
	Eigen::VectorXd getLoc(int64_t index)
	{
		return phe->vloc[index].cast<double>();
	}
	Eigen::VectorXd getLoc()
	{
		return loc;
	}
	std::shared_ptr<TensorData> getBatch(int64_t index_batch);
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
	int64_t Batch_Num;
	std::vector<int64_t> nknots;  //the reponse number per individual;
private:
	torch::Tensor y;
	torch::Tensor x;
	torch::Tensor z;
	Eigen::VectorXd pos;
	Eigen::VectorXd loc;
	std::shared_ptr<PhenoData> phe;
	std::shared_ptr<GenoData> gene;
	std::shared_ptr<CovData> cov;
	std::vector<int64_t> Batch_index;
	std::vector<int64_t> nind_in_each_batch;
	//std::vector<int64_t> nknots;  //the reponse number per individual;
};
