#pragma once
#include <torch/torch.h>
#include "Layer.h"
#include "TensorData.h"

class NN : public torch::nn::Module
{
public:
	NN(std::vector<int64_t> dims, double lamb = 1);
	void build(int64_t ncovs = 0);
	torch::Tensor forward(std::shared_ptr<TensorData> data);
	torch::Tensor penalty(int64_t nind);
	int64_t epoch = 0;
	torch::Tensor loss;
	//bool realize = false;
private:
	double lamb = 1;
	std::vector<int64_t> dims;
	std::vector<std::shared_ptr<LayerA>> models;
	std::vector<int> layers;
};

