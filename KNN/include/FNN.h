#pragma once
#include <torch/torch.h>
#include <vector>
#include "Layer.h"
#include "Haar.h"
#include "SplineBasis.h"
#include "TensorData.h"


class FNN : public torch::nn::Module
{
public:
	FNN(std::vector<int64_t> dims, double lamb = 1); 
	torch::Tensor forward(std::shared_ptr<TensorData> data);
	torch::Tensor penalty(int64_t nind);
//	torch::Tensor training(std::shared_ptr<TensorData> dataset, int64_t epoches=1e6);
	//void fit_init(std::shared_ptr<TensorData> data);
	//void fit_end();

	template<class T>
	void build(bool singleknot, int64_t ncovs=0)
	{

		std::string layer_name = "model";
		if (dims.size() == 1)
		{
			models.push_back(register_module<LayerD>("model0", std::make_shared<LayerD>(std::make_shared<T>(dims[0]))));
			//models.push_back(std::make_shared<LayerD>(new Haar(dims[0])));
			layers.push_back(4);
		}

		if (dims.size() == 2)
		{
			if (dims[1] == 1)
			{
				models.push_back(register_module<LayerC>("model0", std::make_shared<LayerC>(std::make_shared<T>(dims[0]))));
				//	models.push_back(std::make_shared<LayerC>( new Haar(dims[0]) ) );
				layers.push_back(3);
			}
			else
			{
				models.push_back(register_module<LayerB>("model0", std::make_shared<LayerB>(std::make_shared<T>(dims[0]), std::make_shared<T>(dims[1]),ncovs)));
				//models.push_back(std::make_shared<LayerB>(new Haar(dims[0]), new Haar(dims[1])));
				layers.push_back(2);
				models[models.size() - 1]->singleknot = singleknot;
			}

		}
		if (dims.size() > 2)
		{

			models.push_back(register_module<LayerB>("model0", std::make_shared<LayerB>(std::make_shared<T>(dims[0]), std::make_shared<T>(dims[1],false), ncovs, false)));
			//models.push_back(std::make_shared<LayerB>(new Haar(dims[0]), new Haar(dims[1],false),false));
			layers.push_back(2);
			int i = 1;
			for (; i < dims.size() - 2; i++)
			{
				std::string layeri = layer_name + std::to_string(i);
				models.push_back(register_module<LayerB>(layeri, std::make_shared<LayerB>(std::make_shared<T>(dims[i], false), std::make_shared<T>(dims[i + 1], false))));
				layers.push_back(2);
			}
			if (dims[dims.size() - 1] == 1)
			{
				std::string layeri = layer_name + std::to_string(i);
				models.push_back(register_module<LayerC>(layeri, std::make_shared<LayerC>(std::make_shared<T>(dims[dims.size() - 2], false))));
				layers.push_back(3);
			}
			else
			{
				std::string layeri = layer_name + std::to_string(i);
				models.push_back(register_module<LayerB>(layeri, std::make_shared<LayerB>(std::make_shared<T>(dims[dims.size() - 2], false), std::make_shared<T>(dims[dims.size() - 1]))));
				layers.push_back(2);
				models[models.size() - 1]->singleknot = singleknot;
			}
		}


	};
	
	int64_t epoch=0;
	torch::Tensor loss;
private:
	double lamb=1;
	std::vector<std::shared_ptr<Layer>> models;
	std::vector<int> layers;


	void realization(std::shared_ptr<TensorData> data);
	std::vector<torch::Tensor> LastLayer;
	std::vector<int64_t> dims;

};