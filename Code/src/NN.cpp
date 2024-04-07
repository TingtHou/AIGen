#include "../include/NN.h"

NN::NN(std::vector<int64_t> dims, double lamb)
{
	this->dims = dims;
	this->lamb = lamb;
	models.clear();
	layers.clear();
}

void NN::build(int64_t ncovs)
{
	std::string layer_ = "model";
	int64_t i = 0;
	std::string layer_name = layer_ + std::to_string(i);
	models.push_back
	(
		register_module<LayerA>
		(
			layer_name, std::make_shared<LayerA>(dims[i], dims[i + 1],ncovs)
		)
	);
	layers.push_back(1);
	i++;
	for (; i < dims.size()-1; i++)
	{
		std::string layer_name = layer_ + std::to_string(i);
		models.push_back
		(
			register_module<LayerA>
			(
				layer_name, std::make_shared<LayerA>( dims[i], dims[i+1])
			)
		);
		layers.push_back(1);
	}

}

torch::Tensor NN::forward(std::shared_ptr<TensorData> data)
{
	torch::Tensor res = models[0]->forward(data->getX(),data->getZ());
	//	std::cout << res.index({torch::indexing::Slice(1,10)}) << std::endl;
	for (int i = 1; i < layers.size(); i++)
	{
	
		res = res.sigmoid();
		res = models[i]->forward(res);
	}
	switch (data->dataType)
	{
	case 0:
		res = res * data->std_y + data->mean_y;
		break;
	case 1:
		res = res.sigmoid();
		break;
	case 2:
		break;
	}

	//	std::cout << data->std_y << "\t" << data->mean_y;
	return res;
	//return res * data->std_y + data->mean_y;
}

torch::Tensor NN::penalty(int64_t nind)
{
	torch::Tensor penalty = torch::zeros(1);
	int i = 0;
	for (; i < layers.size() - 1; i++)
	{
		penalty += models[i]->pen()*1e-7;
	}
	penalty += models[i]->pen();
	return penalty * lamb;
}
