#include "../include/FNN.h"


FNN::FNN(std::vector<int64_t> dims, double lamb)
{
	this->lamb = lamb;
	this->dims = dims;
//	models.resize(dims.size() - 1);
	/*
	std::string layer_name="model";
	if (dims.size()==1)
	{
		models.push_back(register_module<LayerD>("model0", std::make_shared<LayerD>(new Haar(dims[0]))));
		//models.push_back(std::make_shared<LayerD>(new Haar(dims[0])));
		layers.push_back(4);
	}
	
	if (dims.size()==2)
	{
		if (dims[1]==1)
		{
			models.push_back(register_module<LayerC>("model0", std::make_shared<LayerC>(new Haar(dims[0]))));
		//	models.push_back(std::make_shared<LayerC>( new Haar(dims[0]) ) );
			layers.push_back(3);
		}
		else
		{
			models.push_back(register_module<LayerB>("model0", std::make_shared<LayerB>(new Haar(dims[0]), new Haar(dims[1]))));
			//models.push_back(std::make_shared<LayerB>(new Haar(dims[0]), new Haar(dims[1])));
			layers.push_back(2);
		}

	}
	if (dims.size()>2)
	{
		
		models.push_back(register_module<LayerB>("model0", std::make_shared<LayerB>(new Haar(dims[0]), new Haar(dims[1], false), false)));
		//models.push_back(std::make_shared<LayerB>(new Haar(dims[0]), new Haar(dims[1],false),false));
		layers.push_back(2);
		int i = 1;
		for (; i < dims.size()-2; i++)
		{
			std::string layeri = layer_name + "i";
			models.push_back(register_module<LayerB>(layeri, std::make_shared<LayerB>(new Haar(dims[i],false), new Haar(dims[i + 1], false))));
			layers.push_back(2);
		}
		if (dims[dims.size()-1]==1)
		{
			std::string layeri = layer_name + "i";
			models.push_back(register_module<LayerC>(layeri, std::make_shared<LayerC>(new Haar(dims[dims.size() - 2],false))));
			layers.push_back(3);
		}
		else
		{
			std::string layeri = layer_name + "i";
			models.push_back(register_module<LayerB>(layeri, std::make_shared<LayerB>(new Haar(dims[dims.size() - 2], false), new Haar(dims[dims.size() - 1]))));
			layers.push_back(2);
		}
	}
	
	*/
}

torch::Tensor FNN::forward(std::shared_ptr<TensorData> data)
{
	realization(data);
	std::shared_ptr<Layer> model(nullptr);
	switch (layers[0])
	{
	case 2:
		model = std::dynamic_pointer_cast<LayerB>(models[0]);
		break;
	case 3:
		model = std::dynamic_pointer_cast<LayerC>(models[0]);
		break;
	case 4:
		model = std::dynamic_pointer_cast<LayerD>(models[0]);
		break;
	}
//	std::cout << data->getX().sizes() << std::endl;
	torch::Tensor res = model->forward(data->getX(), data->getZ());
//	std::cout << res.sizes() << std::endl;
	for (int i = 1; i < layers.size(); i++)
	{
		std::shared_ptr<Layer> model;
		switch (layers[i])
		{
		case 2:
			model=std::dynamic_pointer_cast<LayerB>(models[i]);
			break;
		case 3:
			model = std::dynamic_pointer_cast<LayerC>(models[i]);
			break;
		case 4:
			model = std::dynamic_pointer_cast<LayerD>(models[i]);
			break;
		}
		res = res.sigmoid();
		res = model->forward(res);
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
}

torch::Tensor FNN::penalty(int64_t nind)
{
	if (layers.size()==1)
	{
		std::shared_ptr<Layer> model;
		switch (layers[0])
		{
		case 2:
			model = std::dynamic_pointer_cast<LayerB>(models[0]);
			break;
		case 3:
			model = std::dynamic_pointer_cast<LayerC>(models[0]);
			break;
		case 4:
			model = std::dynamic_pointer_cast<LayerD>(models[0]);
			break;
		}
		return model->pen(lamb, lamb);
	}
	torch::Tensor penalty = torch::zeros(1);
	int i = 0;
	for (; i < layers.size()-1; i++)
	{

		std::shared_ptr<Layer> model;
		switch (layers[i])
		{
		case 2:
			model = std::dynamic_pointer_cast<LayerB>(models[i]);
			break;
		case 3:
			model = std::dynamic_pointer_cast<LayerC>(models[i]);
			break;
		case 4:
			model = std::dynamic_pointer_cast<LayerD>(models[i]);
			break;
		}
		penalty += model->pen(1e-9, 1e-3);
	}
	std::shared_ptr<Layer> model;
	switch (layers[i])
	{
	case 2:
		model = std::dynamic_pointer_cast<LayerB>(models[i]);
		break;
	case 3:
		model = std::dynamic_pointer_cast<LayerC>(models[i]);
		break;
	case 4:
		model = std::dynamic_pointer_cast<LayerD>(models[i]);
		break;
	}
	penalty += model->pen(1e0, lamb);
	return penalty / (double)nind;

}
/*
torch::Tensor FNN::training(std::shared_ptr<TensorData> dataset, int64_t epoches)
{
	fit_init(dataset);
	this->train();
	torch::optim::Adam optimizer(this->parameters());
	//auto Loss = torch::nn::MSELoss();
	int64_t epoch=0;
	torch::Tensor risk;
	torch::Tensor loss;
	auto options = torch::TensorOptions().dtype(torch::kFloat64);
	double risk_min = (double)INFINITY;
	int64_t k = 0;
	std::stringstream stream;
	while (epoch < epoches)
	{
		optimizer.zero_grad();
		torch::Tensor prediction =forward(dataset);
	//	std::cout << prediction.index({ torch::indexing::Slice(1,10) }) << std::endl;
		loss = torch::mse_loss(prediction, dataset->getY());
		torch::Tensor pen = penalty(dataset->nind);
		risk = loss / dataset->std_y.pow(2) + pen;
		if (risk.item<double>() < risk_min)
		{
			risk_min = risk.item<double>();
			k = 0;
			torch::save(this, stream);
			for (const auto& p : this->parameters()) {
				std::cout << p << std::endl;
			}
		}
		else
		{
			k++;
			if (k==100)
			{
				break;
			}
		}
	//	std::cout << "epoch: " << epoch << "\tMSE: " << loss << "\trisk: " << risk << std::endl;
		//if (epoch%100 ==0)
		//{
		//	std::cout << "epoch: " << epoch << "\tMSE: " << loss<<"\trisk: "<<risk << std::endl;
		//}
		risk.backward();
		optimizer.step();
		epoch++;
	}
	fit_end();
	for (const auto& p : this->parameters()) {
		std::cout << p << std::endl;
	}
	//torch::load(*this, stream);
	for (const auto& p : this->parameters()) {
		std::cout << p << std::endl;
	}
	return loss;
}
*/
void FNN::realization(std::shared_ptr<TensorData> data)
{
	int i = 0;
	for (; i < layers.size(); i++)
	{
		Eigen::VectorXd pos;
		Eigen::VectorXd loc;
		if (models[i]->bs0!=nullptr)
		{
			if (i == 0)
			{
				if (data->pos0 == -9)
				{
					data->pos0 =(double) data->getPos().minCoeff() - 1 / models[i]->bs0->n_basis / 2;
				}
				if (data->pos1 == -9)
				{
					data->pos1 = (double)data->getPos().maxCoeff() + 1 / models[i]->bs1->n_basis / 2;
				}
				models[i]->bs0->length = data->pos1 - data->pos0 + 1;
				pos = (data->getPos().array() - data->pos0).array() / (models[i]->bs0->length - 1);
			}
			else
			{
				std::vector<double> pos_v;
				double value = 1 / (double)models[i]->bs0->n_basis / 2;
				while (value<1)
				{
					pos_v.push_back(value);
					value += 1 / (double)models[i]->bs0->n_basis;
				}
				pos = Eigen::Map<Eigen::VectorXd>(pos_v.data(), pos_v.size());
				models[i]->bs0->length = pos.size();
			}
			models[i]->bs0->evaluate(pos);
	//		std::cout << "model " << i << " bs0 mat:\n" << models[i]->bs0->mat << std::endl;
		}
		if (i+1< layers.size())
		{
			std::vector<double> loc_v;
	
			double value = 1 / (double)models[i]->bs1->n_basis / 2;
			while (value<1)
			{
				loc_v.push_back(value);
				value += 1 / (double)models[i]->bs1->n_basis;
			}
			models[i]->bs1->length = loc_v.size();
			loc = Eigen::Map<Eigen::VectorXd>(loc_v.data(), loc_v.size());
			models[i]->bs1->evaluate(loc);
	//		std::cout << "model " << i << " bs1 mat:\n" << models[i]->bs1->mat << std::endl;
		}
	}
	i--;
	if (models[i]->bs1!=nullptr)
	{
		if (data->loc0 == -9)
		{
			data->loc0 = data->getLoc().minCoeff() - 1 / (double)models[i]->bs1->n_basis / 2;
		}
		if (data->loc1 == -9)
		{
			data->loc1 = data->getLoc().maxCoeff() + 1 / (double)models[i]->bs1->n_basis / 2;
		}
	
		Eigen::VectorXd loc = (data->getLoc().array() - data->loc0).array() / (data->loc1 - data->loc0);
		models[i]->bs1->evaluate(loc);
	//	models[i]->index.push_back(loc.size()-1);
//		std::cout << "model " << i << " bs1 mat:\n" << models[i]->bs1->mat << std::endl;
	}

	
}
/*
void FNN::fit_init(std::shared_ptr<TensorData> data)
{
	realization(data);
	realize = true;
}

void FNN::fit_end()
{
	realize = false;
}
*/