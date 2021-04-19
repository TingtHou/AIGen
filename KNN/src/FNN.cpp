#include "../include/FNN.h"


FNN::FNN(std::vector<int64_t> dims, double lamb)
{
	this->lamb = lamb;
	this->dims = dims;

}

torch::Tensor FNN::forward(std::shared_ptr<TensorData> data)
{
	/*
	if(!realize)
		realization(data);
	
	else
	{
		if (!data->isBalanced)
		{
			if (models[layers.size() - 1]->bs1 != nullptr)
			{
			
					models[layers.size() - 1]->bs1->mat = LastLayer[subtrain+ID];
			}
		}
	}
	*/
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
	int i = 1;
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
		res = res.sigmoid();
		res = model->forward(res);
	}
	torch::Tensor Final;
	//std::shared_ptr<Layer> model_final(nullptr);
	res = res.sigmoid();
	switch (layers[i])
	{
	case 2:
		Final = torch::zeros({ 0,1 });
		model = std::dynamic_pointer_cast<LayerB>(models[i]);
		if (i == (layers.size() - 1) && !data->isBalanced)
		{
			int64_t start_p = 0;
			int64_t expand_size=0;
			Eigen::VectorXd total_loc = data->getLoc();
			for (int64_t index = 1; index <= data->nind; index++)
			{
				expand_size = std::accumulate(data->nknots.begin(), data->nknots.begin() +index, 0);
				torch::Tensor xi = res.index({ torch::indexing::Slice(start_p, expand_size ), });

				Eigen::VectorXd loc(expand_size - start_p);

				loc << total_loc.block(start_p, 0, expand_size - start_p, 1);
				models[i]->bs1->evaluate(loc);
				
		
				torch::Tensor tmp= model->forward(xi);

				Final=torch::cat({ Final,tmp }, 0);

				start_p = expand_size;
			}
		}
		else
		{
			Final = model->forward(res);
		}
		break;
	case 3:
		model = std::dynamic_pointer_cast<LayerC>(models[i]);
		Final = model->forward(res);
	//	Final = model->forward(res);
	//	std::cout << Final << std::endl;
		break;
	case 4:
		model = std::dynamic_pointer_cast<LayerD>(models[i]);
		Final = model->forward(res);
		break;
	}

	switch (data->dataType)
	{
	case 0:
		Final = Final * data->std_y + data->mean_y;
		break;
	case 1:
		Final = Final.sigmoid();
		break;
	case 2:
		break;
	}
	
	return Final;
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
		}
		if (i+1< layers.size())
		{
		//	models[i]->singleknot = false;
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
	//	models[i]->singleknot = true;
		if (data->loc0 == -9)
		{
			data->loc0 = data->getLoc().minCoeff() - 1 / (double)models[i]->bs1->n_basis / 2;
		}
		if (data->loc1 == -9)
		{
			data->loc1 = data->getLoc().maxCoeff() + 1 / (double)models[i]->bs1->n_basis / 2;
		}
		if (data->isBalanced)
		{
			Eigen::VectorXd loc = (data->getLoc().array() - data->loc0).array() / (data->loc1 - data->loc0);
			models[i]->bs1->evaluate(loc);
		}
		
	
		/*
		if (!data->isBalanced)
		{
			LastLayer.push_back(models[i]->bs1->mat);
		}
		*/
	
	//	models[i]->index.push_back(loc.size()-1);
	//	std::cout << "model " << i << " bs1 mat:\n" << models[i]->bs1->mat.index({ torch::indexing::Slice(torch::indexing::None, 10) }) << std::endl;
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