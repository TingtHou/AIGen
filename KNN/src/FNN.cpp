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
//	std::cout << "y "<<data->getY().sizes()[0]<<"*"<< data->getY().sizes()[1]  << "\n"<< data->getY().index({ torch::indexing::Slice(torch::indexing::None, 10) }) << std::endl;
//	std::cout << "x " << data->getX().sizes()[0] << "*" << data->getX().sizes()[1] << "\n" << data->getX().index({ torch::indexing::Slice(torch::indexing::None, 10), torch::indexing::Slice(torch::indexing::None, 10) }) << std::endl;
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
	//std::cout << data->getX().sizes() << std::endl;
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
				loc = (loc.array() - data->loc0).array() / (data->loc1 - data->loc0);
				models[i]->bs1->evaluate(loc);
				
			//	std::cout << xi << std::endl;
				torch::Tensor tmp= model->forward(xi);
		//		std::cout << tmp << std::endl;
				Final=torch::cat({ Final,tmp }, 0);

				start_p = expand_size;
			}
		}
		else
		{
			Final = model->forward(res);
//			std::cout << Final.sizes() << std::endl;
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
		//	std::cout << "model " << i << " bs0 mat:\n" << models[i]->bs0->mat.sizes()[0]<<"x"<< models[i]->bs0->mat.sizes()[1] << std::endl;
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
	//		std::cout << "model " << i << " bs1 mat:\n" << models[i]->bs1->mat.sizes()[0] << "x" << models[i]->bs1->mat.sizes()[1] << std::endl;
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
