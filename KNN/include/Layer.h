#pragma once
#include <torch/torch.h>
#include <iostream>
#include "Basis.h"

struct LayerA : torch::nn::Module
{
	LayerA(int64_t  in_dim, int64_t  out_dim, int64_t  z_dim = 0)
	{
		torch::manual_seed(629);
		fc = register_module("fc", torch::nn::Linear(in_dim + z_dim, out_dim));
	}

	/*
	torch::Tensor forward(torch::Tensor x)
	{
		return fc(x);
	}
	*/


	torch::Tensor forward(torch::Tensor x, torch::Tensor z= torch::empty(0))
	{
		bool isCov = z.sizes()[0] != 0 && z.sizes()[1] != 0;
		bool isGene = x.sizes()[0] != 0 && x.sizes()[1] != 0;
		if (isCov)
		{
			if (!isGene)
			{
				x = z;
			}
			else
			{
				x = torch::cat({ x,z }, 1);
			}
			
		}
	
		return fc(x);
	}

	torch::Tensor pen()
	{
		torch::Tensor penalty = torch::zeros(1);
		for (const auto& pair : named_parameters())
		{
			if (pair.value().requires_grad() && pair.key().find("fc") != std::string::npos)
			{
			//	std::cout << pair.key() << ": " << pair.value() << std::endl << " Power 2: " << pair.value().pow(2);
				penalty += pair.value().pow(2).sum();
			}
		
		}
		return penalty;
	}

	torch::nn::Linear fc{ nullptr };
};


struct Layer : torch::nn::Module
{
	std::shared_ptr<Basis> bs0;
	std::shared_ptr<Basis> bs1;
	virtual torch::Tensor forward(torch::Tensor x , torch::Tensor cov= torch::empty(0))=0;
	virtual torch::Tensor pen(double lamb0 = 1, double lamb1 = 1) = 0;
	bool singleknot = false;  //each phenotype is interploted at  different single knot, or the model will be each by each individual. 

};


struct LayerB : Layer
{
	
	LayerB(std::shared_ptr<Basis> bs0, std::shared_ptr<Basis> bs1, int64_t n_covs = 0, bool bias = true)
	{
		torch::manual_seed(629);
		this->bs0 = bs0;
		this->bs1 = bs1;
		fc0 = register_module("fc0", torch::nn::Linear(torch::nn::LinearOptions(bs0->n_basis , bs1->n_basis).bias(bias)));
		if (n_covs!=0)
		{
			fc1 = register_module("covs", torch::nn::Linear(torch::nn::LinearOptions(n_covs, bs1->n_basis).bias(false)));
		}
	
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor cov = torch::empty(0))
	{
		x = (x.matmul(bs0->mat)) / bs0->length;

		x = fc0->forward(x);
		if (cov.sizes()[1] != 0 && cov.sizes()[0] != 0)
		{
			//x = torch::cat({ x,cov }, 1);
			cov = fc1->forward(cov);
			x = x + cov;
		}
		if (!singleknot)
		{
			return x.matmul(bs1->mat.t());
		}
		else
		{
		//	std::cout << bs1->mat.sizes() << "\t" << x.sizes() << std::endl;
			if (x.sizes()[0]!=0)
			{
				x = x * bs1->mat;
				
			}
	//		std::cout << x.sum(1, true).index({ torch::indexing::Slice(torch::indexing::None, 10),torch::indexing::Slice(torch::indexing::None, torch::indexing::None) }) << std::endl;
			return x.sum(1, true);
		}

	}

	torch::Tensor pen(double lamb0 = 1, double lamb1 = 1)
	{
		torch::Tensor penalty = torch::zeros(1);
		std::string weight_str = ".weight", bias_str=".bias";

		for (const auto& pair : named_parameters())
		{
			auto param = pair.value();
			auto name = pair.key();
			if (param.requires_grad() && name.find("fc") != std::string::npos)
			{
				if (name.compare(name.length()- weight_str.length(),weight_str.length(),weight_str)==0)
				{
			//		std::cout << param << std::endl;
					penalty += bs1->pen_2d(param, *bs0,  lamb0,  lamb1);
				}
				if (name.compare(name.length() - bias_str.length(), bias_str.length(), bias_str) == 0)
				{
					penalty += bs1->pen_1d(param, lamb1);
				}
			}

		}
		return penalty;
	}

	torch::nn::Linear fc0{ nullptr };
	torch::nn::Linear fc1{ nullptr };
};


struct LayerC : Layer
{
	
	LayerC(std::shared_ptr<Basis> bs0, int64_t n_covs = 0)
	{
		torch::manual_seed(629);
		this->bs0 = bs0;
		fc0 = register_module("fc0", torch::nn::Linear(bs0->n_basis + bs0->linear, 1));
		if (n_covs != 0)
		{
			fc1 = register_module("covs", torch::nn::Linear(torch::nn::LinearOptions(n_covs, 1).bias(false)));
		}
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor cov = torch::empty(0))
	{
		x = x.matmul(bs0->mat) / bs0->length;
		x = fc0->forward(x);
		if (cov.sizes()[1] != 0 && cov.sizes()[0] != 0)
		{
			//x = torch::cat({ x,cov }, 1);
			cov = fc1->forward(cov);
			x = x + cov;
		}
	//	std::cout << x.index({ torch::indexing::Slice(torch::indexing::None, 10),torch::indexing::Slice(torch::indexing::None, torch::indexing::None) }) << std::endl;
		return x;
	}



	torch::Tensor pen(double lamb0 = 1, double lamb1 = 1)
	{
		torch::Tensor penalty = torch::zeros(1);
		std::string weight_str = ".weight", bias_str = ".bias";

		for (const auto& pair : named_parameters())
		{
			auto param = pair.value();
			auto name = pair.key();
			if (param.requires_grad() && name.find("fc") != std::string::npos)
			{
				if (name.compare(name.length() - weight_str.length(), weight_str.length(), weight_str) == 0)
				{
					param = param.reshape({ -1, });
					penalty += bs0->pen_1d(param, lamb1);
				}
				if (name.compare(name.length() - bias_str.length(), bias_str.length(), bias_str) == 0)
				{
					penalty += param.index({ 0 }).pow(2);
				}
			}

		}

		return penalty;
	}

	torch::nn::Linear fc0{ nullptr };
	torch::nn::Linear fc1{ nullptr };
};



struct LayerD : Layer
{
	LayerD(std::shared_ptr<Basis> bs1)
	{
		this->bs1 = bs1;
		fc0 = register_module("fc0", torch::nn::Linear(1, bs1->n_basis));
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor cov = torch::empty(0))
	{
		x = fc0->bias;
		if (!singleknot)
		{
			return x.matmul(bs1->mat.t());
		}
		else
		{
			x = x * bs1->mat;
			return x.sum(1, true);
		}
	//	return x.matmul(bs1->mat.t());
		////
		// add index later
	}


	torch::Tensor pen(double lamb0 = 1, double lamb1 = 1)
	{
		torch::Tensor penalty = torch::zeros(1);
		std::string weight_str = ".weight", bias_str = ".bias";

		for (const auto& pair : named_parameters())
		{
			auto param = pair.value();
			auto name = pair.key();
			if (param.requires_grad() && name.find("fc") != std::string::npos)
			{
				if (name.compare(name.length() - bias_str.length(), bias_str.length(), bias_str) == 0)
				{
					penalty += bs1->pen_1d(param, lamb1);
				}
			}

		}

		return penalty;
	}

	torch::nn::Linear fc0{ nullptr };
};
