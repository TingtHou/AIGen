#pragma once
#include <torch/torch.h>
#include <iostream>
#include "CommonFunc.h"
#include "TensorData.h"


template<class Optim, class Net, class Loss>
torch::Tensor training(std::shared_ptr<Net> net, std::shared_ptr<TensorData> train, std::shared_ptr<TensorData> test, int64_t epoches = 1e6)
{
	//net->fit_init(dataset);
	net->train();
	Optim optimizer(net->parameters());
	std::stringstream ss;
	int64_t epoch = 0;
	double risk_min = INFINITY;
	int64_t k = 0;
	auto f_loss = std::make_shared<Loss>();
//	f_loss->operator()
	//auto f = torch::nn::MSELoss();
	while (epoch < epoches)
	{
		optimizer.zero_grad();
		torch::Tensor loss = torch::zeros(1);
		torch::Tensor pen = torch::zeros(1);
		if (train->isBalanced)
		{
			torch::Tensor prediction = net->forward(train);
			pen += net->penalty(train->nind);
			//torch::Tensor loss = torch::mse_loss(prediction, dataset->getY());
			loss += (*f_loss)(prediction, train->getY());
		}
		else
		{
			for (size_t i = 0; i < train->nind; i++)
			{
				auto sample = train->getSample(i);
				torch::Tensor prediction = net->forward(sample);
	//			std::cout << prediction << std::endl;
		//		std::cout << sample->getY()<< std::endl;
				pen += net->penalty(sample->nind);
				//torch::Tensor loss = torch::mse_loss(prediction, dataset->getY());
				loss += (*f_loss)(prediction, sample->getY());
			}
			loss /= (double)train->nind;
		}
		
		torch::Tensor risk = loss / train->std_y.pow(2) + pen;
//		std::cout << risk<<"\t"<<loss<<"\t" << pen << std::endl;
		if (epoch %100 ==0)
		{
			net->eval();
			torch::Tensor loss_test = torch::zeros(1);
			if (test->isBalanced)
			{
				torch::Tensor pred_test = net->forward(test);
				loss_test = (*f_loss)(pred_test, test->getY());
			}
			else 
			{
				for (size_t i = 0; i < test->nind; i++)
				{
					auto sample = test->getSample(i);
					torch::Tensor pred_test = net->forward(sample);
					loss_test += (*f_loss)(pred_test, sample->getY());
				}
				loss_test /= (double)test->nind;
			}
			
			std::cout << "===================================\nepoch: " <<epoch << "\nTraning loss: " << loss.item<double>()<<"\nTesting loss: "<< loss_test.item<double>() << std::endl;
			net->train();
		}
		if (epoch % 10 == 0)
		{
			
			if (risk.item<double>() < risk_min)
			{
				ss.str(std::string());
				risk_min = risk.item<double>();
				torch::serialize::OutputArchive output_archive;
				net->save(output_archive);
				output_archive.save_to(ss);
				net->epoch = epoch;
				net->loss = loss;
				k = 0;
			}
			else
			{
				k++;
				if (k == 100)
				{
					break;
				}
			}
		}
		risk.backward();
		optimizer.step();
		epoch++;
	}
	//net->fit_end();
//	for (const auto& p : net->parameters()) {
//			std::cout << p << std::endl;
//	}
	torch::serialize::InputArchive input_archive;
	input_archive.load_from(ss);
	net->load(input_archive);
	net->eval();
	torch::Tensor loss_test = torch::zeros(1);
	if (test->isBalanced)
	{
		torch::Tensor pred_test = net->forward(test);
		loss_test = (*f_loss)(pred_test, test->getY());
	}
	else
	{
		for (size_t i = 0; i < test->nind; i++)
		{
			auto sample = test->getSample(i);
			torch::Tensor pred_test = net->forward(sample);
			loss_test += (*f_loss)(pred_test, sample->getY());
		}
		loss_test /= (double)test->nind;
	}
//	for (const auto& p : net->parameters()) {
//		std::cout << p << std::endl;
//	}
	return loss_test;
}

template<class Net, class Loss>
torch::Tensor Testing(std::shared_ptr<Net> net, std::shared_ptr<TensorData> dataset)
{
	net->eval();
	torch::Tensor prediction = net->forward(dataset);
	auto f_loss = std::make_shared<Loss>();
	torch::Tensor loss = (*f_loss)(prediction, dataset->getY());
	return loss;
}