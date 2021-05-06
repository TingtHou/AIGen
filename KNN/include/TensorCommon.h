#pragma once
#include <torch/torch.h>
#include <iostream>
#include "CommonFunc.h"
#include "TensorData.h"
#include <ATen/Parallel.h>
#include <ATen/ATen.h>

template<class Optim, class Net, class Loss>
torch::Tensor training(std::shared_ptr<Net> net, std::shared_ptr<TensorData> train, std::shared_ptr<TensorData> valid, double lr, int64_t epoches = 1e6)
{
	//net->fit_init(dataset);
	net->train();
	Optim optimizer(net->parameters(), lr);
	std::stringstream ss;
	int64_t epoch = 0;
	double risk_min = INFINITY;
	int64_t k = 0;
	auto f_loss = std::make_shared<Loss>();
	while (epoch < epoches)
	{
		for (size_t i = 0; i < train->Batch_Num; i++)
		{
			optimizer.zero_grad();
			auto miniBatch = train->getBatch(i);
			torch::Tensor prediction = net->forward(miniBatch);
		//	std::cout << miniBatch->getY().index({ torch::indexing::Slice(torch::indexing::None, 10)}) << std::endl;
		//	std::cout << prediction.index({ torch::indexing::Slice(torch::indexing::None, 10)}) << std::endl;
			//std::cout << prediction.sizes()[0]<<  "\t"<< prediction.sizes()[1] << std::endl;
			torch::Tensor pen = net->penalty(miniBatch->nind);
			torch::Tensor loss = (*f_loss)(prediction, miniBatch->getY());
			torch::Tensor risk = loss / miniBatch->std_y.pow(2) + pen;
			risk.backward();
			optimizer.step();
	//		std::cout << "===================================\nepoch: " << epoch << "\t loss: " << loss.item<double>() << "\tpen: "<< pen <<"\trisk: "<< risk<< std::endl;
		}


		if (epoch % 10 == 0)
		{
			net->eval();
			torch::Tensor pred_test = net->forward(train);
			torch::Tensor loss_test = (*f_loss)(pred_test, train->getY());
			torch::Tensor pen_test = net->penalty(train->nind);
			torch::Tensor risk_loss = loss_test / train->std_y.pow(2) + pen_test;
			net->train();
			if (risk_loss.item<double>() < risk_min)
			{
				ss.str(std::string());
				risk_min = risk_loss.item<double>();
				net->epoch = epoch;
				net->loss = loss_test;
				torch::serialize::OutputArchive output_archive;
				net->save(output_archive);
				output_archive.save_to(ss);
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
			if (epoch % 1000 == 0)
			{
				std::cout << "===================================\nepoch: " << epoch << "\nTraining loss: " << loss_test.item<double>() << std::endl;
			}
			
		}
		
		
		epoch++;
	}

	torch::serialize::InputArchive input_archive;
	input_archive.load_from(ss);
	net->load(input_archive);
	torch::Tensor loss_vaild = torch::full(1, -9);
	if (valid->nind)
	{
		net->eval();
		torch::Tensor pred_vaild = net->forward(valid);
		loss_vaild = (*f_loss)(pred_vaild, valid->getY());
		net->train();
	}
	
//	for (const auto& p : net->parameters()) {
//	std::cout << p << std::endl;
//	}
	return loss_vaild;
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