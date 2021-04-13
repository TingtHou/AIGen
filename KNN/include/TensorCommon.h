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
		if (train->isBalanced)
		{
			for (size_t i = 0; i < train->Batch_Num; i++)
			{
				optimizer.zero_grad();
				auto miniBatch = train->getBatch(i);
				torch::Tensor prediction = net->forward(miniBatch);
				torch::Tensor pen = net->penalty(miniBatch->nind);
				torch::Tensor loss = (*f_loss)(prediction, miniBatch->getY());
				torch::Tensor risk = loss / miniBatch->std_y.pow(2) + pen;
				risk.backward();
				optimizer.step();
			}
		
		}
		else
		{
			optimizer.zero_grad();
			torch::Tensor loss = torch::zeros(1);
			torch::Tensor pen = torch::zeros(1);
			for (int64_t i = 0; i < train->nind; i++)
			{
				
				auto sample = train->getSample(i);
				torch::Tensor prediction = net->forward(sample);
				pen += net->penalty(sample->nind);
				loss += (*f_loss)(prediction, sample->getY());
	//			std::cout << sample->getY().sizes() << std::endl;
			}
			loss /= (double)train->nind;
			pen /= (double)train->nind;
			torch::Tensor risk = loss / train->std_y.pow(2) + pen;
			risk.backward();
			optimizer.step();
		
		}
		
	
		if (epoch % 10 == 0)
		{
			net->eval();
			torch::Tensor loss_test = torch::zeros(1);
			torch::Tensor pen_test = torch::zeros(1);
			if (valid->nind != 0)
			{
				
				if (valid->isBalanced)
				{
					torch::Tensor pred_test = net->forward(valid);
					loss_test += (*f_loss)(pred_test, valid->getY());
					pen_test += net->penalty(valid->nind);
				}
				else
				{
					for (size_t i = 0; i < valid->nind; i++)
					{
	//					std::cout << "ind" << i << std::endl;
						auto sample = valid->getSample(i);
						torch::Tensor pred_test = net->forward(sample);
						pen_test += net->penalty(sample->nind);
						loss_test += (*f_loss)(pred_test, sample->getY());
					}
					loss_test /= (double)valid->nind;
					pen_test /= (double)valid->nind;
				
				}
			}
			torch::Tensor risk_loss = loss_test / valid->std_y.pow(2) + pen_test;
			
			if (risk_loss.item<double>() < risk_min)
			{
				ss.str(std::string());
				risk_min = risk_loss.item<double>();
				torch::serialize::OutputArchive output_archive;
				net->save(output_archive);
				output_archive.save_to(ss);
				net->epoch = epoch;
				net->loss = loss_test;
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
			if (epoch % 100 == 0)
			{
				std::cout << "===================================\nepoch: " << epoch << "\nTraining loss: " << loss_test.item<double>() << std::endl;
			}
	//		std::cout << "===================================\nepoch: " << epoch << "\nTraning loss: " << loss_test.item<double>() << std::endl;
			net->train();
		}
	//	if (epoch==1)
	//	{
	//		net->realize = true;
	//	}
		epoch++;
	}

	torch::serialize::InputArchive input_archive;
	input_archive.load_from(ss);
	net->load(input_archive);
	/*
	torch::Tensor loss_test = torch::zeros(1);
	if (test->nind!=0)
	{
		net->eval();
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
	}
	*/
	//for (const auto& p : net->parameters()) {
	//	std::cout << p << std::endl;
	//}
	return 	net->loss;
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