#include "../include/TensorData.h"
#include <iostream>
TensorData::TensorData(PhenoData& phe, GenoData& gene, CovData& cov)
{
	this->phe = std::make_shared<PhenoData>(phe);
	this->gene = std::make_shared<GenoData>(gene);;
	this->cov = std::make_shared<CovData>(cov);;
	x = dtt::eigen2libtorch(gene.Geno);
	z = dtt::eigen2libtorch(cov.Covariates);
	pos = gene.pos.cast<double>();
	if (pos.size()>0)
	{
		double  delta_pos = (pos.maxCoeff() - pos.minCoeff()) / 100;
		pos0 = pos.minCoeff() - delta_pos;
		pos1 = pos.maxCoeff() + delta_pos;
	}
	
	isBalanced = phe.isBalance;
	if (phe.isBalance)
	{
		y = dtt::eigen2libtorch(phe.Phenotype);
	}
	if (phe.loc.size() > 0)
	{
		loc = phe.loc.cast<double>();
		double  delta_loc = (loc.maxCoeff() - loc.minCoeff()) / 100;
		loc0 = loc.minCoeff() - delta_loc;
		loc1 = loc.maxCoeff() + delta_loc;
	}
	mean_y = torch::full(1, phe.mean);
	std_y = torch::full(1, phe.std);
	nind = phe.nind;
//	std::cout << "mean: " << mean_y << "\n" << "std: " << std_y << std::endl;
}
/*
TensorData::TensorData(PhenoData &phe, GenoData &gene, CovData &cov)
{
	y = dtt::eigen2libtorch(phe.Phenotype);
	x = dtt::eigen2libtorch(gene.Geno);
	z = dtt::eigen2libtorch(cov.Covariates);
	pos = gene->pos.cast<float>();
	loc = phe->loc.cast<float>();
	mean_y = y.mean();
	std_y = y.std();
	nind = y.sizes()[0];
	
}
*/
TensorData::TensorData(std::shared_ptr<PhenoData> phe, std::shared_ptr<GenoData> gene, std::shared_ptr<CovData> cov)
{
	this->phe = phe;
	this->gene = gene;
	this->cov = cov;
	x = dtt::eigen2libtorch(gene->Geno);
	z = dtt::eigen2libtorch(cov->Covariates);
	z = z.to(torch::kF64);
	pos = gene->pos.cast<double>();
	if (pos.size()>0)
	{
		double  delta_pos = (pos.maxCoeff() - pos.minCoeff()) / 100;
		pos0 = pos.minCoeff() - delta_pos;
		pos1 = pos.maxCoeff() + delta_pos;
	}
	isBalanced = phe->isBalance;
	if (phe->isBalance)
	{
		y = dtt::eigen2libtorch(phe->Phenotype);
	}
	if (phe->loc.size() > 0)
	{
		loc = phe->loc.cast<double>();
		double  delta_loc = (loc.maxCoeff() - loc.minCoeff()) / 100;
		loc0 = loc.minCoeff() - delta_loc;
		loc1 = loc.maxCoeff() + delta_loc;
	}
	mean_y = torch::full(1, phe->mean);
	std_y = torch::full(1, phe->std);
	nind = phe->fid_iid.size();
	//std::cout << "mean: " << mean_y << "\n" << "std: " << std_y << std::endl;
}

TensorData::TensorData(torch::Tensor phe, torch::Tensor gene, torch::Tensor cov, Eigen::VectorXd pos, Eigen::VectorXd loc, double loc0, double loc1)
{
	this->y = phe;
	this->x = gene;
	this->z = cov;
	this->pos = pos;
	this->loc = loc;
	if (pos.size()>0)
	{
		double  delta_pos = (pos.maxCoeff() - pos.minCoeff()) / 100;
		pos0 = pos.minCoeff() - delta_pos;
		pos1 = pos.maxCoeff() + delta_pos;
	} 
	this->loc0 = loc0;
	this->loc1 = loc1;
	nind = phe.sizes()[0];
}
/*
TensorData::TensorData(std::shared_ptr<PhenoData> phe, torch::Tensor gene, torch::Tensor cov, Eigen::VectorXd pos,  double loc0, double loc1)
{
	this->phe = phe;
	this->x = gene;
	this->z = cov;
	this->pos = pos;
	this->loc = loc;
	if (pos.size() > 0)
	{
		double  delta_pos = (pos.maxCoeff() - pos.minCoeff()) / 100;
		pos0 = pos.minCoeff() - delta_pos;
		pos1 = pos.maxCoeff() + delta_pos;
	}
	if (phe->loc.size() > 0)
	{
		loc = phe->loc.cast<double>();
	}
	this->loc0 = loc0;
	this->loc1 = loc1;
	mean_y = torch::full(1, phe->mean);
	std_y = torch::full(1, phe->std);
	nind = phe->nind;
}

*/
void TensorData::setPos(Eigen::VectorXd pos)
{
	this->pos = pos;
}

void TensorData::setLoc(Eigen::VectorXd loc)
{
	this->loc = loc;
}

void TensorData::setBatchNum(int64_t Batch_Num)
{
	this->Batch_Num = Batch_Num;
	int64_t Batch_size = nind / Batch_Num;
	for (size_t i = 0; i < Batch_Num; i++)
	{
		Batch_index.push_back(i* Batch_size);
	}
	Batch_index.push_back(nind);
}


std::shared_ptr<TensorData> TensorData::getSample(int64_t index)
{
	Eigen::MatrixXf y_tmp( phe->vPhenotype[index].size(),1);
	y_tmp.col(0) = phe->vPhenotype[index];
	torch::Tensor yi = dtt::eigen2libtorch(y_tmp);
	torch::Tensor xi = x.index({ index, });
//	xi = xi.reshape({ 1, x.sizes()[1] });
	//std::cout << xi << std::endl;
	xi = xi.repeat({ y_tmp.size(), 1 });
	//std::cout << xi << std::endl;
//	
	torch::Tensor zi = z.index({ index, });
//	std::cout << zi.sizes()[0] << "\t" << zi.sizes()[1] << std::endl;
	zi = zi.repeat({ y_tmp.size(), 1 });
//	std::cout << zi << std::endl;
//	zi = zi.reshape({ 1, z.sizes()[1] });
	auto Sample_i = std::make_shared<TensorData>(yi, xi, zi, pos, phe->vloc[index].cast<double>(),loc0,loc1);
	Sample_i->setMean_STD(mean_y, std_y);
	Sample_i->dataType = this->dataType;
	return Sample_i;
}

std::tuple<std::shared_ptr<TensorData>, std::shared_ptr<TensorData>> TensorData::GetsubTrain(double ratio)
{
	torch::Tensor x_subtrain = x.index({ torch::indexing::Slice(torch::indexing::None,nind * ratio -1), });
	torch::Tensor x_valid = x.index({ torch::indexing::Slice(nind * ratio , torch::indexing::None), });
	torch::Tensor z_subtrain = z.index({ torch::indexing::Slice(torch::indexing::None,nind * ratio -1), });
	torch::Tensor z_valid = z.index({ torch::indexing::Slice(nind * ratio , torch::indexing::None), });
	std::shared_ptr< TensorData> subTrain=nullptr;
	std::shared_ptr< TensorData> valid=nullptr;
	if (isBalanced)
	{
		torch::Tensor y_subtrain = y.index({ torch::indexing::Slice(torch::indexing::None,nind * ratio -1), });
		torch::Tensor y_valid = y.index({ torch::indexing::Slice(nind * ratio, torch::indexing::None), });
		Eigen::VectorXd loc_subtrain;
		Eigen::VectorXd loc_valid;
		if (y.sizes()[1] == 1 && loc.size() != 0)
		{
			loc_subtrain.resize(y_subtrain.sizes()[0]);
			loc_valid.resize(y_valid.sizes()[0]);
			loc_subtrain << loc.block(0, 0, loc_subtrain.size(), 1);
		}
		else
		{
			loc_subtrain = loc;
			loc_valid = loc;
		}
		 subTrain = std::make_shared<TensorData>(y_subtrain, x_subtrain, z_subtrain, pos, loc_subtrain, loc0, loc1);
		 valid = std::make_shared<TensorData>(y_valid, x_valid, z_valid, pos, loc_valid, loc0, loc1);
		 subTrain->setMean_STD(y_subtrain.mean(), y_subtrain.std());
		 subTrain->dataType = this->dataType;
		 valid->setMean_STD(y_valid.mean(), y_valid.std());
		 valid->dataType = this->dataType;
	}
	else
	{
		PhenoData newP_subTrain{*phe};
		PhenoData newP_valid{ *phe };
		newP_subTrain.vPhenotype.clear();
		newP_subTrain.vloc.clear();
		newP_valid.vloc.clear();
		newP_subTrain.nind = ratio * phe->nind;
		newP_valid.vPhenotype.clear();
		newP_valid.nind = phe->nind - ratio * phe->nind;
		newP_subTrain.vPhenotype.insert(newP_subTrain.vPhenotype.end(), phe->vPhenotype.begin(), phe->vPhenotype.begin() + ratio * phe->nind - 1);
		newP_subTrain.vloc.insert(newP_subTrain.vloc.end(), phe->vloc.begin(), phe->vloc.begin() + ratio * phe->nind - 1);
		newP_valid.vPhenotype.insert(newP_valid.vPhenotype.end(), phe->vPhenotype.begin() + ratio * phe->nind, phe->vPhenotype.end());
		newP_valid.vloc.insert(newP_valid.vloc.end(), phe->vloc.begin() + ratio * phe->nind, phe->vloc.end());
	}


	
	

	return std::make_tuple(subTrain, valid);
}

torch::Tensor TensorData::getY()
{
	return y;
}

torch::Tensor TensorData::getX()
{
	return x;
}

torch::Tensor TensorData::getZ()
{
	return z;
	// TODO: insert return statement here
}

std::shared_ptr<TensorData> TensorData::getBatch(int64_t index_batch)
{
	torch::Tensor yi = y.index({ torch::indexing::Slice(Batch_index[index_batch], Batch_index[index_batch+1] - 1), });
	torch::Tensor xi = x.index({ torch::indexing::Slice(Batch_index[index_batch], Batch_index[index_batch+1] - 1), });
	torch::Tensor zi = z.index({ torch::indexing::Slice(Batch_index[index_batch], Batch_index[index_batch+1] - 1), });
	Eigen::VectorXd loci;
	if (yi.sizes()[1]==1 && loc.size()!=0)
	{
		loci.resize(Batch_index[index_batch + 1] -1 - Batch_index[index_batch]);
		loci << loc.block(Batch_index[index_batch], 0, loci.size(), 1);
	}
	else
	{
		loci = loc;
	}
	auto Sample_i = std::make_shared<TensorData>(yi, xi, zi, pos, loci, loc0, loc1);
	Sample_i->setMean_STD(yi.mean(), yi.std());
	Sample_i->dataType = this->dataType;
	return Sample_i;
}

void TensorData::setMean_STD(torch::Tensor mean, torch::Tensor std)
{
	this->mean_y = mean;
	this->std_y = std;
}
