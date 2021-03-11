#include "../include/TensorData.h"
#include <iostream>
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
TensorData::TensorData(PhenoData &phe, GenoData &gene, CovData &cov)
{
	this->phe = std::make_shared< PhenoData>(phe);
	this->gene = std::make_shared<GenoData>(gene);
	this->cov = std::make_shared<CovData>(cov);
	x = dtt::eigen2libtorch(gene.Geno);
	x=x.to(torch::kF64);
	z = dtt::eigen2libtorch(cov.Covariates);
	z = z.to(torch::kF64);
	pos = gene.pos.cast<double>();
	double  delta_pos = (pos.maxCoeff() - pos.minCoeff()) / 100;
	pos0 = pos.minCoeff() - delta_pos;
	pos1 = pos.maxCoeff() + delta_pos;
	isBalanced = phe.isBalance;
	if (phe.isBalance)
	{
		y = dtt::eigen2libtorch(phe.Phenotype);
		y = y.to(torch::kF64);
		if (phe.loc.size() > 0)
		{
			loc = phe.loc.cast<double>();
			double  delta_loc = (loc.maxCoeff() - loc.minCoeff()) / 100;
			loc0 = loc.minCoeff() - delta_loc;
			loc1 = loc.maxCoeff() + delta_loc;
		}
	
	}
	mean_y = torch::full(1, phe.mean);
	std_y = torch::full(1, phe.std);
	nind = phe.fid_iid.size();
	
}

TensorData::TensorData(torch::Tensor phe, torch::Tensor gene, torch::Tensor cov, Eigen::VectorXd pos, Eigen::VectorXd loc)
{
	this->y = phe;
	this->x = gene;
	this->z = cov;
	this->pos = pos;
	this->loc = loc;
	double  delta_pos = (pos.maxCoeff() - pos.minCoeff()) / 100;
	pos0 = pos.minCoeff() - delta_pos;
	pos1 = pos.maxCoeff() + delta_pos;
	double  delta_loc = (loc.maxCoeff() - loc.minCoeff()) / 100;
	loc0 = loc.minCoeff() - delta_loc;
	loc1 = loc.maxCoeff() + delta_loc;
	nind = phe.sizes()[0];
}

void TensorData::setPos(Eigen::VectorXd pos)
{
	this->pos = pos;
}

void TensorData::setLoc(Eigen::VectorXd loc)
{
	this->loc = loc;
}


std::shared_ptr<TensorData> TensorData::getSample(int64_t index)
{
	Eigen::MatrixXf y_tmp(1, phe->vPhenotype[index].size());
	y_tmp.row(0) = phe->vPhenotype[index];
	torch::Tensor yi = dtt::eigen2libtorch(y_tmp);
	torch::Tensor xi = x.index({ index, });
	xi = xi.reshape({ 1, x.sizes()[1] });
	torch::Tensor zi = z.index({ index, });
	zi = zi.reshape({ 1, z.sizes()[1] });
	auto Sample_i = std::make_shared<TensorData>(yi, xi, zi, pos, phe->vloc[index].cast<double>());
	Sample_i->setMean_STD(mean_y, std_y);
	return Sample_i;
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

void TensorData::setMean_STD(torch::Tensor mean, torch::Tensor std)
{
	this->mean_y = mean;
	this->std_y = std;
}
