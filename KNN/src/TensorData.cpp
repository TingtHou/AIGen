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
		if (phe.loc.size() > 0)
		{
			loc = phe.loc.cast<double>();
		}
		nknots.assign(phe.nind, 1);
	}
	else
	{
		std::vector<double> y_tmp;
		std::vector<double> loc_tmp;
		torch::Tensor new_x = torch::zeros({0,x.sizes()[1]});
	//	std::cout << new_x << std::endl;
		torch::Tensor new_z = torch::zeros({ 0,z.sizes()[1] });
		for (int64_t i = 0; i < phe.vPhenotype.size(); i++)
		{
			nknots.push_back(phe.vPhenotype[i].size());
			torch::Tensor xi = x.index({ i, });
			torch::Tensor zi = z.index({ i, });
			for (int64_t j = 0; j < phe.vPhenotype[i].size(); j++)
			{
				y_tmp.push_back(phe.vPhenotype[i][j]);
				loc_tmp.push_back(phe.vloc[i][j]);
			
			}
			xi = xi.repeat({ phe.vPhenotype[i].size(), 1 });
		//	std::cout << xi << std::endl;
			zi = zi.repeat({ phe.vPhenotype[i].size(), 1 });
			new_x=torch::cat({ new_x,xi }, 0);
			new_z = torch::cat({ new_z,zi }, 0);
		}
	//	std::cout << x << std::endl;
	//	std::cout << z << std::endl;
		x = new_x;
		z = new_z;
	//	std::cout << x << std::endl;
//		std::cout << z << std::endl;
		Eigen::MatrixXd y_Een = Eigen::Map<Eigen::MatrixXd>(y_tmp.data(), y_tmp.size(), 1);
		loc= Eigen::Map<Eigen::VectorXd>(loc_tmp.data(), loc_tmp.size(), 1);
		y = dtt::eigen2libtorch(y_Een);
	}
	if (loc.size() > 0)
	{
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
		if (phe->loc.size() > 0)
		{
			loc = phe->loc.cast<double>();
		}
		nknots.assign(phe->nind, 1);
	}
	else
	{
		std::vector<double> y_tmp;
		std::vector<double> loc_tmp;
		torch::Tensor new_x = torch::zeros({ 0,x.sizes()[1] });
		//	std::cout << new_x << std::endl;
		torch::Tensor new_z = torch::zeros({ 0,z.sizes()[1] });;

		for (int64_t i = 0; i < phe->vPhenotype.size(); i++)
		{

			nknots.push_back(phe->vPhenotype[i].size());
			torch::Tensor xi = x.index({ i, });
			torch::Tensor zi = z.index({ i, });
			for (int64_t j = 0; j < phe->vPhenotype[i].size(); j++)
			{
				y_tmp.push_back(phe->vPhenotype[i][j]);
				loc_tmp.push_back(phe->vloc[i][j]);
			}
			xi = xi.repeat({ phe->vPhenotype[i].size(), 1 });
	//		std::cout << xi << std::endl;
			zi = zi.repeat({ phe->vPhenotype[i].size(), 1 });
			new_x = torch::cat({ new_x,xi }, 0);
			new_z = torch::cat({ new_z,zi }, 0);
		}
	//	std::cout << x << std::endl;
	//	std::cout << z << std::endl;
		x = new_x;
		z = new_z;
	//	std::cout << x << std::endl;
	//	std::cout << z << std::endl;
		Eigen::MatrixXd y_Een = Eigen::Map<Eigen::MatrixXd>(y_tmp.data(), y_tmp.size(), 1);
		loc = Eigen::Map<Eigen::VectorXd>(loc_tmp.data(), loc_tmp.size(), 1);
		y = dtt::eigen2libtorch(y_Een);
	}
	if (loc.size() > 0)
	{
		double  delta_loc = (loc.maxCoeff() - loc.minCoeff()) / 100;
		loc0 = loc.minCoeff() - delta_loc;
		loc1 = loc.maxCoeff() + delta_loc;
	}
	mean_y = torch::full(1, phe->mean);
	std_y = torch::full(1, phe->std);
	nind = phe->nind;
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
		int64_t expand_size = std::accumulate(nknots.begin(), nknots.begin() + i * Batch_size,0);
		Batch_index.push_back(expand_size);
		nind_in_each_batch.push_back(i * Batch_size);

	}
	Batch_index.push_back(std::accumulate(nknots.begin(), nknots.end(), 0));
	nind_in_each_batch.push_back(nind);
}


std::shared_ptr<TensorData> TensorData::getSample(int64_t index)
{
	Eigen::MatrixXf y_tmp( phe->vPhenotype[index].size(),1);
	y_tmp.col(0) = phe->vPhenotype[index];
	torch::Tensor yi = dtt::eigen2libtorch(y_tmp);
	torch::Tensor xi = x.index({ index, });
	xi = xi.repeat({ y_tmp.size(), 1 });

	torch::Tensor zi = z.index({ index, });
	zi = zi.repeat({ y_tmp.size(), 1 });
	auto Sample_i = std::make_shared<TensorData>(yi, xi, zi, pos, phe->vloc[index].cast<double>(),loc0,loc1);
	Sample_i->setMean_STD(mean_y, std_y);
	Sample_i->dataType = this->dataType;
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

std::shared_ptr<TensorData> TensorData::getBatch(int64_t index_batch)
{
	torch::Tensor yi = y.index({ torch::indexing::Slice(Batch_index[index_batch], Batch_index[index_batch+1] ), });
	//std::cout << y.sizes()[0]<<"\t"<< y.sizes()[1]<< std::endl;
	//std::cout << yi << std::endl;
	torch::Tensor xi = x.index({ torch::indexing::Slice(Batch_index[index_batch], Batch_index[index_batch+1] ), });
	//std::cout << x.sizes()[0] << "\t" << x.sizes()[1] << std::endl;
	//std::cout << xi << std::endl;
	torch::Tensor zi = z.index({ torch::indexing::Slice(Batch_index[index_batch], Batch_index[index_batch+1] ), });
	//std::cout << z.sizes()[0] << "\t" << z.sizes()[1] << std::endl;
	//std::cout << zi << std::endl;
	Eigen::VectorXd loci;
	if (yi.sizes()[1]==1 && loc.size()!=0)
	{
		loci.resize(Batch_index[index_batch + 1] - Batch_index[index_batch]);
		loci << loc.block(Batch_index[index_batch], 0, loci.size(), 1);
	}
	else
	{
		loci = loc;
	}
	auto Sample_i = std::make_shared<TensorData>(yi, xi, zi, pos, loci, loc0, loc1);
	Sample_i->isBalanced = isBalanced;
	Sample_i->setMean_STD(yi.mean(), yi.std());
	Sample_i->dataType = this->dataType;
	Sample_i->nknots.insert(Sample_i->nknots.begin(), nknots.begin() + nind_in_each_batch[index_batch], nknots.begin() + nind_in_each_batch[index_batch + 1]);
	Sample_i->nind = Sample_i->nknots.size();
	return Sample_i;
}

void TensorData::setMean_STD(torch::Tensor mean, torch::Tensor std)
{
	this->mean_y = mean;
	this->std_y = std;
}
