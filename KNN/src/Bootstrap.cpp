#include "../include/Bootstrap.h"

Bootstrap::Bootstrap(DataManager& dm, Random * rd)
{
	this->dm = dm;
	this->rd = rd;
	Ori_phe = dm.getPhenotype();
	Ori_KernelList = dm.GetKernel();
	Ori_Covs = dm.GetCovariates();
	Weights= dm.GetWeights();
	nind = Ori_phe.Phenotype.size();
	Covs.Covariates.resize(Ori_phe.fid_iid.size(), Ori_Covs.npar);
	phe.Phenotype.resize(Ori_phe.fid_iid.size());
	Covs.npar = Ori_Covs.npar;

}

Bootstrap::~Bootstrap()
{
}

void Bootstrap::generate()
{
	Covs.Covariates.setZero();
	phe.Phenotype.setZero();
	KernelList.clear();
	ID_Keeping.clear();
	GenerateID();
	GeneratePhe_Cov();
	for (int i = 0; i < Ori_KernelList->size(); i++)
	{
		KernelData kd;
		GenerateKernel(Ori_KernelList->at(i), kd);
		KernelList.push_back(kd);
	}
}

void Bootstrap::GenerateID()
{
	for (int i = 0; i < nind; i++)
	{
		int id = rd->Uniform(0, nind - 1);
	//	auto fid_iid_each = fid_iid.left.find(id);
		ID_Keeping.push_back(id);
//		std::string keep = fid_iid_each->second;
//		fid_iid_keeping.insert({i++, keep});
	}
}

void Bootstrap::GeneratePhe_Cov()
{
	
	for (int i=0; i< ID_Keeping.size();i++)
	{
//		std::string rowID = it_row->second;
		//auto itcov = Ori_phe.fid_iid.right.find(rowID);
	//	int cov_ID = itcov->second;
		Covs.Covariates.row(i) << Ori_Covs.Covariates.row(ID_Keeping[i]);
//		auto it = Ori_phe.fid_iid.right.find(rowID);
//		int OriPheID = it->second;
		phe.Phenotype[i] = Ori_phe.Phenotype[ID_Keeping[i]];
	}
}

void Bootstrap::GenerateKernel(KernelData& ori_kernel, KernelData& sub_kernel)
{

	sub_kernel.kernelMatrix.resize(nind, nind);
	long long total_size = (nind + 1) * nind / 2;
	/*
	#pragma omp parallel for 
	for (long long k = 0; k < total_size; k++)
	{
		long long i = k / nind, j = k % nind;
		if (j < i) i = nind - i, j = nind - j - 1;
		if (ID_Keeping[i] == ID_Keeping[j] && i != j)
		{
			sub_kernel.kernelMatrix(i, j) = 1;
		}
		else
		{
			sub_kernel.kernelMatrix(i, j)= sub_kernel.kernelMatrix(j,i) = ori_kernel.kernelMatrix(ID_Keeping[i], ID_Keeping[j]);
		}
	}
*/
	int m, n;
	#pragma omp parallel for private(n) collapse(2)
	for ( m = 0; m < ID_Keeping.size(); m++)
	{
		for ( n = 0; n < ID_Keeping.size(); n++)
		{
			if (ID_Keeping[m] == ID_Keeping[n] && m!=n)
			{
				sub_kernel.kernelMatrix(m, n) = 1;
			}
			else
			{
				sub_kernel.kernelMatrix(m, n) = ori_kernel.kernelMatrix(ID_Keeping[m], ID_Keeping[n]);
			}
			//			kernel.VariantCountMatrix(i, j) = tmpKernel.VariantCountMatrix(OriKernelRowID, OriKernelColID);
		}

	}
	
//	sub_kernel.fid_iid = fid_iid_keeping;
}
