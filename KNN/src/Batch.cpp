#include "../include/Batch.h"


Batch::Batch(std::vector<Eigen::MatrixXf>& kernels, Eigen::VectorXf &phe, int splitnum, int seed, bool isclear)
{
	this->kernels = kernels;
	this->splitnum = splitnum;
	nkernels = kernels.size();
	this->phe = phe;
	nInd = phe.size();
	this->seed = seed;
	this->isclear = isclear;
}

void Batch::start()
{
	shuffle();
	if (isclear)
	{
		kernels.clear();
	}
}

void Batch::GetBatchKernels(std::vector<std::vector<Eigen::MatrixXf>>& BatchedKernel)
{
	BatchedKernel = KernelBatched;
}

void Batch::GetBatchPhe(std::vector<Eigen::VectorXf>& phe)
{
	phe = pheBatched;
}

Batch::~Batch()
{
}

void Batch::shuffle()
{
	std::vector<int> shuffledID;
	for (int i=0;i<nInd;i++)
	{
		shuffledID.push_back(i);
	}
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(shuffledID), std::end(shuffledID), rng);
	int nINDinEach = nInd / splitnum;
	std::vector<std::vector<int>> IDinEach;
	std::vector<int> IDs;
	int count = 0;
	for (int i = 0; i < nInd; i++)
	{
		if (count < nINDinEach)
		{
			IDs.push_back(shuffledID[i]);
			count++;
		}
		else
		{
			IDinEach.push_back(IDs);
			IDs.clear();
			count = 0;
			IDs.push_back(shuffledID[i]);
			count++;
		}
	}
	if (IDs.size() > 0)
	{
		IDinEach.push_back(IDs);
		IDs.clear();
		count = 0;
	}
	KernelBatched.resize(IDinEach.size());
	pheBatched.resize(IDinEach.size());
//	#pragma omp parallel for
	for (int i=0;i< IDinEach.size();i++)
	{
		std::vector<Eigen::MatrixXf> KerneleachBatch;
		for (int j=0;j<nkernels;j++)
		{
			Eigen::MatrixXf subMatrx(IDinEach[i].size(), IDinEach[i].size());
			GetSubMatrix(kernels[j], subMatrx, IDinEach[i], IDinEach[i]);
			KerneleachBatch.push_back(subMatrx);
			
		}
		Eigen::VectorXf subVector(IDinEach[i].size());
		GetSubVector(phe, subVector, IDinEach[i]);
		pheBatched[i]=(subVector);
		KernelBatched[i]=(KerneleachBatch);
	}

}
