#include "../include/Batch.h"
//@brief:	Initilize batch function;
//@param:	kernels		A vector of Eigen matrixs of variance components;
//@param:	phe			A Eigen vector of phenotype;
//@param:	Covs		A Eigen matrix of covariates;
//@param:	splitnum	The number of batches;
//@param:	seed		The seed for shuffle process;
//@param:	isclear		Clear old "kernels" or not;
//@ret:		void;
Batch::Batch(std::vector<Eigen::MatrixXf *>& kernels, Eigen::VectorXf &phe, Eigen::MatrixXf& Covs, int splitnum, int seed, bool isclear)
{
	this->kernels = kernels;
	this->splitnum = splitnum;
	nkernels = kernels.size();
	this->phe = phe;
	nInd = phe.size();
	this->seed = seed;
	this->isclear = isclear;
	this->Covs = Covs;
}

//@brief:	Starting to generate subset data for each batch;
//@param:	void;
//@ret:		void;
void Batch::start(int dataType)
{
	switch (dataType)
	{
	case 0:
		shuffle();
		break;
	case 1:
		shuffle_binary();
		break;
	case 2:
		throw std::string("KNN cannot hold categorical data currently.");
	}
	if (isclear)
	{
		kernels.clear();
	}
}

//@brief:	Get Kernels for each batch;
//@param:	BatchedKernel	A 2D vector to store kernels for each batch;
//@ret:		void;
void Batch::GetBatchKernels(std::vector<std::vector<Eigen::MatrixXf>>& BatchedKernel)
{
	BatchedKernel = KernelBatched;
}

//@brief:	Get phe for each batch;
//@param:	phe	A vector to store phenotypes for each batch;
//@ret:		void;
void Batch::GetBatchPhe(std::vector<Eigen::VectorXf>& phe)
{
	phe = pheBatched;
}

//@brief:	Get covariate matrixes for each batch;
//@param:	cov		A vector to store covariates for each batch;
//@ret:		void;
void Batch::GetBatchCov(std::vector<Eigen::MatrixXf>& cov)
{
	cov = CovsBatched;
}

Batch::~Batch()
{
}

//@brief:	shuffle the indivisual IDs, and get phenotypes and kernels by shuffled IDs for each batch;
//@param:	void;
//@ret:		void;
void Batch::shuffle()
{
	std::vector<int> shuffledID;
	for (int i=0;i<nInd;i++)
	{
		shuffledID.push_back(i);
	}
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(shuffledID), std::end(shuffledID), rng);
	std::vector<std::vector<int>> IDinEach(splitnum);
	unsigned long nINDinEach = nInd / splitnum;
	std::vector<int> IDs;
	int count = 0;
	int gid = 0;
	for (int i = 0; i < nInd; i++)
	{
		if (count < nINDinEach)
		{
			IDs.push_back(shuffledID[i]);
			count++;
		}
		else
		{
			IDinEach[gid].insert(IDinEach[gid].end(), IDs.begin(), IDs.end());
			//IDinEach.push_back(IDs);
			IDs.clear();
			count = 0;
			IDs.push_back(shuffledID[i]);
			count++;
			gid++;
		}
	}
	gid = 0;
	for (int i = 0; i < IDs.size(); i++)
	{
		IDinEach[gid].insert(IDinEach[gid].end(), IDs[i]);
		if (gid >= splitnum)
		{
			gid++;
		}
		else
		{
			gid = 0;
		}
	}
	IDs.clear();
	/*
	if (IDs.size() > 0)
	{
		IDinEach.push_back(IDs);
		IDs.clear();
		count = 0;
	}888*/
	KernelBatched.resize(IDinEach.size());
	pheBatched.resize(IDinEach.size());
	CovsBatched.resize(IDinEach.size());
//	#pragma omp parallel for
	for (int i=0;i< IDinEach.size();i++)
	{
		std::vector<Eigen::MatrixXf> KerneleachBatch;
		for (int j=0;j<nkernels;j++)
		{
			Eigen::MatrixXf subMatrx(IDinEach[i].size(), IDinEach[i].size());
			GetSubMatrix(kernels[j], &subMatrx, IDinEach[i], IDinEach[i]);
			KerneleachBatch.push_back(subMatrx);
			
		}
		Eigen::VectorXf subVector(IDinEach[i].size());
		GetSubVector(phe, subVector, IDinEach[i]);
		Eigen::MatrixXf subCovMatrix(IDinEach[i].size(), Covs.cols());
		GetSubMatrix(&Covs, &subCovMatrix, IDinEach[i]);
		pheBatched[i]=(subVector);
		KernelBatched[i]=(KerneleachBatch);
		CovsBatched[i] = subCovMatrix;
	}

}

void Batch::shuffle_binary()
{
	std::vector<int> fid_iid_case;
	std::vector<int> fid_iid_control;
	for (int i = 0; i < phe.size(); i++)
	{
		if (abs(phe[i]-1)>10^-4)
		{
			fid_iid_case.push_back(i);
		}
		else
		{
			fid_iid_control.push_back(i);
		}
	}
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(fid_iid_case), std::end(fid_iid_case), rng);
	std::shuffle(std::begin(fid_iid_control), std::end(fid_iid_control), rng);

	unsigned long nINDinEach_case = fid_iid_case.size() / splitnum;
	unsigned long nINDinEach_contorl = fid_iid_control.size() / splitnum;
	std::vector<std::vector<int>> IDinEach;
	std::vector<int> IDs;
	int count = 0;
	int gid = 0;
	for (int i = 0; i < fid_iid_case.size(); i++)
	{
		if (count < nINDinEach_case)
		{
			IDs.push_back(fid_iid_case[i]);
			count++;
		}
		else
		{
			IDinEach[gid].insert(IDinEach[gid].end(), IDs.begin(), IDs.end());
			IDinEach.push_back(IDs);
			IDs.clear();
			count = 0;
			IDs.push_back(fid_iid_case[i]);
			count++;
			gid++;
		}
	}

	gid = 0;
	for (int i = 0; i < IDs.size(); i++)
	{
		IDinEach[gid].insert(IDinEach[gid].end(), IDs[i]);
		if (gid >= splitnum)
		{
			gid++;
		}
		else
		{
			gid = 0;
		}
	}
	IDs.clear();
	/*
	if (IDs.size() > 0)
	{
		IDinEach.push_back(IDs);
		IDs.clear();
		count = 0;
	}
	*/
	count = 0;
	int batch_id = 0;
	for (int i = 0; i < fid_iid_control.size(); i++)
	{
		if (count < nINDinEach_contorl)
		{
			IDs.push_back(fid_iid_control[i]);
			count++;
		}
		else
		{
			IDinEach[batch_id].insert(IDinEach[batch_id].end(),IDs.begin(),IDs.end());
			IDs.clear();
			count = 0;
			IDs.push_back(fid_iid_control[i]);
			count++;
			batch_id++;
		}
	}

	gid = 0;
	for (int i = 0; i < IDs.size(); i++)
	{
		IDinEach[gid].insert(IDinEach[gid].end(), IDs[i]);
		if (gid >= splitnum)
		{
			gid++;
		}
		else
		{
			gid = 0;
		}
	}
	IDs.clear();
	/*
	if (IDs.size() > 0)
	{
		IDinEach[batch_id].insert(IDinEach[batch_id].end(), IDs.begin(), IDs.end());
		IDs.clear();
		count = 0;
	}
	*/
	
	KernelBatched.resize(IDinEach.size());
	pheBatched.resize(IDinEach.size());
	CovsBatched.resize(IDinEach.size());
	//	#pragma omp parallel for
	for (int i = 0; i < IDinEach.size(); i++)
	{
		std::vector<Eigen::MatrixXf> KerneleachBatch;
		for (int j = 0; j < nkernels; j++)
		{
			Eigen::MatrixXf subMatrx(IDinEach[i].size(), IDinEach[i].size());
			GetSubMatrix(kernels[j], &subMatrx, IDinEach[i], IDinEach[i]);
			KerneleachBatch.push_back(subMatrx);

		}
		Eigen::VectorXf subVector(IDinEach[i].size());
		GetSubVector(phe, subVector, IDinEach[i]);
		Eigen::MatrixXf subCovMatrix(IDinEach[i].size(), Covs.cols());
		GetSubMatrix(&Covs, &subCovMatrix, IDinEach[i]);
		pheBatched[i] = (subVector);
		KernelBatched[i] = (KerneleachBatch);
		CovsBatched[i] = subCovMatrix;
	}
}
