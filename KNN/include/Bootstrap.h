#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include "DataManager.h"
#include "KernelManage.h"
#include "Random.h"
#include "CommonFunc.h"
#include "Random.h"
class Bootstrap
{
public:
	Bootstrap(DataManager & dm, Random * rd);
	~Bootstrap();
	void generate();
	PhenoData& getPhenotype() { return phe; };										//Get phenotype data
	std::vector<KernelData>* GetKernel() { return &KernelList; };						//Get kernel data
	CovData& GetCovariates() { return Covs; };								//Get covariate data
	Eigen::VectorXf& GetWeights() { return Weights; };								//Get Weights data
private:
	long long nind;
	DataManager dm;
	PhenoData phe;																	//phenotype data
	PhenoData Ori_phe;
	std::vector<KernelData> *Ori_KernelList;
	std::vector<KernelData> KernelList;												//a vector of kernel data
	CovData Covs;																	//intercept + all covariates
	CovData Ori_Covs;																	//intercept + all covariates
	Eigen::VectorXf Weights;														//Weights for Interative MINQUE
	boost::bimap<int, std::string> fid_iid_keeping;										// keep individuals ID for analysis
	std::vector<int> ID_Keeping;
	Random *rd;
	void GenerateID();
	void GeneratePhe_Cov();
	void GenerateKernel(KernelData& ori_kernel, KernelData& sub_kernel);

};