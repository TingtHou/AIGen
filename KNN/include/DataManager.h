#pragma once
#define EIGEN_USE_MKL_ALL
#include "CommonFunc.h"
#include <boost/program_options.hpp>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/bimap.hpp>
#include <Eigen/Dense>
#include "KernelManage.h"
#include "PlinkReader.h"
#include "CommonFunc.h"
#include "easylogging++.h"

//Read data from files, impute the missing data, and match phenotype and kernel data
//Usage:
//DataManager dm;
//dm.readPhe("Address of phenotype") ## Read phenotype
//dm.readKernel("Address of kernel file") ## Read a kernel matrix
//dm.readmKernel("Address of kernel list file") ## Read multiple kernels
//dm.readGeno("Address of kernel list file",True) ## Read genotype data from plink format files

class DataManager
{
public:
	DataManager();
	void readPhe(std::string phefilename);											//Read phenotype
	void readKernel(std::string prefix);											//Read a kernel matrix
	void readmKernel(std::string mkernelfilename);									//Read multiple kernels
	void readGeno(std::vector<std::string> filelist, bool isImpute);                //Read genotype data from plink format files, [ped, map] or [bed, bim, fam];
	void readWeight(std::string filename);							                //Read weight file for Iterative MIQNUE
	void readCovariates(std::string qfilename,std::string dfilename);				//Read covariates files, including quantitative and discrete covari
	void readkeepFile(std::string filename);										//Read specific file containing individuals ID for analysis
	PhenoData& getPhenotype() {	return phe;	};										//Get phenotype data
	std::vector<KernelData>* GetKernel() { return &KernelList; };						//Get kernel data
	CovData& GetCovariates() { return Covs; };								//Get covariate data
	Eigen::VectorXf& GetWeights() { return Weights; };								//Get Weights data
	void SetKernel(std::vector<KernelData> KernelList)								//Replace internal kernel data with specificed external kernel data
		{ this->KernelList=KernelList; };
	void match();																	//match phenotype data and kernel data
	GenoData getGenotype() { return geno; };										//Get genotype data
	~DataManager();
private:
	PhenoData phe;																	//phenotype data
	GenoData geno;																	//genotype data
	std::vector<KernelData> KernelList;												//a vector of kernel data
	CovData Covs;																	//intercept + all covariates
	Eigen::VectorXf Weights;														//Weights for Interative MINQUE
	boost::bimap<int, std::string> fid_iid_keeping;										// keep individuals ID for analysis
	
	
private:
	void readResponse(std::string resopnsefile, PhenoData &phe);					//as named
	void readmkernel(std::string mkernel);											//read multiple kernels
	void readCovariates_quantitative(std::string covfilename, CovData & Quantitative);									//Read quantitative Covariates
	void readCovariates_Discrete(std::string covfilename, CovData & Discrete);									//Read discrete Covariates
//	void match(PhenoData &phenotype, KernelData &kernel);							//match phenotype data and one kernel matrix
	void match_Kernels(KernelData& kernel, boost::bimap<int, std::string>& overlapped);

};

