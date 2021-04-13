// MINQUE.cpp : This file contains the 'main' function. Program execution begins and ends there.


#include <iostream>
#include <mkl.h>
#include <map>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <time.h>
#include <iomanip>
#include <random>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <boost/date_time.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/math/distributions/normal.hpp>
#include "../include/KernelGenerator.h"
#include "../include/KernelExpansion.h"
#include "../include/Batch.h"
#include "../include/easylogging++.h"
#include "../include/ToolKit.h"
#include "../include/LinearRegression.h"
#include "../include/Random.h"
#include "../include/PlinkReader.h"
#include "../include/Options.h"
#include "../include/CommonFunc.h"
#include "../include/KernelManage.h"
#include "../include/DataManager.h"
#include "../include/imnq.h"
#include "../include/Prediction.h"
#include "../include/MINQUE0.h"
//#include "../include/Bootstrap.h"
#include "../include/Random.h"
#include "../include/KNN_Imp.h"
#include "../include/FNN.h"
#include "../include/NN.h"
#include "../include/TensorData.h"
#include "../include/TensorCommon.h"
#include <torch/torch.h>
#include <type_traits>
/*
#ifndef CPU  
	#include "../include/cuMINQUE0.h"
	#include "../include/cuimnq.h" 
#endif
*/


INITIALIZE_EASYLOGGINGPP

void readKNNParameter(boost::program_options::variables_map programOptions, MinqueOptions& minque);

void ReadData(boost::program_options::variables_map programOptions, DataManager &dm);

int MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager& dm, Eigen::VectorXf& VarComp, Eigen::VectorXf& predict);
std::vector<std::shared_ptr<Evaluate>>   FNNAnalysis(boost::program_options::variables_map programOptions, DataManager& dm);



//void Bootstraping(boost::program_options::variables_map programOptions, MinqueOptions& minopt, DataManager& dm, int times, Random& seed,
//	std::vector<Eigen::VectorXf> &Vars_BP, std::vector<Eigen::VectorXf> &Predicts_BP, std::vector<int> &iterateTimes_BP);
/*
#ifndef CPU  
void cudaMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs);
void cudaMINQUE1(MinqueOptions& minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho);
#endif
*/

void TryMain(int argc, const char *const argv[])
{
	boost::bimap<std::string,int> kernelPool;
	kernelPool.insert({ "car",CAR });
	kernelPool.insert({ "identity",Identity });
	kernelPool.insert({ "product",Product });
	kernelPool.insert({ "polynomial",Polymonial });
	kernelPool.insert({ "gaussian",Gaussian });
	kernelPool.insert({ "ibs",IBS });
	Options opt(argc, argv);
//	MinqueOptions minopt;

	boost::program_options::variables_map programOptions = opt.GetOptions();
	if (1 == argc || programOptions.count("help"))
	{
		std::cout << opt.GetDescription() << std::endl;
		return;
	}
	if (programOptions.count("version"))
	{
		std::cout << "Kernel Based Neural Network Software alpha 0.1.1" << std::endl;
		std::cout << "Release Date: "<<__DATE__<<" "<<__TIME__ << std::endl;
		return;

	}
	string result= "result.txt";
	std::string logfile="result.log";
	if (programOptions.count("out"))
	{
		result = programOptions["out"].as < std::string >();
		std::string basename = GetBaseName(result);
		std::string ParentPath = GetParentPath(result);
		int positionDot = basename.rfind('.');
		logfile = ParentPath + "/" + basename.substr(0, positionDot) + ".log";
		programOptions.erase("out");
	}
	if (programOptions.count("log"))
	{
		logfile = programOptions["log"].as < std::string >();
		programOptions.erase("log");
	}
	if (programOptions.count("thread"))
	{
		int nthread = programOptions["thread"].as<int>();
		omp_set_num_threads(nthread);
		//	mkl_set_num_threads(nthread);
	}
	
//////init logger
	el::Configurations conf;
	conf.setToDefault();
	conf.parseFromText(("*GLOBAL:\n  FILENAME = " + logfile+"\n  To_Standard_Output = false"));
	el::Loggers::reconfigureLogger("default", conf);
	std::cout << opt.print() << std::endl;
	LOG(INFO) << opt.print();

	DataManager dm;
	ReadData(programOptions, dm);
	dm.match();
	////////////////////////////////////////////////////////////////////////////////////
	/////FNN or NN implemnt
	//////////////////////////////////////////////////////////////////////////////
	if (programOptions.count("FNN") || programOptions.count("NN"))
	{
		std::vector<std::shared_ptr<Evaluate>>  error = FNNAnalysis(programOptions, dm);
		ofstream out;
		out.open(result, ios::out);
		LOG(INFO) << "---Result----";
		std::cout << "---Result----" << std::endl;
		std::stringstream ss;
	
		if (programOptions.count("load"))
		{
			LOG(INFO) << "---full dataset----";
			ss << "---full dataset----" << std::endl;
			if (dm.getPhenotype().dataType != 0)
			{
				ss << "misclassification error:\t" << error[0]->getMSE() << std::endl << "AUC:\t" << error[0]->getAUC() << std::endl;
			}
			else
			{
				ss << "MSE:\t" << error[0]->getMSE() << std::endl << "Correlation:\t" << error[0]->getCor().transpose() << std::endl;
			}
			std::cout << ss.str();
			LOG(INFO) << ss.str();
			out << ss.str();
		}
		else
		{
			if (error[0] != nullptr)
			{
				LOG(INFO) << "---Testing dataset----";
				ss << "---Testing dataset----" << std::endl;
				if (dm.getPhenotype().dataType != 0)
				{
					ss << "misclassification error:\t" << error[0]->getMSE() << std::endl << "AUC:\t" << error[0]->getAUC() << std::endl;
				}
				else
				{
					ss << "MSE:\t" << error[0]->getMSE() << std::endl << "Correlation:\t" << error[0]->getCor().transpose() << std::endl;
				}
			}
			else
			{
			
				LOG(WARNING) << "Testing dataset does not exist.";
			}
			
			LOG(INFO) << "---full dataset----";
			ss << "---full dataset----" << std::endl;
			if (dm.getPhenotype().dataType != 0)
			{
				ss << "misclassification error:\t" << error[1]->getMSE() << std::endl << "AUC:\t" << error[1]->getAUC() << std::endl;
			}
			else
			{
				ss << "MSE:\t" << error[1]->getMSE() << std::endl << "Correlation:\t" << error[1]->getCor().transpose() << std::endl;
			}
			std::cout << ss.str();
			LOG(INFO) << ss.str();
			out << ss.str();
		}
	
		out.close();
	}

	/////////////////////////////////////////////////////////
	///KNN implement
	////////////////////////////////////////////////////////////////
	std::vector<KernelData> kernelList; 
		//generate a built-in kernel or not
	if (programOptions.count("make-kernel"))
	{
		int kerneltype=-1;
		std::string kernelname = programOptions["make-kernel"].as < std::string >();
		if (isNum(kernelname))
		{
			kerneltype = stoi(kernelname);
		}
		else
		{

			boost::to_lower(kernelname);
			if (kernelPool.left.count(kernelname))
			{
				auto it = kernelPool.left.find(kernelname);
				kerneltype = it->second;
			}
			
		}
		if (kerneltype!=-1)
		{
			auto it = kernelPool.right.find(kerneltype);
			std::cout << "Generate a " + it->second + " kernel from input genotype"<< std::endl;
			LOG(INFO) << "Generate a " + it->second + " kernel from input genotype";
		}
		else
		{
			throw std::string("Error: Invalided kernel name \"" + kernelname+"\"");
		}
		GenoData gd = dm.getGenotype();
		if (programOptions.count("std"))
		{
			std::cout <<"Standardize the genotype matrix" << std::endl;
			LOG(INFO) << "Standardize the genotype matrix";
			stdSNPmv(gd.Geno);
		}
		Eigen::VectorXf weight(gd.Geno.cols());
		weight.setOnes();
		// read weight vector from specific file. If the file is not settled, the weight will be identity vector.
		if (programOptions.count("weight"))
		{
			std::string weightVFile = programOptions["weight"].as < std::string >();
			ifstream infile;
			infile.open(weightVFile);
			if (!infile.is_open())
			{
				throw std::string("Error: cannot open the file [" + weightVFile + "] to read.");
			}
			int id = 0;
			while (!infile.eof())
			{
				std::string eleInVector;
				getline(infile, eleInVector);
				if (!eleInVector.empty() && eleInVector.back() == 0x0D)
					eleInVector.pop_back();
				if (infile.fail())
				{
					continue;
				}
				boost::algorithm::trim(eleInVector);
				if (eleInVector.empty())
				{
					continue;
				}
				if (id < weight.size())
				{
					weight[id++] = stod(eleInVector);
				}
			}
		}
		float constant = 1;
		float deg = 2;
		float sigmma = 1;
		bool scale = false;
		if (programOptions.count("constant"))
			constant=programOptions["constant"].as < float >();
		if (programOptions.count("deg"))
			deg = programOptions["deg"].as < float >();
		if (programOptions.count("sigma"))
			sigmma = programOptions["sigma"].as < float >();
		if (programOptions.count("scale"))
			scale = programOptions["scale"].as < bool >();
		KernelGenerator kernelGenr(gd, kerneltype, weight,scale, constant, deg, sigmma);
		kernelList.push_back(kernelGenr.getKernel());
	}
	//The built-in kernel will overwrite the kernel from file
	if (kernelList.size())
	{
		dm.SetKernel(kernelList);
	}
	if (programOptions.count("recode"))
	{

		for (int i = 0; i < kernelList.size(); i++)
		{
			KernelWriter kw(kernelList[i]);
			std::string outname = programOptions["recode"].as < std::string >();
			kw.writeText(outname);
		}
	}
	//output kernel matrices as binary format
	if (programOptions.count("make-bin"))
	{
	//	throw std::string("Error: make-bin function is under maintenance, and will back soon.");
		std::string outname = programOptions["make-bin"].as < std::string >();
		bool isfloat = true;
		if (programOptions.count("precision"))
		{
			isfloat = programOptions["precision"].as < int >();
		}
		for (int i = 0; i < dm.GetKernel()->size(); i++)
		{
			if (i > 1)
			{
				outname += i;
			}
			KernelWriter kw(dm.GetKernel()->at(i));

			kw.setprecision(isfloat);
			kw.write(outname);
		}
		
	}
	//if the phenotype is inputed, the estimation will be started.
	if (programOptions.count("KNN"))
	{
			Eigen::VectorXf VarComp;
			Eigen::VectorXf predict;
			int iterateTimes=MINQUEAnalysis(programOptions, dm, VarComp,predict);

			ofstream out;
			out.open(result, ios::out);
			LOG(INFO) << "---Result----";
			std::cout << "---Result----" << std::endl;
			out << "Source\tVariance" << std::endl;
			LOG(INFO) << "Source\tVariance";
			int i = 0;
			float VG = 0;
			float VP = 0;
			std::stringstream ss;
			for (; i < VarComp.size() - 1; i++)
			{
				//	std::cout << VarComp.at(i) << " ";
				int index = i + 1;
				ss.str("");
				ss << "V(G" << index << ")\t" << VarComp(i);
				out << ss.str() << std::endl;
				std::cout << ss.str() << std::endl;
				LOG(INFO) << ss.str();
				VG += VarComp(i);
			}
			VP = VG;
			ss.str("");
			ss << "V(e)\t" << VarComp(i);
			out << ss.str() << std::endl;
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str();
			VP += VarComp(i);
			ss.str("");
			ss << "Vp\t" << VP;
			out << ss.str() << std::endl;
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str();
			for (i = 0; i < VarComp.size() - 1; i++)
			{
				int index = i + 1;
				ss.str("");
				ss << "V(G" << index << ")/Vp\t" << VarComp(i) / VP;
				out << ss.str() << std::endl;
				std::cout << ss.str() << std::endl;
				LOG(INFO) << ss.str();
			}
			ss.str("");
			ss << "Sum of V(G)/Vp\t" << VG / VP;
			out << ss.str() << std::endl;
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str();
			ss.str("");
			if (dm.getPhenotype().dataType==1)
			{
				boost::math::normal dist(0.0, 1.0);
				double h_O = VG / VP;
				double K = dm.getPhenotype().Phenotype.col(0).mean();
				double q = quantile(dist, K);
				double z = pdf(dist, q);
				double h_l = h_O * K * (1 - K) * z * z;
				ss << "liability h\t" << h_l;
			}
			ss << "Iterate Times:\t" << iterateTimes;
			out << ss.str() << std::endl;
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str();
			if (programOptions.count("predict"))
			{
				int mode = programOptions["predict"].as<int>();
				std::cout << "---Prediction----" << std::endl;
				out << "---Prediction----" << std::endl;

				std::stringstream ss;
				if (dm.getPhenotype().dataType ==1)
				{
					ss << "misclassification error:\t" << predict[0] << std::endl << "AUC:\t" << predict[1] << std::endl;
				}
				else
				{
					ss << "MSE:\t" << predict[0] << std::endl << "Correlation:\t" << predict[1] << std::endl;
				}
				std::cout << ss.str();
				LOG(INFO) << ss.str();
				out << ss.str();
			}
			out.close();

	}
	
}

int main(int argc, const char *const argv[])
{
	/////////////////////////////////////////////////////////////////////
	std::cout<<"\n"
		"@----------------------------------------------------------@\n"
		"|        KNN       |     v alpha 1.0.1  |    "<<__DATE__<<"   |\n"
		"|----------------------------------------------------------|\n"
		"|    Statistical Genetics and Statistical Learning Group   |\n"
		"@----------------------------------------------------------@\n"
		"\n";
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	try
	{
		omp_set_nested(true);
		omp_set_dynamic(false);
		mkl_set_dynamic(false);
		TryMain(argc, argv);
	}
	catch (string &e)
	{
		std::cout << e << std::endl;
		LOG(ERROR) << e;
	}
	catch (const std::exception &err)
	{
		std::cout << err.what() << std::endl;
		LOG(ERROR) << err.what();
	}
	catch (...)
	{
		std::cout << "Unknown Error" << std::endl;
		LOG(ERROR) << "Unknown Error";
	}
	//get now time
	boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
	std::cout << "\n\nAnalysis finished: " << timeLocal << std::endl;
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::ratio<1, 1>> duration_s(end - start);
	//std::cout << "Total elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	cout << "Computational time: " << duration_s.count() <<" second(s)."<< endl;
	return 0;
}

void ReadData(boost::program_options::variables_map programOptions, DataManager &dm)
{
	std::vector<std::string> GFile;
	bool isImpute = false;
	std::string qcovariatefile;
	std::string dcovariatefile;
	if (programOptions.count("phe"))
	{
		std::string reponsefile = programOptions["phe"].as < std::string >();
		std::cout << "Reading Phenotype from [" + reponsefile + "]." << std::endl;
		LOG(INFO) << "Reading Phenotype from [" + reponsefile + "].";
		dm.readPhe(reponsefile);
	}
	if (programOptions.count("covar"))
	{
		dcovariatefile = programOptions["covar"].as < std::string >();
	}
	if (programOptions.count("qcovar"))
	{
		qcovariatefile = programOptions["qcovar"].as < std::string >();
	}
	bool intercept = programOptions["intercept"].as<bool>();
	dm.readCovariates(qcovariatefile, dcovariatefile,intercept);
	if (programOptions.count("kernel"))
	{
		std::string kernelfiles = programOptions["kernel"].as<std::string >();
		std::cout << "Reading kernel list from [" + kernelfiles + "]." << std::endl;
		LOG(INFO) << "Reading kernel list from [" + kernelfiles + "].";
		dm.readKernel(kernelfiles);
	}
	if (programOptions.count("mkernel"))
	{
		std::string mkernelfile = programOptions["mkernel"].as<std::string >();
		std::cout << "Reading kernel list from [" + mkernelfile + "]." << std::endl;
		LOG(INFO) << "Reading kernel list from [" + mkernelfile + "].";
		dm.readmKernel(mkernelfile);
	}
	if (programOptions.count("weights"))
	{
		std::string weightfile = programOptions["weights"].as<std::string >();
		std::cout << "Reading weights from [" + weightfile + "]." << std::endl;
		LOG(INFO) << "Reading weights from [" + weightfile + "].";
		dm.readWeight(weightfile);
	}
	if (programOptions.count("keep"))
	{
		std::string keepingfile = programOptions["keep"].as < std::string >();
		std::cout << "Reading keeping individuals from [" + keepingfile + "]." << std::endl;
		LOG(INFO) << "Reading keeping individuals from [" + keepingfile + "].";
		dm.readkeepFile(keepingfile);
	}
	if (programOptions.count("bfile"))
	{
		std::string Prefix = programOptions["bfile"].as<std::string >();
		GFile.push_back(Prefix + ".bed");
		GFile.push_back(Prefix + ".bim");
		GFile.push_back(Prefix + ".fam");
	}
	if (programOptions.count("file"))
	{
		std::string Prefix = programOptions["file"].as<std::string >();
		GFile.push_back(Prefix + ".ped");
		GFile.push_back(Prefix + ".map");
	}
	if (programOptions.count("ped"))
	{
		std::string filename = programOptions["ped"].as<std::string >();
		GFile.push_back(filename);
	}
	if (programOptions.count("map"))
	{
		std::string filename = programOptions["map"].as<std::string >();
		GFile.push_back(filename);
	}
	if (programOptions.count("bed"))
	{
		std::string filename = programOptions["bed"].as<std::string >();
		GFile.push_back(filename);
	}
	if (programOptions.count("bim"))
	{
		std::string filename = programOptions["bim"].as<std::string >();
		GFile.push_back(filename);
	}
	if (programOptions.count("fam"))
	{
		std::string filename = programOptions["fam"].as<std::string >();
		GFile.push_back(filename);
	}
	if (programOptions.count("impute"))
	{
		isImpute = programOptions["impute"].as<bool>();
	}
	
	if (GFile.size()!=0)
	{
		dm.readGeno(GFile, isImpute);
		/*
		std::ofstream out;
		Eigen::MatrixXf g = dm.getGenotype().Geno;
		out.open("../train/g.data");
		out << g;
		out.close();
		*/
	}
	
}

int MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager &dm, Eigen::VectorXf &VarComp, Eigen::VectorXf &predict)
{
	MinqueOptions minopt;
	readKNNParameter(programOptions, minopt);
	bool GPU = false;
	if (programOptions.count("GPU"))
	{
		GPU = true;
	}
	PhenoData phe=dm.getPhenotype();
	std::vector<KernelData>* kd;
	kd= dm.GetKernel();
	CovData Covs=dm.GetCovariates();
	VarComp=dm.GetWeights();
		// initialize the variance components vector with pre-set weights
	Eigen::VectorXf fix(Covs.npar);
	fix.setZero();
	fix[0] = -999;
	float iterateTimes = 0;
	bool isecho = false;
	std::vector<Eigen::MatrixXf *> Kernels;
	if (programOptions.count("echo"))
	{
		isecho = programOptions["echo"].as<bool>();
	}
	if (programOptions.count("alphaKNN"))
	{
	//	throw std::string("Error: alphaKNN function is under maintenance, and will back soon.");
		
		int alpha = programOptions["alphaKNN"].as<int>();
		KernelExpansion ks(kd, alpha);
	//	Kernels = ks.GetExtendMatrix();
		auto Kmatrices = ks.GetExtendMatrix();
		Kernels.resize(Kmatrices->size());
		for (int i = 0; i < Kmatrices->size(); i++)
		{
			Kernels[i] = new Eigen::MatrixXf(Kmatrices->at(i).rows(), Kmatrices->at(i).cols());
		   //	Kernels.push_back(&(Kmatrices->at(i)));
			*Kernels[i] = Kmatrices->at(i);
		}
		kd->clear();
	}
	else
	{
		for (int i = 0; i < kd->size(); i++)
		{
			Kernels.push_back(&(kd->at(i).kernelMatrix));
		}
	}
	if (VarComp.size() != 0)
	{
		if (VarComp.size() != (Kernels.size() + 1))
		{
			throw std::string("Error: The size of pre-specified weight vector is not equal to the number of variance components.");
		}
	}
	if (programOptions.count("fix"))
	{
		fix[0] = 1;
	}
	if (programOptions.count("predict"))
	{
		if (!programOptions.count("fix"))
		{
			throw std::string("Error: arguments [--fix] and [--predict] must be used at the same time.");
		}
		
	}
	if (GPU)
	{
		/*
#ifndef CPU  
		if (programOptions.count("minque0"))
		{
			cudaMINQUE0(minopt, Kernels, phe, Covs.Covariates, VarComp, fix);
			iterateTimes = 1;
		}
		else
		{
			cudaMINQUE1(minopt, Kernels, phe, Covs.Covariates, VarComp, fix, iterateTimes, isecho);
		}
#else*/
		throw std::string("This is a CPU program. Please use GPU version.");
//#endif
	}
	else
	{
		if (programOptions.count("batch"))
		{
	//		throw std::string("Error: batch function is under maintenance, and will back soon.");
			
			int nthread = 10;
			int nsplit= programOptions["batch"].as<int>();
			int seed = 0;
			if (!programOptions.count("pseudo"))
			{
				minopt.allowPseudoInverse = 0;
			}
			if (programOptions.count("thread"))
			{
				nthread = programOptions["thread"].as<int>();
			}
			if (programOptions.count("seed"))
			{
				seed = programOptions["seed"].as<int>();
			}
			if (programOptions.count("minque0"))
			{
				BatchMINQUE0(minopt, Kernels, phe, Covs.Covariates, VarComp, fix, nsplit, seed, nthread);
				iterateTimes = 1;
			}
			else
			{
				BatchMINQUE1(minopt, Kernels, phe, Covs.Covariates, VarComp, fix, iterateTimes, nsplit, seed, nthread, isecho);
			}
			
			
		}
		else
		{
			isecho = true;
			if (programOptions.count("echo"))
			{
				isecho = programOptions["echo"].as<bool>();
			}	
			if (programOptions.count("minque0"))
			{
				cMINQUE0(minopt, Kernels, phe, Covs.Covariates, VarComp, fix);
				iterateTimes = 1;
			}
			else
			{
				if (!VarComp.size())
				{
					std::cout << "Using results from MINQUE(0) as inital value." << std::endl;
					LOG(INFO) << "Using results from MINQUE(0) as inital value.";
					std::vector<Eigen::MatrixXf*> Kernels_imq(Kernels.size());
					int nind = phe.fid_iid.size();
					for (int i = 0; i < Kernels.size(); i++)
					{
						Kernels_imq[i] = new Eigen::MatrixXf(nind, nind);
						*Kernels_imq[i] = *Kernels[i];
					}
					cMINQUE0(minopt, Kernels_imq, phe, Covs.Covariates, VarComp, fix);
					for (int i = 0; i < Kernels.size(); i++)
					{
						delete Kernels_imq[i];
						Kernels_imq[i] = nullptr;
					}
				}
				
				cMINQUE1(minopt, Kernels, phe, Covs.Covariates, VarComp, fix, iterateTimes, isecho);
			}
			
		}
	}
	
	if (programOptions.count("predict"))
	{
		int mode = programOptions["predict"].as<int>();
		std::cout << "---Prediction----" << std::endl;
	//	out<< "---Prediction----" << std::endl;
		Eigen::VectorXf pheV;
		if (phe.Phenotype.cols() == 1)
		{
			pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
		}
		else
		{
			throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
		}
		Prediction pred(pheV, Kernels, VarComp, Covs.Covariates, fix, phe.dataType==1? true:false, mode);
		Evaluate Eva(pheV, pred.getPredictY(),phe.dataType);
		predict.resize(2);
		
		std::stringstream ss;
		if (phe.dataType==1)
		{
			predict[0] = Eva.getMSE();
			predict[1] = Eva.getAUC();
			//ss << "misclassification error:\t" << pred.getMSE() << std::endl << "AUC:\t" << pred.getAUC() << std::endl;
		}
		else
		{
			predict[0] = Eva.getMSE();
			predict[1] = Eva.getCor()[0];
			//ss << "MSE:\t" << pred.getMSE() << std::endl << "Correlation:\t" << pred.getCor() << std::endl;
		}
//		std::cout << ss.str();
//		LOG(INFO) << ss.str();
//		out<<  ss.str();
	}
	return iterateTimes;
//	out.close();
}

std::vector<std::shared_ptr<Evaluate>>  FNNAnalysis(boost::program_options::variables_map programOptions, DataManager& dm)
{
	std::vector<std::shared_ptr<Evaluate>> prediction_error;
 /// All data in FNN/NN framework will be treated as double precision.
	auto options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor one = torch::ones(1, options);
	torch::set_default_dtype(one.dtype());
	///////////////////////////////////////////////
	int basis= programOptions["basis"].as < int >();
	float seed = programOptions["seed"].as < float >();
	int epoch= programOptions["epoch"].as < int >();
	float ratio= programOptions["ratio"].as < float >();
	double lambda = programOptions["lambda"].as <double>();
	bool isFNN = programOptions.count("FNN");
	int64_t ncovs = dm.getCov_prt()->npar;
	int64_t nSNPs = dm.getGeno_prt()->pos.size();
	if (!ncovs && !nSNPs)
	{
		throw std::string("Error: there is not input data, genotype data and covariates data are missing. Please check the inputs.");
	}
	if (ncovs && !nSNPs)
	{
		isFNN = false;
		LOG(WARNING) << "Warning: The genotype data is missing, only the covariates data are used. NN is applied instead.";
	}
	std::shared_ptr< Dataset> data_full = dm.GetDataset();
	std::shared_ptr<Dataset> train=nullptr;
	std::shared_ptr<Dataset> test =nullptr;
	
	std::shared_ptr<TensorData> data = std::make_shared<TensorData>(data_full->phe, data_full->geno, data_full->cov);
	bool sinlgeknot = (data->isBalanced && data->getLoc().size() == data->nind) || !data->isBalanced;   // univariate analysis, with response is interpolated on different knot.
																										// if the data is unbalanced, we train the model per object, which will be considered as  univariate analysis, with response is interpolated on different knot.

	int  loss = programOptions["loss"].as < int >();

	if (loss!=0 && dm.getPhe_prt()->Phenotype.cols()!=1)
	{
		throw std::string("The multivarite multiclass analysis does not support!");
	}

	std::vector<int64_t> dims;
	
	if (programOptions.count("layer"))
	{
		std::string layers = programOptions["layer"].as < std::string >();
		std::vector<std::string> strVec;
		boost::algorithm::split(strVec, layers, boost::algorithm::is_any_of(","), boost::token_compress_on);
		for (int i = 0; i < strVec.size(); i++)
		{
			dims.push_back(atoi(strVec.at(i).c_str()));
		}
		if (dims[dims.size()-1]>1 && dm.getPhe_prt()->vloc.size()==0)
		{
			throw std::string("Trying to interplote the response, but the knots are missing.");
		}
	}
	if (programOptions.count("load"))
	{
		std::string loadNet = programOptions["load"].as < std::string >();
		//	string model_path = "model.pt";
		std::cout << "Loading model from ["<<loadNet<<"]." << std::endl;
		if (isFNN)
		{
			
			std::cout << "Funtional neural network analysis is runing" << std::endl;
			LOG(INFO) << "Funtional neural network analysis is runing";
			std::shared_ptr<FNN> f = std::make_shared<FNN>(dims, lambda);
			if (!basis)
			{

				f->build<Haar>(sinlgeknot, ncovs);
			}
			else
			{
				f->build<SplineBasis>(sinlgeknot, ncovs);
			}
			torch::serialize::InputArchive archive;
		//	std::string file("test_model.pt");
			archive.load_from(loadNet);
			f->load(archive);
			std::shared_ptr< Evaluate> Total = nullptr;
			if (data->isBalanced)
			{
				torch::Tensor pred_test = f->forward(data);
				Total = std::make_shared< Evaluate>(data->getY(), pred_test, data->dataType);
				//	Evaluate Total(data->getY(), pred_test, data->dataType);
			}
			else
			{
				if (data->dataType != 0)
				{
					size_t i = 0;
					auto sample = data->getSample(i++);
					torch::Tensor pred_test = f->forward(sample);
					for (; i < data->nind; i++)
					{
						auto sample = data->getSample(i);
						torch::Tensor pred_test_new = f->forward(sample);
						pred_test = torch::cat({ pred_test, pred_test_new }, 0);
					}
					Total = std::make_shared< Evaluate>(data->getY(), pred_test, data->dataType);
				}

			}
			prediction_error.push_back(Total);
		}
		else
		{
			std::cout << "Neural network analysis is runing" << std::endl;
			LOG(INFO) << "Neural network analysis is runing";
			//	std::cout << "========================" << std::endl;
			std::shared_ptr<NN> f = std::make_shared<NN>(dims, lambda);
			f->build(ncovs);
			torch::serialize::InputArchive archive;
			//	std::string file("test_model.pt");
			archive.load_from(loadNet);
			f->load(archive);

			torch::Tensor pred_total = f->forward(data);
			std::shared_ptr<Evaluate> Total=std::make_shared<Evaluate>(data->getY(), pred_total, data->dataType);
					   			
			prediction_error.push_back(Total);
		}
	}
	else
	{
		std::stringstream ss;
		ss << "Split the dataset into two sub-dataset, training, testing";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		ss.str("");
		ss.clear();
		std::tie(train, test) = data_full->split(seed, ratio);
		ss << "Spliting Completed. There are " << train->phe.nind << " individuals in the training dataset, and " << test->phe.nind << " individuals in the testing dataset";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		ss.str("");
		ss.clear();
		std::shared_ptr<Dataset> subtrain = nullptr;
		std::shared_ptr<Dataset> valid = nullptr;

		ss << "Split the training dataset into two sub-dataset, subtraining, validation.";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		ss.str("");
		ss.clear();
		std::tie(subtrain, valid) =train->split(seed, ratio);
		//std::shared_ptr<TensorData> d=std::make_shared<TensorData>(dm.getPhenotype(), dm.getGenotype(), dm.GetCovariates());
		std::cout << "Apply the training data for Analysis." << std::endl;
		ss << "Spliting Completed. There are " << subtrain->phe.nind << " individuals in the training dataset used for training, and " << valid->phe.nind << " individuals for validation.";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		ss.str("");
		ss.clear();
	//	std::shared_ptr<TensorData> train_tensor = std::make_shared<TensorData>(train->phe, train->geno, train->cov);
		std::shared_ptr<TensorData> test_tensor = std::make_shared<TensorData>(test->phe, test->geno, test->cov);
	//	train_tensor->dataType = loss;
		test_tensor->dataType = loss;
		//std::shared_ptr<TensorData> subtrain_tensor = nullptr;
	//	std::shared_ptr<TensorData> valid_tensor =nullptr;
		std::shared_ptr<TensorData> subtrain_tensor = std::make_shared<TensorData>(subtrain->phe, subtrain->geno, subtrain->cov);
		std::shared_ptr<TensorData> valid_tensor = std::make_shared<TensorData>(valid->phe, valid->geno, valid->cov);
		subtrain_tensor->dataType = loss;
		valid_tensor->dataType = loss;
		//std::tie(subtrain, valid) = train_tensor->GetsubTrain(ratio);
		int64_t batchnum = 1;
		if (programOptions.count("batch"))
		{
	
			batchnum = programOptions["batch"].as<int>();
		
		}
		subtrain_tensor->setBatchNum(batchnum);
		double lr= programOptions["lr"].as < double >();
		int optimType = programOptions["optim"].as < int >();
		if (isFNN)
		{
			std::cout << "Funtional neural network analysis is runing" << std::endl;
			LOG(INFO) << "Funtional neural network analysis is runing";
			//		std::cout << "========================" << std::endl;
			std::shared_ptr<FNN> f = std::make_shared<FNN>(dims, lambda);
			torch::Tensor test_loss;
			if (!basis)
			{

				f->build<Haar>(sinlgeknot, ncovs);
			}
			else
			{
				f->build<SplineBasis>(sinlgeknot, ncovs);
			}
			switch (loss)
			{
			case 0:
				if (optimType == 0)
				{
					test_loss = training<torch::optim::Adam, FNN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else if (optimType == 1)
				{
					test_loss = training<torch::optim::SGD, FNN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else
				{
					throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
				}
				//test_loss = training<torch::optim::Adam, FNN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, epoch);
				break;
			case 1:
				if (optimType == 0)
				{
					test_loss = training<torch::optim::Adam, FNN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else if (optimType == 1)
				{
					test_loss = training<torch::optim::SGD, FNN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else
				{
					throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
				}
				
				//test_loss=Testing<FNN, torch::nn::BCELoss>(f, test_tensor);
				break;
			case 2:
				if (optimType == 0)
				{
					test_loss = training<torch::optim::Adam, FNN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else if (optimType == 1)
				{
					test_loss = training<torch::optim::SGD, FNN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else
				{
					throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
				}
				//test_loss=Testing<FNN, torch::nn::CrossEntropyLoss>(f, test_tensor);
				break;
			}
			//		for (const auto& p : f->parameters()) 
			//		{
			//			std::cout << p << std::endl;
			//		}
			if (programOptions.count("save"))
			{
				std::string saveNet = programOptions["save"].as < std::string >();
				//	string model_path = "model.pt";
				torch::serialize::OutputArchive output_archive;
				f->save(output_archive);
				output_archive.save_to(saveNet);
			}
			std::cout << "Training completed." << std::endl;
			std::cout << "========================" << std::endl;
			std::stringstream ss;
			ss << "epoch: " << f->epoch << "\tTraining: loss: " << test_loss.item<double>();
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str() << std::endl;

		///////////////////////////////////////////////////////////////////////
			/// evaluate the testing dataset
			if (test_tensor->nind != 0)
			{

				std::shared_ptr<Evaluate> test = nullptr;
				if (test_tensor->isBalanced)
				{
					torch::Tensor pred_test = f->forward(test_tensor);
					test = std::make_shared< Evaluate>(test_tensor->getY(), pred_test, test_tensor->dataType);
					//	Evaluate Total(data->getY(), pred_test, data->dataType);
				}
				else
				{
					if (test_tensor->dataType == 0)
					{
						
						torch::Tensor loss=torch::zeros(1);
						for (size_t i = 0; i < test_tensor->nind; i++)
						{
						
							auto sample = test_tensor->getSample(i);
							torch::Tensor pred_test = f->forward(sample);
							loss += torch::mse_loss(pred_test, sample->getY());
						}
						loss=loss/ test_tensor->nind;
						test = std::make_shared< Evaluate>();
						test->setMSE(loss.item<double>());
					}
				}
				
				prediction_error.push_back(test);
			//	prediction_error.push_back(test->getAUC());
			}
			else
			{
	//			prediction_error.push_back(-9);
				prediction_error.push_back(nullptr);
			}
			////////////////////////////////////////////////////////////
			// evaluate the total dataset
			std::shared_ptr<Evaluate> Total = nullptr;
			if (data->isBalanced)
			{
				torch::Tensor pred_total = f->forward(data);
				Total = std::make_shared< Evaluate>(data->getY(), pred_total, data->dataType);
				prediction_error.push_back(Total);
				//	Evaluate Total(data->getY(), pred_test, data->dataType);
			}
			else
			{
				if (data->dataType == 0)
				{
					
					torch::Tensor loss = torch::zeros(1);
					for (size_t i = 0; i < data->nind; i++)
					{
						auto sample = data->getSample(i);
						torch::Tensor pred_test = f->forward(sample);
						loss += torch::mse_loss(pred_test, sample->getY());
						
					}
					loss = loss / test_tensor->nind;
					Total = std::make_shared< Evaluate>();
					Total->setMSE(loss.item<double>());
					prediction_error.push_back(Total);

				}
				else
				{
					prediction_error.push_back(nullptr);
				}
			}
		

			
		//	prediction_error.push_back(Total->getAUC());
		}
		else
		{
			std::cout << "Neural network analysis is runing" << std::endl;
			LOG(INFO) << "Neural network analysis is runing";
			//	std::cout << "========================" << std::endl;
			std::shared_ptr<NN> f = std::make_shared<NN>(dims, lambda);
			f->build(ncovs);
			torch::Tensor train_loss;
			switch (loss)
			{
			case 0:
		//		train_loss = training<torch::optim::Adam, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				if (optimType == 0)
				{
					train_loss = training<torch::optim::Adam, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else if (optimType == 1)
				{
					train_loss = training<torch::optim::SGD, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else
				{
					throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
				}
				//	test_loss=Testing<NN, torch::nn::MSELoss>(f, test_tensor);
				break;
			case 1:
			//	train_loss = training<torch::optim::Adam, NN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				if (optimType == 0)
				{
					train_loss = training<torch::optim::Adam, NN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else if (optimType == 1)
				{
					train_loss = training<torch::optim::SGD, NN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else
				{
					throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
				}
				//test_loss=Testing<NN, torch::nn::BCELoss>(f, test_tensor);
				break;
			case 2:
			//	train_loss = training<torch::optim::Adam, NN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				if (optimType == 0)
				{
					train_loss = training<torch::optim::Adam, NN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else if (optimType == 1)
				{
					train_loss = training<torch::optim::SGD, NN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
				}
				else
				{
					throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
				}
				//test_loss=Testing<NN, torch::nn::CrossEntropyLoss>(f, test_tensor);
				break;
			}
		//	for (const auto& p : f->parameters())
		//	{
		//		std::cout << p << std::endl;
		//	}
			std::cout << "Training completed." << std::endl;
			std::cout << "========================" << std::endl;
			std::stringstream ss;
			ss <<  "epoch: " << f->epoch << "\tTraining: loss: " << train_loss.item<double>();
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str() << std::endl;



			if (test_tensor->getY().sizes()[0] != 0)
			{
				torch::Tensor pred_test = f->forward(test_tensor);
				std::shared_ptr<Evaluate> test = std::make_shared<Evaluate>(test_tensor->getY(), pred_test, test_tensor->dataType);
				prediction_error.push_back(test);
	//			prediction_error.push_back(test.getAUC());
			}
			else
			{
				prediction_error.push_back(nullptr);
				//prediction_error.push_back(-9);
			}

			
			torch::Tensor pred_total = f->forward(data);
			std::shared_ptr<Evaluate> Total = std::make_shared<Evaluate>(data->getY(), pred_total, data->dataType);
		
			prediction_error.push_back(Total);
		//	prediction_error.push_back(Total.getAUC());
		}
	}

	
	return prediction_error;
}

void readKNNParameter(boost::program_options::variables_map programOptions, MinqueOptions& minque)
{
	if (programOptions.count("iterate"))
	{
		minque.iterate = programOptions["iterate"].as<int>();
	}
	if (programOptions.count("tolerance"))
	{
		minque.tolerance = programOptions["tolerance"].as<float>();
	}
	if (programOptions.count("pseudo"))
	{
		minque.allowPseudoInverse = programOptions["pseudo"].as<bool>();
	}
	if (programOptions.count("inverse"))
	{
		std::string Decomposition = programOptions["inverse"].as < std::string >();
		boost::to_lower(Decomposition);
		if (isNum(Decomposition))
		{
			int number;
			std::istringstream iss(Decomposition);
			iss >> number;
			switch (number)
			{
			case 0:
				minque.MatrixDecomposition = 0;
				break;
			case 1:
				minque.MatrixDecomposition = 1;
				break;
			case 2:
				minque.MatrixDecomposition = 2;
				break;
			case 3:
				minque.MatrixDecomposition = 3;
				break;
			default:
			{
				throw std::string("The parameter \"--inverse " + Decomposition + "\" is not correct, please check it. More detail --help");
				break;
			}
			}
		}
		else
		{
			if (Decomposition == "cholesky")
			{
				minque.MatrixDecomposition = 0;
			}
			else if (Decomposition == "lu")
			{
				minque.MatrixDecomposition = 1;
			}
			else if (Decomposition == "qr")
			{
				minque.MatrixDecomposition = 2;

			}
			else if (Decomposition == "svd")
			{
				minque.MatrixDecomposition = 3;
			}
			else
			{
				throw  std::string("The parameter \"--inverse " + Decomposition + "\" is not correct, please check it. More detail --help");
			}
		}


	}
	if (programOptions.count("ginverse"))
	{
		std::string Decomposition = programOptions["ginverse"].as < std::string >();
		boost::to_lower(Decomposition);
		if (isNum(Decomposition))
		{
			int number;
			std::istringstream iss(Decomposition);
			iss >> number;
			switch (number)
			{
			case 2:
				minque.altMatrixDecomposition = 2;
				break;
			case 3:
				minque.altMatrixDecomposition = 3;
				break;
			default:
				throw std::string("The argument \"--ginverse " + Decomposition + "\" is not correct, please check it. More detail --help");
				break;
			}
		}
		else
		{
			if (Decomposition == "qr")
			{
				minque.altMatrixDecomposition = 2;

			}
			else if (Decomposition == "svd")
			{
				minque.altMatrixDecomposition = 3;
			}
			else
			{
				throw  std::string("The parameter \"--ginverse " + Decomposition + "\" is not correct, please check it. More detail --help");
			}
		}
	}
}
