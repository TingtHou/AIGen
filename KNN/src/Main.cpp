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
#include "../include/Bootstrap.h"
#include "../include/Random.h"
#ifndef CPU  
	#include "../include/cuMINQUE0.h"
	#include "../include/cuimnq.h" 
#endif



INITIALIZE_EASYLOGGINGPP

void readAlgrithomParameter(boost::program_options::variables_map programOptions, MinqueOptions& minque);

void ReadData(boost::program_options::variables_map programOptions, DataManager &dm);

int MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager& dm, MinqueOptions& minopt, Eigen::VectorXf& VarComp, Eigen::VectorXf& predict);

void BatchMINQUE1(MinqueOptions& minque, std::vector<Eigen::MatrixXf *>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, int nsplit, int seed, int nthread, bool isecho);

void cMINQUE1(MinqueOptions& minque, std::vector<Eigen::MatrixXf *>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho);


void BatchMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf *>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, int nsplit, int seed, int nthread);

void cMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf*>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs);


void Bootstraping(boost::program_options::variables_map programOptions, MinqueOptions& minopt, DataManager& dm, int times, Random& seed,
	std::vector<Eigen::VectorXf> &Vars_BP, std::vector<Eigen::VectorXf> &Predicts_BP, std::vector<int> &iterateTimes_BP);

#ifndef CPU  
void cudaMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs);
void cudaMINQUE1(MinqueOptions& minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho);
#endif

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
	MinqueOptions minopt;

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
	Random rd(0);
	if (programOptions.count("seed"))
	{
		float seed = programOptions["seed"].as<float>();
		rd.setseed(seed);
	}
//////init logger
	el::Configurations conf;
	conf.setToDefault();
	conf.parseFromText(("*GLOBAL:\n  FILENAME = " + logfile+"\n  To_Standard_Output = false"));
	el::Loggers::reconfigureLogger("default", conf);
	std::cout << opt.print() << std::endl;
	LOG(INFO) << opt.print();
	readAlgrithomParameter(programOptions, minopt);
	DataManager dm;
	ReadData(programOptions, dm);
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
		throw std::string("Error: make-bin function is under maintenance, and will back soon.");
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
	if (dm.getPhenotype().fid_iid.size() != 0)
	{
		dm.match();
		if (!programOptions.count("skip"))
		{
			Eigen::VectorXf VarComp;
			Eigen::VectorXf predict;
			int iterateTimes=MINQUEAnalysis(programOptions, dm, minopt, VarComp,predict);

			//////////////////////////////////////////////////////////
		

			if (programOptions.count("bootstrap"))
			{
				int times = programOptions["bootstrap"].as < int >();
				std::vector<Eigen::VectorXf> Vars_BP;
				std::vector<Eigen::VectorXf> Predicts_BP;
				std::vector<int> iterateTimes_BP;
				std::cout << "---Bootstraping--" << std::endl;
				LOG(INFO) << "---Bootstraping----";
				Bootstraping(programOptions, minopt, dm, times, rd, Vars_BP, Predicts_BP, iterateTimes_BP);
				ofstream bp;
				bp.open("bootstrap.txt", ios::out);
				for (int i = 0; i < times; i++)
				{
					float VG = 0;
					float VP = 0;
					int j = 0;
					for (j; j < Vars_BP[i].size() - 1; j++)
					{
						//	std::cout << VarComp.at(i) << " ";
						VG += Vars_BP[i](j);
					}
					VP = VG;
					VP+= Vars_BP[i](j);
					bp << "# " << i <<"\t"<< Vars_BP[i].transpose() << "\t" << VG / VP << std::endl;
				}
				bp.close();
			}


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
				if (dm.getPhenotype().isbinary)
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
	
}

int main(int argc, const char *const argv[])
{
	/////////////////////////////////////////////////////////////////////
	std::cout<<"\n"
		"@----------------------------------------------------------@\n"
		"|        KNN       |     v alpha 0.1.1  |    "<<__DATE__<<"   |\n"
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
	dm.readCovariates(qcovariatefile, dcovariatefile);
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
	}
	
}

int MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager &dm, MinqueOptions &minopt, Eigen::VectorXf &VarComp, Eigen::VectorXf &predict)
{
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
		for (int i = 0; i < Kmatrices->size(); i++)
		{
			Kernels.push_back(&(Kmatrices->at(i)));
		}
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
#else
		throw std::string("This is a CPU program. Please use GPU version.");
#endif
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
					cMINQUE0(minopt, Kernels, phe, Covs.Covariates, VarComp, fix);
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
		Prediction pred(phe.Phenotype, Kernels, VarComp, Covs.Covariates, fix, phe.isbinary,mode);
		predict.resize(2);
		
		std::stringstream ss;
		if (phe.isbinary)
		{
			predict[0] = pred.getMSE();
			predict[1] = pred.getAUC();
			//ss << "misclassification error:\t" << pred.getMSE() << std::endl << "AUC:\t" << pred.getAUC() << std::endl;
		}
		else
		{
			predict[0] = pred.getMSE();
			predict[1] = pred.getCor();
			//ss << "MSE:\t" << pred.getMSE() << std::endl << "Correlation:\t" << pred.getCor() << std::endl;
		}
//		std::cout << ss.str();
//		LOG(INFO) << ss.str();
//		out<<  ss.str();
	}
	return iterateTimes;
//	out.close();
}
int MINQUEAnalysis(boost::program_options::variables_map programOptions, Bootstrap& dm, MinqueOptions& minopt, Eigen::VectorXf& VarComp, Eigen::VectorXf& predict)
{
	bool GPU = false;
	if (programOptions.count("GPU"))
	{
		GPU = true;
	}
	PhenoData phe = dm.getPhenotype();
	std::vector<KernelData>* kd;
	kd = dm.GetKernel();
	CovData Covs = dm.GetCovariates();
	VarComp = dm.GetWeights();
	// initialize the variance components vector with pre-set weights
	Eigen::VectorXf fix(Covs.npar);
	fix.setZero();
	fix[0] = -999;
	float iterateTimes = 0;
	bool isecho = false;
	std::vector<Eigen::MatrixXf*> Kernels;
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
		for (int i = 0; i < Kmatrices->size(); i++)
		{
			Kernels.push_back(&(Kmatrices->at(i)));
		}
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
#else
		throw std::string("This is a CPU program. Please use GPU version.");
#endif
	}
	else
	{
		if (programOptions.count("batch"))
		{
			//		throw std::string("Error: batch function is under maintenance, and will back soon.");

			int nthread = 10;
			int nsplit = programOptions["batch"].as<int>();
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
					std::vector<Eigen::MatrixXf*> Kernels_imq;
					int nind = phe.fid_iid.size();
					for (int i = 0; i < Kernels.size(); i++)
					{
						Kernels_imq[i] = new Eigen::MatrixXf(nind,nind);
						*Kernels_imq[i] = *Kernels[i];
					}
					cMINQUE0(minopt, Kernels_imq, phe, Covs.Covariates, VarComp, fix);
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
		Prediction pred(phe.Phenotype, Kernels, VarComp, Covs.Covariates, fix, phe.isbinary, mode);
		predict.resize(2);

		std::stringstream ss;
		if (phe.isbinary)
		{
			predict[0] = pred.getMSE();
			predict[1] = pred.getAUC();
			//ss << "misclassification error:\t" << pred.getMSE() << std::endl << "AUC:\t" << pred.getAUC() << std::endl;
		}
		else
		{
			predict[0] = pred.getMSE();
			predict[1] = pred.getCor();
			//ss << "MSE:\t" << pred.getMSE() << std::endl << "Correlation:\t" << pred.getCor() << std::endl;
		}
		//		std::cout << ss.str();
		//		LOG(INFO) << ss.str();
		//		out<<  ss.str();
	}
	return iterateTimes;
	//	out.close();
}

void BatchMINQUE1(MinqueOptions &minque, std::vector<Eigen::MatrixXf *>& Kernels, PhenoData & phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float & iterateTimes, int nsplit,int seed, int nthread, bool isecho)
{
    int nkernel = Kernels.size();
	bool nofix = coefs[0] == -999 ? true : false;
	Batch b = Batch(Kernels, phe.Phenotype, Covs, nsplit, seed, true);
	b.start();
	std::vector<std::vector<Eigen::MatrixXf>> KernelsBatch;
	std::vector<Eigen::VectorXf> PheBatch;
	std::vector<Eigen::MatrixXf> CovBatch;
	b.GetBatchKernels(KernelsBatch);
	b.GetBatchPhe(PheBatch);
	b.GetBatchCov(CovBatch);
	std::vector<float> time(KernelsBatch.size());
	std::vector<std::vector<float>> varsBatch(KernelsBatch.size());
	std::vector<std::vector<float>> fixsBatch(KernelsBatch.size());
	std::cout << "starting CPU MINQUE" << std::endl;
	omp_set_num_threads(nthread);
	#pragma omp parallel for
	for (int i = 0; i < KernelsBatch.size(); i++)
	{
		
		printf("Starting CPU-based analysis on thread %d\n", i);
		LOG(INFO) << "Starting CPU-based analysis on thread " << i;
		imnq varest;
		varest.isEcho(isecho);
		varest.setThreadId(i);
		varest.setOptions(minque);
		varest.importY(PheBatch[i]);
		varest.pushback_X(CovBatch[i],false);
		if (variances.size() != 0)
		{
			varest.pushback_W(variances);
		}
		Eigen::MatrixXf e(PheBatch[i].size(), PheBatch[i].size());
		e.setIdentity();
		for (int j = 0; j < KernelsBatch[i].size(); j++)
		{
			varest.pushback_Vi(&KernelsBatch[i][j]);
		}
		varest.pushback_Vi(&e);
		try
		{
			varest.estimateVCs();
			varsBatch[i].resize(varest.getvcs().size());	
			Eigen::VectorXf::Map(&varsBatch[i][0], varest.getvcs().size()) = varest.getvcs();
			if (!nofix)
			{
				varest.estimateFix();
				fixsBatch[i].resize(varest.getfix().size());
				Eigen::VectorXf::Map(&fixsBatch[i][0], varest.getfix().size()) = varest.getfix();
			}
			printf("The thread %d is finished\n", i);
			LOG(INFO) << "The thread " << i << " is finished";
		}
		catch (const std::exception& err)
		{
			stringstream ss;
			ss << "[Warning]: The thread " << i << " is interrupt, because " << err.what();
			printf("%s\n", ss.str().c_str());
			LOG(WARNING)<< ss.str();
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;	
			if (!nofix)
			{
				fixsBatch[i].resize(1);
				fixsBatch[i][0] = -999;
			}
		}
	}
	for (int i=0;i<varsBatch.size();i++)
	{
		stringstream ss;
		ss << "Thread ID: " << i<<"\t";
		for (int j=0;j<varsBatch[i].size();j++)
		{
			ss <<  varsBatch[i][j] << "\t";
		}
		LOG(INFO) << ss.str();
	}
	auto it = varsBatch.begin();
	while (it != varsBatch.end())
	{
		auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
		if (itzero != (*it).end())
		{
			it = varsBatch.erase(it);
		}
		else
		{
			++it;
		}
	}
	variances.resize(nkernel + 1);
	for (int i = 0; i < nkernel + 1; i++)
	{
		float sum = 0;
		for (int j = 0; j < varsBatch.size(); j++)
		{
			sum += varsBatch[j][i];
		}
		variances[i]=sum / float(varsBatch.size());
	}

	if (!nofix)
	{
		auto it = fixsBatch.begin();
		while (it != fixsBatch.end())
		{
			auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
			if (itzero != (*it).end())
			{
				it = fixsBatch.erase(it);
			}
			else
			{
				++it;
			}
		}
		coefs.resize(fixsBatch[0].size());
		for (int i = 0; i < coefs.size(); i++)
		{
			float sum = 0;
			for (int j = 0; j < fixsBatch.size(); j++)
			{
				sum += fixsBatch[j][i];
			}
			coefs[i] = sum / float(fixsBatch.size());
		}
	}
	
	iterateTimes = accumulate(time.begin(), time.end(), 0.0) / time.size(); ;
}

void cMINQUE1(MinqueOptions & minque, std::vector<Eigen::MatrixXf *>& Kernels, PhenoData & phe, Eigen::MatrixXf & Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs,float & iterateTimes,  bool isecho)
{
	imnq varest;
	varest.setOptions(minque);
	varest.isEcho(isecho);
	varest.importY(phe.Phenotype);
	varest.pushback_X(Covs, false);
	for (int i = 0; i < Kernels.size(); i++)
	{
		varest.pushback_Vi(Kernels[i]);
	}
	Eigen::MatrixXf e(phe.fid_iid.size(), phe.fid_iid.size());
	e.setIdentity();
	varest.pushback_Vi(&e);
	if (variances.size() != 0)
	{
		varest.pushback_W(variances);
	}
	std::cout << "starting CPU MINQUE(1) " << std::endl;
	varest.estimateVCs();
	variances = varest.getvcs();
	iterateTimes = varest.getIterateTimes();
	if (coefs[0] != -999)
	{
		varest.estimateFix();
		coefs = varest.getfix();
	}
	
}



void BatchMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf *>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs,  int nsplit, int seed, int nthread)
{
	int nkernel = Kernels.size();
	bool nofix = coefs[0] == -999 ? true : false;
	Batch b = Batch(Kernels, phe.Phenotype, Covs, nsplit, seed, true);
	b.start();
	std::vector<std::vector<Eigen::MatrixXf>> KernelsBatch;
	std::vector<Eigen::VectorXf> PheBatch;
	std::vector<Eigen::MatrixXf> CovBatch;
	b.GetBatchKernels(KernelsBatch);
	b.GetBatchPhe(PheBatch);
	b.GetBatchCov(CovBatch);
	std::vector<float> time(KernelsBatch.size());
	std::vector<std::vector<float>> varsBatch(KernelsBatch.size());
	std::vector<std::vector<float>> fixsBatch(KernelsBatch.size());
	std::cout << "starting CPU MINQUE" << std::endl;
	omp_set_num_threads(nthread);
	#pragma omp parallel for
	for (int i = 0; i < KernelsBatch.size(); i++)
	{
		clock_t t1 = clock();
		printf("Starting CPU-based analysis on thread %d\n", i);
		LOG(INFO) << "Starting CPU-based analysis on thread " << i;
		MINQUE0 varest(minque.MatrixDecomposition, minque.altMatrixDecomposition, minque.allowPseudoInverse);
		varest.setThreadId(i);
		varest.importY(PheBatch[i]);
		varest.pushback_X(CovBatch[i], false);
		Eigen::MatrixXf e(PheBatch[i].size(), PheBatch[i].size());
		e.setIdentity();
		for (int j = 0; j < KernelsBatch[i].size(); j++)
		{
			varest.pushback_Vi(&KernelsBatch[i][j]);
		}
		varest.pushback_Vi(&e);
		try
		{
			varest.estimateVCs();
			varsBatch[i].resize(varest.getvcs().size());
			Eigen::VectorXf::Map(&varsBatch[i][0], varest.getvcs().size()) = varest.getvcs();	
			if (!nofix)
			{
				varest.estimateFix();
				fixsBatch[i].resize(varest.getfix().size());
				Eigen::VectorXf::Map(&fixsBatch[i][0], varest.getfix().size()) = varest.getfix();
			}
			printf("The thread %d is finished\n", i);
			LOG(INFO) << "The thread " << i << " is finished";
		}
		catch (const std::exception & err)
		{
			stringstream ss;
			ss << "[Warning]: The thread " << i << " is interrupt, because " << err.what();
			printf("%s\n", ss.str().c_str());
			LOG(WARNING) << ss.str();
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
			if (!nofix)
			{
				fixsBatch[i].resize(1);
				fixsBatch[i][0] = -999;
			}
		}
	}
	for (int i = 0; i < varsBatch.size(); i++)
	{
		stringstream ss;
		ss << "Thread ID: " << i << "\t";
		for (int j = 0; j < varsBatch[i].size(); j++)
		{
			ss << varsBatch[i][j] << "\t";
		}
		LOG(INFO) << ss.str();
	}
	auto it = varsBatch.begin();
	while (it != varsBatch.end())
	{
		auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
		if (itzero != (*it).end())
		{
			it = varsBatch.erase(it);
		}
		else
		{
			++it;
		}
	}
	
	variances.resize(nkernel + 1);
	for (int i = 0; i < nkernel + 1; i++)
	{
		float sum = 0;
		for (int j = 0; j < varsBatch.size(); j++)
		{
			sum += varsBatch[j][i];
		}
		variances[i] = sum / float(varsBatch.size());
	}
	if (!nofix)
	{
		auto it = fixsBatch.begin();
		while (it != fixsBatch.end())
		{
			auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
			if (itzero != (*it).end())
			{
				it = fixsBatch.erase(it);
			}
			else
			{
				++it;
			}
		}
		coefs.resize(fixsBatch[0].size());
		for (int i = 0; i < coefs.size(); i++)
		{
			float sum = 0;
			for (int j = 0; j < fixsBatch.size(); j++)
			{
				sum += fixsBatch[j][i];
			}
			coefs[i] = sum / float(fixsBatch.size());
		}
	}
	
}

void cMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf *> &Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs)
{
	MINQUE0 varest(minque.MatrixDecomposition,minque.altMatrixDecomposition, minque.allowPseudoInverse);
	varest.importY(phe.Phenotype);
	varest.pushback_X(Covs, false);
	for (int i = 0; i < Kernels.size(); i++)
	{
		varest.pushback_Vi(Kernels[i]);
	}
	Eigen::MatrixXf e(phe.Phenotype.size(), phe.Phenotype.size());
	e.setIdentity();
	varest.pushback_Vi(&e);
	std::cout << "starting CPU MINQUE(0) " << std::endl;
	LOG(INFO) << "starting CPU MINQUE(0) ";
	varest.estimateVCs();
	variances = varest.getvcs();
	if (coefs[0]!=-999)
	{
		varest.estimateFix();
		coefs = varest.getfix();
	}
	std::stringstream ss;
	ss << std::fixed << "Thread ID: 0"<< std::setprecision(3) << "\tIt: " << 0 << "\t" << varest.getvcs().transpose();
	printf("%s\n", ss.str().c_str());
	LOG(INFO) << ss.str();
}

void Bootstraping(boost::program_options::variables_map programOptions, MinqueOptions& minopt, DataManager& dm, int times, Random &seed, 
					std::vector<Eigen::VectorXf> &Vars_BP,std::vector<Eigen::VectorXf> &Predicts_BP,std::vector<int> &iterateTimes_BP)
{

	Bootstrap bo(dm, &seed);
	for (int i = 0; i < times; i++)
	{
		std::cout << "Bootstrap # " << i << std::endl;
		LOG(INFO)<< "Bootstrap # " << i ;
		bo.generate();
		Eigen::VectorXf VarComp;
		Eigen::VectorXf predict;
		int iterateTimes = MINQUEAnalysis(programOptions, bo, minopt, VarComp, predict);
		Vars_BP.push_back(VarComp);
		Predicts_BP.push_back(predict);
		iterateTimes_BP.push_back(iterateTimes);
	}

}

#ifndef CPU  
void cudaMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs)
{

	cuMINQUE0 cuvarest(minque.MatrixDecomposition, minque.altMatrixDecomposition, minque.allowPseudoInverse);
	cuvarest.importY(phe.Phenotype);
	cuvarest.pushback_X(Covs, false);
	for (int i = 0; i < Kernels.size(); i++)
	{
		cuvarest.pushback_Vi(Kernels[i]);
	}
	Eigen::MatrixXf e(phe.fid_iid.size(), phe.fid_iid.size());
	e.setIdentity();
	cuvarest.pushback_Vi(e);
	std::cout << "starting GPU MINQUE " << std::endl;
	cuvarest.init();
	cuvarest.estimateVCs();
	variances = cuvarest.getvcs();
	if (coefs[0] != -999)
	{
		cuvarest.estimateFix();
		coefs = cuvarest.getfix();
	}
	std::stringstream ss;
	ss << std::fixed << "Thread ID: 0" << std::setprecision(3) << "\tIt: " << 0 << "\t" << cuvarest.getvcs().transpose();
	printf("%s\n", ss.str().c_str());
	LOG(INFO) << ss.str();
}

void cudaMINQUE1(MinqueOptions& minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho)
{
	cuimnq cuvarest;
	cuvarest.setOptions(minque);
	cuvarest.isEcho(isecho);
	cuvarest.importY(phe.Phenotype);
	cuvarest.pushback_X(Covs, false);
	if (variances.size() != 0)
	{
		cuvarest.pushback_W(variances);
	}
	for (int i = 0; i < Kernels.size(); i++)
	{
		cuvarest.pushback_Vi(Kernels[i]);
	}

	Eigen::MatrixXf e(phe.fid_iid.size(), phe.fid_iid.size());
	e.setIdentity();
	cuvarest.pushback_Vi(e);
	std::cout << "starting GPU MINQUE " << std::endl;
	cuvarest.estimateVCs();
	variances = cuvarest.getvcs();
	iterateTimes = cuvarest.getIterateTimes();
	if (coefs[0] != -999)
	{
		cuvarest.estimateFix();
		coefs = cuvarest.getfix();
	}

}
#endif



void readAlgrithomParameter(boost::program_options::variables_map programOptions, MinqueOptions& minque)
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