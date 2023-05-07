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

int MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager& dm, Eigen::VectorXf& VarComp, Eigen::VectorXf& fix, Eigen::VectorXf& predict);
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
	std::string result= "result.txt";
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
		std::ofstream out;
		out.open(result, std::ios::out);
		LOG(INFO) << "---Result----";
		std::cout << "---Result----" << std::endl;
		std::stringstream ss;
	
		if (programOptions.count("load"))
		{
			LOG(INFO) << "---full dataset----";
			ss << "---full dataset----" << std::endl;
			if (error[0]->dataType != 0)
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
			LOG(INFO) << "---Training dataset----";
			ss << "---Training dataset----" << std::endl;
			if (error[1]->dataType != 0)
			{
				ss << "misclassification error:\t" << error[1]->getMSE() << std::endl << "AUC:\t" << error[1]->getAUC() << std::endl;
			}
			else
			{
				ss << "MSE:\t" << error[1]->getMSE() << std::endl << "Correlation:\t" << error[1]->getCor().transpose() << std::endl;
			}


			if (error[0] != nullptr)
			{
				LOG(INFO) << "---Testing dataset----";
				ss << "---Testing dataset----" << std::endl;
				if (error[0]->dataType != 0)
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
			
			std::cout << ss.str();
			LOG(INFO) << ss.str();
			out << ss.str();
		}
	
		out.close();
	}

	/////////////////////////////////////////////////////////
	///KNN implement
	////////////////////////////////////////////////////////////////
	std::vector<std::shared_ptr<KernelData>> kernelList; 
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
			std::ifstream infile;
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
		for (int i = 0; i < dm.GetKernel().size(); i++)
		{
			if (i > 1)
			{
				outname += i;
			}
			KernelWriter kw(dm.GetKernel().at(i));

			kw.setprecision(isfloat);
			kw.write(outname);
		}
		
	}
	//if the phenotype is inputed, the estimation will be started.
	if (programOptions.count("KNN"))
	{
			Eigen::VectorXf VarComp;
			Eigen::VectorXf predict;
			Eigen::VectorXf fix;
			int iterateTimes=MINQUEAnalysis(programOptions, dm, VarComp, fix, predict);

			std::ofstream out;
			out.open(result, std::ios::out);
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
				double h_l = h_O * K * (1 - K) /( z * z);
				ss << "liability h\t" << h_l<<std::endl;
			}
			ss << "Iterate Times:\t" << iterateTimes;
			out << ss.str() << std::endl;
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str();
			if (programOptions.count("fix"))
			{
				std::vector<std::string> names = dm.GetCovariates().names;
				std::stringstream ss;
				ss << "------------------------------------" << std::endl;
				ss<< "Cov\tEffect" << std::endl;
				for (size_t i = 0; i < names.size(); i++)
				{
					ss << names[i] << "\t" << fix[i] << "\n";
				}
				LOG(INFO) << ss.str();
				std::cout<<ss.str() << std::endl;
				out << ss.str() << std::endl;
			}
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
	int status=0;
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	try
	{
		omp_set_nested(true);
		omp_set_dynamic(false);
		mkl_set_dynamic(false);
		TryMain(argc, argv);
	}
	catch (std::string &e)
	{
		std::cout << e << std::endl;
		LOG(ERROR) << e;
		status = 1;
	}
	catch (const std::exception &err)
	{
		std::cout << err.what() << std::endl;
		LOG(ERROR) << err.what();
		status = 1;
	}
	catch (...)
	{
		std::cout << "Unknown Error" << std::endl;
		LOG(ERROR) << "Unknown Error";
		status=1;
	}
	//get now time
	boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
	std::cout << "\n\nAnalysis finished: " << timeLocal << std::endl;
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::ratio<1, 1>> duration_s(end - start);
	//std::cout << "Total elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	std::cout << "Computational time: " << duration_s.count() <<" second(s)."<< std::endl;
	return status;
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

int MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager &dm, Eigen::VectorXf &VarComp, Eigen::VectorXf & fix,Eigen::VectorXf &predict)
{
	MinqueOptions minopt;
	readKNNParameter(programOptions, minopt);
	bool GPU = false;
	float ratio = programOptions["ratio"].as < float >();
	float seed = programOptions["seed"].as < float >();
	bool NoE = programOptions.count("NoE");
	if (programOptions.count("GPU"))
	{
		GPU = true;
	}
	if (abs(ratio - 1) > 1e-6)
	{
		dm.shuffle(seed,ratio);
	}
	PhenoData phe=dm.getPhenotype();
	std::vector<std::shared_ptr<KernelData>> kd;
	kd= dm.GetKernel();
	CovData Covs=dm.GetCovariates();
	VarComp=dm.GetWeights();
	long long train_size = phe.nind * ratio;
	long long test_size = phe.nind - train_size;
	// initialize the variance components vector with pre-set weights
	//Eigen::VectorXf fix(Covs.npar);
	fix.resize(Covs.npar);
	fix.setZero();
	float iterateTimes = 0;
	bool isecho = false;
	std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels;
	if (programOptions.count("echo"))
	{
		isecho = programOptions["echo"].as<bool>();
	}
	if (programOptions.count("vcs"))
	{
		std::string VCs_str = programOptions["vcs"].as < std::string >();
		std::cout << "Loading initial values of variance components from command line." << std::endl;
		std::vector<std::string> strVec;
		boost::algorithm::split(strVec, VCs_str, boost::algorithm::is_any_of(","), boost::token_compress_on);
		VarComp.resize(strVec.size());
		for (int i = 0; i < strVec.size(); i++)
		{
			VarComp[i] = std::stof(strVec.at(i));
		}
	}

	if (programOptions.count("load"))
	{
		std::string saveNet = programOptions["load"].as < std::string >();
		
		std::ifstream Binifstream(saveNet, std::ios::binary);
		if (!Binifstream.is_open())
		{
			Binifstream.close();
			throw std::string("Error: can not open the file [" + saveNet + "] to read.");
		}
		std::cout << "Loading initial values of variance components from binary file [" << saveNet << "]" << std::endl;
		Binifstream.seekg(0, std::ios::end);
		std::streampos fileSize = Binifstream.tellg();
		unsigned long long bytesize = sizeof(float);
		Binifstream.seekg(0, std::ios::beg);
		char* f_buf = new char[fileSize];
		//std::string f_buf;
		if (!(Binifstream.read(f_buf, fileSize))) // read up to the size of the buffer
		{
			if (!Binifstream.eof()) // end of file is an expected condition here and not worth 
							   // clearing. What else are you going to read?
			{
				throw std::string("Error: the size of the [" + saveNet + "] file is incomplete? EOF is missing.");
			}
			else
			{
				throw std::string("Error: Unknow error when reading [" + saveNet + "].");
			}
		}
		VarComp.resize(kd.size() + 1);
		for (long long k = 0; k < kd.size()+1; k++)
		{
		
			char* str2 = new char[bytesize];
			unsigned long long pointer = k * bytesize;
			memcpy(str2, &f_buf[pointer], bytesize);
		    VarComp[k]= *(float*)str2;
			delete str2;
		}
	}
	std::cout << "Intital values of variance components are " << VarComp.transpose() << std::endl;
	if (programOptions.count("alphaKNN"))
	{
	//	throw std::string("Error: alphaKNN function is under maintenance, and will back soon.");
		
		int alpha = programOptions["alphaKNN"].as<int>();
		KernelExpansion ks(kd, alpha);
	//	Kernels = ks.GetExtendMatrix();
		Kernels = ks.GetExtendMatrix();
		/*
		Kernels.resize(Kmatrices.size());
		for (int i = 0; i < Kmatrices.size(); i++)
		{
			Kernels[i] = std::make_shared<Eigen::MatrixXf> (Kmatrices->at(i).rows(), Kmatrices->at(i).cols());
		   //	Kernels.push_back(&(Kmatrices->at(i)));
			*Kernels[i] = Kmatrices->at(i);
		}
		*/
		kd.clear();
	}
	else
	{
		
		for (int i = 0; i < kd.size(); i++)
		{
			Kernels.push_back(kd[i]->kernelMatrix);
		}
	}
	if (VarComp.size() != 0)
	{
		if (VarComp.size() != (Kernels.size() + 1))
		{
			throw std::string("Error: The size of pre-specified weight vector is not equal to the number of variance components.");
		}
	}
	std::vector<std::shared_ptr<Eigen::MatrixXf>> TrainingSet=Kernels;
	Eigen::VectorXf phe_train(phe.nind);
	Eigen::VectorXf phe_test(test_size);
	Eigen::MatrixXf Cov_train = Covs.Covariates;
	Eigen::MatrixXf Cov_test(test_size, Covs.npar);
	if (phe.Phenotype.cols() == 1)
	{
		phe_train = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
	}
	else
	{
		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
	}
	if (abs(ratio-1)>1e-6)
	{
		std::stringstream ss;
		ss << "Split the dataset into two sub-dataset, training, testing";
		TrainingSet.clear();
		TrainingSet.resize(Kernels.size());
		for (size_t i = 0; i < Kernels.size(); i++)
		{
			TrainingSet[i]=std::make_shared<Eigen::MatrixXf>(train_size,train_size);
			*TrainingSet[i]= Kernels[i]->block(0, 0, train_size, train_size);
		//	TrainingSet.push_back(train);
		//	std::cout << train->block(0, 0, 10, 10) << std::endl;
		}
		Eigen::VectorXf All_Phe = phe_train;
		phe_train.resize(train_size);
		phe_test.resize(test_size);
		Cov_train.resize(train_size, Covs.npar);
		Cov_test.resize(test_size, Covs.npar);
		phe_train = All_Phe.block(0, 0,train_size,1);

		phe_test = All_Phe.block(train_size, 0, test_size, 1);
	
		Cov_train = Covs.Covariates.block(0, 0, train_size, Covs.npar);
		Cov_test = Covs.Covariates.block(train_size, 0, test_size, Covs.npar);
		if (programOptions["scaley"].as<bool>())
		{
			phe_train = (phe_train.array() - phe_train.mean()) / std::sqrt(Variance(phe_train));
			phe_test = (phe_test.array() - phe_test.mean()) / std::sqrt(Variance(phe_test));
		}
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		ss.str("");
		ss.clear();
		ss << "Spliting Completed. There are " << phe_train.size() << " individuals in the training dataset, and " << phe_test.size() << " individuals in the testing dataset" << "." << std::endl;
		ss<<"In the training datset,  mean is "<< phe_train.mean() <<" and standard deviation is "<<std::sqrt(Variance(phe_train))<<"."<<std::endl;
		ss << "In the testing datset,  mean is " << phe_test.mean() << " and standard deviation is " << std::sqrt(Variance(phe_test)) << "." << std::endl;
		std::cout << ss.str();
		LOG(INFO) << ss.str();
	}
	else
	{
		if (programOptions["scaley"].as<bool>())
		{
			phe_train = (phe_train.array() - phe_train.mean()) / std::sqrt(Variance(phe_train));
		}
	}

	if (!NoE)
	{
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
						/*
						for (size_t i = 0; i < Kernels.size(); i++)
						{
							std::stringstream ss;
							ss << "kernel: " << i << "in Main" << std::endl;
							ss << "First 10x10: \n" << Kernels[i]->block(0, 0, 10, 10) << std::endl;
							ss << "Last 10x10: \n" << Kernels[i]->block(Kernels[i]->rows() - 10, Kernels[i]->cols() - 10, 10, 10);
							LOG(INFO) << ss.str() << std::endl;
						}
						*/
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
					seed = programOptions["seed"].as<float>();
				}
				if (programOptions.count("minque0"))
				{
					BatchMINQUE0(minopt, TrainingSet, phe_train, Cov_train, VarComp, fix, nsplit, seed, nthread, phe.dataType);
					iterateTimes = 1;
				}
				else
				{
					BatchMINQUE1(minopt, TrainingSet, phe_train, Cov_train, VarComp, fix, iterateTimes, nsplit, seed, nthread, isecho, phe.dataType);
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
					cMINQUE0(minopt, TrainingSet, phe_train, Cov_train, VarComp, fix);
					iterateTimes = 1;
				}
				else
				{
					if (!VarComp.size())
					{
						std::cout << "Using results from MINQUE(0) as inital value." << std::endl;
						LOG(INFO) << "Using results from MINQUE(0) as inital value.";

						std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels_imq(TrainingSet.size());
						int nind = phe.fid_iid.size();
						for (int i = 0; i < Kernels.size(); i++)
						{
							Kernels_imq[i] = std::make_shared<Eigen::MatrixXf>(*TrainingSet[i]);
							//*Kernels_imq[i] = *Kernels[i];
						}
						cMINQUE0(minopt, Kernels_imq, phe_train, Cov_train, VarComp, fix);
						for (int i = 0; i < TrainingSet.size(); i++)
						{
							Kernels_imq[i] = nullptr;
						}
						Kernels_imq.clear();
						//cMINQUE0(minopt, Kernels, phe, Covs.Covariates, VarComp, fix);
					}

					cMINQUE1(minopt, TrainingSet, phe_train, Cov_train, VarComp, fix, iterateTimes, isecho);
				}

			}
		}
	}
	
	if (programOptions.count("fix"))
	{
		if (Cov_train.cols()==0)
		{
			std::stringstream ss;
			ss << "[Warning]: There are no covariates, skip the fixed effects estimation process.";
			std::cout << ss.str() << std::endl;
			LOG(WARNING) << ss.str();
		}
		else
		{
			for (int i = 0; i < TrainingSet.size(); i++)
			{
				std::cout << "Used count: " << TrainingSet[i].use_count() << std::endl;
			}
			TrainingSet.clear();
			TrainingSet.resize(Kernels.size());
			for (size_t i = 0; i < Kernels.size(); i++)
			{
				TrainingSet[i] = std::make_shared<Eigen::MatrixXf>(train_size, train_size);
				*TrainingSet[i] = Kernels[i]->block(0, 0, train_size, train_size);
				//	std::cout << train->block(0, 0, 10, 10) << std::endl;
			}
			if (Covs.names.size() <= 0)
			{
				throw(std::string("[Error]: There are neither covariates nor intercept."));
			}
			isecho = true;
			if (programOptions.count("echo"))
			{
				isecho = programOptions["echo"].as<bool>();
			}
			std::cout << "Starting fix effects estimation." << std::endl;
			Fixed_estimator(minopt, TrainingSet, phe_train, Cov_train, VarComp, fix, iterateTimes, isecho);
			std::cout << "Fix effects eatimation is done." << std::endl;
			LOG(INFO) << "Fix effects eatimation is done.";

			std::stringstream ss;
			ss << "Fix effects:" << std::endl;
			for (size_t i = 0; i < Covs.names.size(); i++)
			{
				ss << Covs.names[i] << "\t" << fix[i] << "\n";
			}
			LOG(INFO) << ss.str();
			std::cout << ss.str() << std::endl;
			TrainingSet.clear();
		}
	
	}

	if (programOptions.count("predict") )
	{
		
		if (programOptions.count("effects"))
		{
			std::string fix_effect = programOptions["effects"].as < std::string >();
			std::ifstream Fixifstream(fix_effect, std::ios::in);
			if (!Fixifstream.is_open())
			{
				Fixifstream.close();
				throw std::string("Error: can not open the file [" + fix_effect + "] to read.");
			}
			std::cout << "Loading fix effects values from  file [" << fix_effect << "]" << std::endl;
			
			std::map<std::string, float> fix_est;
			while (!Fixifstream.eof())
			{
				std::string str;
				std::getline(Fixifstream, str);
				if (!str.empty() && str.back() == 0x0D)
					str.pop_back();
				if (Fixifstream.fail())
				{
					continue;
				}
				boost::algorithm::trim(str);
				if (str.empty())
				{
					continue;
				}

				std::vector<std::string> strVec;
				boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"), boost::token_compress_on);
				fix_est.insert({ strVec[0], std::stof(strVec[1]) });
			}
			if (fix_est.size()!=Covs.names.size())
			{
				throw(std::string("[Error]: The input fix effects do not match the data in covariates file. Please check the file again."));
			}
			fix.resize(Covs.names.size());
			for (size_t i = 0; i < Covs.names.size() ; i++)
			{
				if (fix_est.count(Covs.names[i])<=0)
				{
					std::stringstream ss;
					ss << "[Error]: Cannot find the effect value of " << Covs.names[i] << " in the file [" << fix_effect << '].' << std::endl;
					throw(ss.str());
				}
				fix[i] = fix_est[Covs.names[i]];
			}
			std::stringstream ss;
			ss << "Loaded effect value: \n\t";
			if (fix.size() > 0)
			{
				ss << "fix effects:";
				for (size_t i = 0; i < Covs.names.size(); i++)
				{
					ss << Covs.names[i] << ":" << fix[i] << "\t";
				}

			}
		}
		if (programOptions.count("predict"))
		{
			std::stringstream ss;
			ss << "Starting Prediction with variance components: "<< VarComp.transpose()<<std::endl;
			
			if (fix.size() > 0)
			{
				ss << "fix effects:";
				for (size_t i = 0; i < Covs.names.size(); i++)
				{
					ss << Covs.names[i] << ":" << fix[i] << "\t";
				}

			}
			std::cout << ss.str() << std::endl;
			LOG(INFO) << ss.str();
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
			std::shared_ptr<Prediction> pred;
		

			std::shared_ptr<Evaluate> Eva;
			if (abs(ratio-1)<1e-6)
			{
				Eigen::VectorXf phe_all = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
				if (programOptions["scale"].as<bool>())
				{
					phe_all = (phe_all.array() - phe_all.mean()) / std::sqrt(Variance(phe_all));
				}
				pred = std::make_shared<Prediction>(phe_all, Kernels, VarComp, Covs.Covariates, fix, phe.dataType == 1 ? true : false, mode, ratio);
				Eva=std::make_shared<Evaluate>(phe_all, pred->getPredictY(), phe.dataType);
			}
			else
			{
				if (!mode)
				{
					Eigen::VectorXf phe_all(phe_train.size() + phe_test.size());
					phe_all << phe_train, phe_test;
		//			Eigen::VectorXf phe_all = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
//					if (programOptions["scale"].as<bool>())
	//				{
	//					phe_all = (phe_all.array() - phe_all.mean()) / std::sqrt(Variance(phe_all));
//					}
					pred = std::make_shared<Prediction>(phe_all, Kernels, VarComp, Covs.Covariates, fix, phe.dataType == 1 ? true : false, mode, ratio);
				}
				else
				{
					pred = std::make_shared<Prediction>(phe_test, Kernels, VarComp, Cov_test, fix, phe.dataType == 1 ? true : false, mode);
				}
				Eva = std::make_shared<Evaluate>(phe_test, pred->getPredictY(), phe.dataType);
			}
			predict.resize(2);

			//	std::stringstream ss;
			if (phe.dataType == 1)
			{
				predict[0] = Eva->getMSE();
				predict[1] = Eva->getAUC();
				//	ss << "misclassification error:\t" << pred.getMSE() << std::endl << "AUC:\t" << pred.getAUC() << std::endl;
			}
			else
			{
				predict[0] = Eva->getMSE();
				predict[1] = Eva->getAUC();
				//ss << "MSE:\t" << pred.getMSE() << std::endl << "Correlation:\t" << pred.getCor() << std::endl;
			}
			//		std::cout << ss.str();
			//		LOG(INFO) << ss.str();
			//		out<<  ss.str();

			std::ofstream out;
			Eigen::VectorXf Predict_value = pred->getPredictY();
			Eigen::MatrixXf AUROC(Predict_value.rows(), 2);
			AUROC << phe_test, Predict_value;
			out.open("Prediction_KNN.txt", std::ios::out);
			out << "True\tPrediction" << std::endl;
			out << AUROC << std::endl;
			out.close();
		}

	
	}
	


	if (programOptions.count("save"))
	{
		std::string saveNet = programOptions["save"].as < std::string >();
		std::ofstream Binofstream;
		Binofstream.open(saveNet, std::ios::out | std::ios::binary);
		long long filesize = VarComp.size() * sizeof(float);
		char* f_buf = new char[filesize];
		//	#pragma omp parallel for shared(f_buf,kdata)
		for (long long k = 0; k < VarComp.size(); k++)
		{
			
			//int ret = snprintf(str2, sizeof(str2), "%f", kdata.kernelMatrix(i, j));
			unsigned long long pointer = k * sizeof(float);
			float value = VarComp[k];
			memcpy(&f_buf[pointer], (char*)&value, sizeof(float));
		}
		Binofstream.write(f_buf, filesize);
		Binofstream.close();
	}

	return iterateTimes;
//	out.close();
}

std::vector<std::shared_ptr<Evaluate>>  FNNAnalysis(boost::program_options::variables_map programOptions, DataManager& dm)
{
	torch::manual_seed(0);
	std::vector<std::shared_ptr<Evaluate>> prediction_error;
 /// All data in FNN/NN framework will be treated as double precision.
	auto options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor one = torch::ones(1, options);
	torch::set_default_dtype(one.dtype());
	///////////////////////////////////////////////
	int basis= programOptions["basis"].as < int >();
	float seed = programOptions["seed"].as < float >();
	int epoch= programOptions["epoch"].as < double >();
	float ratio= programOptions["ratio"].as < float >();
	
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


	std::vector<int64_t> dims;
	bool ignore_pos = false;
	if (programOptions.count("layer"))
	{
		std::string layers = programOptions["layer"].as < std::string >();
		std::vector<std::string> strVec;
		boost::algorithm::split(strVec, layers, boost::algorithm::is_any_of(","), boost::token_compress_on);
		for (int i = 0; i < strVec.size(); i++)
		{
			dims.push_back(atoi(strVec.at(i).c_str()));
		}
		if (dims[dims.size() - 1] > 1 && dm.getPhe_prt()->vloc.size() == 0)
		{
			throw std::string("Trying to interplote the response, but the knots are missing.");
		}
		
	}

	std::shared_ptr< Dataset> data_full = dm.GetDataset();
	if (dims[dims.size() - 1] == 1 && !data_full->phe.isUnivariate)
	{
		data_full = data_full->wide2long();
	}
	std::shared_ptr<Dataset> train=nullptr;
	std::shared_ptr<Dataset> test =nullptr;
	
	std::shared_ptr<TensorData> data = std::make_shared<TensorData>(data_full->phe, data_full->geno, data_full->cov);
	
	bool sinlgeknot = (data->isBalanced && data->getLoc().size() == data->nind) || !data->isBalanced;   // univariate analysis, with response is interpolated on different knot.
																										// if the data is unbalanced, we train the model per object, which will be considered as  univariate analysis, with response is interpolated on different knot.

	int  loss = programOptions["loss"].as < int >();
	data->dataType = loss;

	if (loss!=0 && dm.getPhe_prt()->Phenotype.cols()!=1)
	{
		throw std::string("The multivarite multiclass analysis does not support!");
	}

	std::vector<double> lambdas;
	if (programOptions.count("lambda"))
	{
		std::string lamb = programOptions["lambda"].as < std::string >();
		std::vector<std::string> strVec;
		boost::algorithm::split(strVec, lamb, boost::algorithm::is_any_of(","), boost::token_compress_on);
		for (int i = 0; i < strVec.size(); i++)
		{
			lambdas.push_back(std::stod(strVec.at(i).c_str()));
		}

	}
	if (programOptions.count("load"))
	{
		if (lambdas.size() > 1)
		{
			throw std::string("[Error]: try to predict the results from well trained network, please input the best lambda from last training.");
		}
		std::string loadNet = programOptions["load"].as < std::string >();
		//	string model_path = "model.pt";
		std::cout << "Loading model from ["<<loadNet<<"]." << std::endl;
		if (isFNN)
		{
			
			std::cout << "Funtional neural network analysis is runing" << std::endl;
			LOG(INFO) << "Funtional neural network analysis is runing";
			
			std::shared_ptr<FNN> f = std::make_shared<FNN>(dims, lambdas[0]);
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
			std::shared_ptr<NN> f = std::make_shared<NN>(dims, lambdas[0]);
			f->build(ncovs);
			f->eval();
		//	torch::serialize::InputArchive archive;
			//	std::string file("test_model.pt");
		//	archive.load_from(loadNet);
			//f->load(archive);
			torch::load(f, loadNet);

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
		//std::cout << test->phe.loc << std::endl;
		ss << "Split the training dataset into two sub-dataset, subtraining, validation.";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		ss.str("");
		ss.clear();
		if (abs(ratio-1)<1e-6)
		{
			std::tie(subtrain, valid) = train->split(seed, 0.8);
		}
		else
		{
			std::tie(subtrain, valid) = train->split(seed, ratio);
		}
		//std::shared_ptr<TensorData> d=std::make_shared<TensorData>(dm.getPhenotype(), dm.getGenotype(), dm.GetCovariates());
		std::cout << "Apply the training data for Analysis." << std::endl;
		ss << "Spliting Completed. There are " << subtrain->phe.nind << " individuals in the training dataset used for training, and " << valid->phe.nind << " individuals for validation.";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		ss.str("");
		ss.clear();
		std::shared_ptr<TensorData> train_tensor = std::make_shared<TensorData>(train->phe, train->geno, train->cov);
		std::shared_ptr<TensorData> test_tensor = std::make_shared<TensorData>(test->phe, test->geno, test->cov);
		train_tensor->dataType = loss;
		test_tensor->dataType = loss;
		/*
		std::ofstream test_y;
		test_y.open("Testing_Y.txt");

		std::ofstream test_z;
		test_z.open("Testing_Z.txt");

		for (int i = 0; i < test->phe.Phenotype.size(); i++)
		{
			test_y << (i+1) << "\t" << 1 << "\t" <<test->phe.Phenotype(i,0) << std::endl;
			if (i==0)
			{
				test_z << "FID\tIID\t";
				for (size_t j = 0; j < test->cov.names.size(); j++)
				{
					test_z << test->cov.names[j] << "\t";
				}
				test_z << std::endl;
			}
			test_z << (i + 1) << "\t" << 1<<"\t" << test->cov.Covariates.row(i)<< std::endl;
		}
	
		test_y.close();
		test_z.close();
		*/
		//std::shared_ptr<TensorData> subtrain_tensor = nullptr;
	//	std::shared_ptr<TensorData> valid_tensor =nullptr;
		std::shared_ptr<TensorData> subtrain_tensor = std::make_shared<TensorData>(subtrain->phe, subtrain->geno, subtrain->cov);
		std::shared_ptr<TensorData> valid_tensor = std::make_shared<TensorData>(valid->phe, valid->geno, valid->cov);
//		std::cout << test_tensor->getY() << std::endl;
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
			std::shared_ptr<FNN> f;
			//		std::cout << "========================" << std::endl;
			double best_lambda=lambdas[0];
			torch::Tensor train_loss = torch::tensor(INFINITY);
			std::stringstream Best_Par;
			int epoch=0;
			for (size_t i = 0; i < lambdas.size(); i++)
			{
				double lambda = lambdas[i];
				f= std::make_shared<FNN>(dims, lambda);
				torch::Tensor valid_loss;
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
						valid_loss = training<torch::optim::Adam, FNN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, FNN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, FNN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, FNN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, FNN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, FNN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else
					{
						throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
					}
					//test_loss=Testing<FNN, torch::nn::CrossEntropyLoss>(f, test_tensor);
					break;
				}
				//	for (const auto& p : f->parameters()) 
				//	{
			//			std::cout << p << std::endl;
			//		}
		
				std::cout << "Training completed." << std::endl;
				std::cout << "========================" << std::endl;
				std::stringstream ss;
				ss <<"Lambda: "<<lambda <<"\tepoch: " << f->epoch << "\tTraining: loss: " << f->loss.item<double>() << "\t Validation loss: " << valid_loss.item<double>();
				std::cout << ss.str() << std::endl;
				LOG(INFO) << ss.str() << std::endl;
				if (valid_loss.item<double>() < train_loss.item<double>())
				{
					best_lambda = lambda;
					train_loss = valid_loss;
					ss.str(std::string());
					epoch=f->epoch;
					torch::serialize::OutputArchive output_archive;
					f->save(output_archive);
					output_archive.save_to(ss);
				}
			}

			if (lambdas.size()>1)
			{
				std::stringstream ss;
				ss << "Select lambda: " << best_lambda << "as best one to train the model.";
				std::cout << ss.str() << std::endl;
				LOG(INFO) << ss.str() << std::endl;
				f = std::make_shared<FNN>(dims, best_lambda);
				torch::serialize::InputArchive input_archive;
				input_archive.load_from(ss);
				f->load(input_archive);
				f->epoch = epoch;
				f->loss = train_loss;
			/*	double lambda = best_lambda;
				f = std::make_shared<FNN>(dims, lambda);
				torch::Tensor valid_loss;
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
						valid_loss = training<torch::optim::Adam, FNN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, FNN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, FNN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, FNN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, FNN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, FNN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else
					{
						throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
					}
					//test_loss=Testing<FNN, torch::nn::CrossEntropyLoss>(f, test_tensor);
					break;
				}
				//	for (const auto& p : f->parameters()) 
				//	{
			//			std::cout << p << std::endl;
			//		}
				train_loss = valid_loss;
				std::cout << "Training completed." << std::endl;
				std::cout << "========================" << std::endl;
				std::stringstream ss1;
				ss1 << "Lambda: " << lambda << "\tepoch: " << f->epoch << "\tTraining: loss: " << f->loss.item<double>() << "\t Validation loss: " << train_loss.item<double>();
				std::cout << ss1.str() << std::endl;
				LOG(INFO) << ss1.str() << std::endl;*/
			}

			if (programOptions.count("save"))
			{
				std::string saveNet = programOptions["save"].as < std::string >();
				//	string model_path = "model.pt";
				torch::serialize::OutputArchive output_archive;
				f->save(output_archive);
				output_archive.save_to(saveNet);
			}
		
			f->eval();

		///////////////////////////////////////////////////////////////////////
			/// evaluate the testing dataset
			if (test_tensor->nind != 0)
			{

				std::shared_ptr<Evaluate> test = nullptr;
				torch::Tensor pred_test = f->forward(test_tensor);
				test = std::make_shared< Evaluate>(test_tensor->getY(), pred_test, test_tensor->dataType);
				prediction_error.push_back(test);	
				torch::Tensor out_data = torch::cat({ test_tensor->getY() , pred_test }, 1);
				Eigen::MatrixXd out_data_M = dtt::libtorch2eigen<double>(out_data);
				std::ofstream out;
				out.open("Prediction_testing.txt", std::ios::out);
				out << out_data_M << std::endl;
				out.close();
			}
			else
			{
				prediction_error.push_back(nullptr);
			}
			////////////////////////////////////////////////////////////
			// evaluate the total dataset
			std::shared_ptr<Evaluate> train = nullptr;
			torch::Tensor pred_train = f->forward(train_tensor);
			train = std::make_shared< Evaluate>(train_tensor->getY(), pred_train, train_tensor->dataType);
			prediction_error.push_back(train);
			torch::Tensor out_data = torch::cat({ train_tensor->getY() , pred_train }, 1);

			Eigen::MatrixXd out_data_M = dtt::libtorch2eigen<double>(out_data);

			std::ofstream out;
			out.open("Prediction_training.txt", std::ios::out);
			out << out_data_M << std::endl;
			out.close();
		}
		else
		{
			std::cout << "Neural network analysis is runing" << std::endl;
			LOG(INFO) << "Neural network analysis is runing";
			//	std::cout << "========================" << std::endl;
			std::shared_ptr<NN> f;
			//		std::cout << "========================" << std::endl;
			double best_lambda = lambdas[0];
			torch::Tensor train_loss = torch::tensor(INFINITY);
			for (size_t i = 0; i < lambdas.size(); i++)
			{
				double lambda = lambdas[i];
				f = std::make_shared<NN>(dims, lambda);
				f->build(ncovs);
				torch::Tensor valid_loss;
				switch (loss)
				{
				case 0:
					//		train_loss = training<torch::optim::Adam, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					if (optimType == 0)
					{
						valid_loss = training<torch::optim::Adam, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, NN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, NN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, NN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, NN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
				ss << "Lambda: " << lambda << "\tepoch: " << f->epoch << "\tTraining: loss: " << f->loss.item<double>() << "\t Validation loss: " << valid_loss.item<double>();
				std::cout << ss.str() << std::endl;
				LOG(INFO) << ss.str() << std::endl;
				if (valid_loss.item<double>() < train_loss.item<double>())
				{
					best_lambda = lambda;
					train_loss = valid_loss;
				}

			}

			if (lambdas.size() > 1)
			{
				std::stringstream ss;
				ss << "Select lambda: " << best_lambda << " as best one to train the model.";
				std::cout << ss.str() << std::endl;
				LOG(INFO) << ss.str() << std::endl;
				double lambda = best_lambda;
				f = std::make_shared<NN>(dims, lambda);
				f->build(ncovs);
				torch::Tensor valid_loss;
				switch (loss)
				{
				case 0:
					//		train_loss = training<torch::optim::Adam, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					if (optimType == 0)
					{
						valid_loss = training<torch::optim::Adam, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
				  		valid_loss = training<torch::optim::SGD, NN, torch::nn::MSELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, NN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, NN, torch::nn::BCELoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
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
						valid_loss = training<torch::optim::Adam, NN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else if (optimType == 1)
					{
						valid_loss = training<torch::optim::SGD, NN, torch::nn::CrossEntropyLoss>(f, subtrain_tensor, valid_tensor, lr, epoch);
					}
					else
					{
						throw std::string("[Error]: The program only supports adam and SDG, please check the optimizer option.");
					}
					//test_loss=Testing<NN, torch::nn::CrossEntropyLoss>(f, test_tensor);
					break;
				}
				train_loss = valid_loss;
				std::cout << "Training completed." << std::endl;
				std::cout << "========================" << std::endl;
				std::stringstream ss1;
				ss1 << "Lambda: " << lambda << "\tepoch: " << f->epoch << "\tTraining: loss: " << f->loss.item<double>() << "\t Validation loss: " << train_loss.item<double>();
				std::cout << ss1.str() << std::endl;
				LOG(INFO) << ss1.str() << std::endl;
				

			}
			f->eval();
			if (programOptions.count("save"))
			{
				std::string saveNet = programOptions["save"].as < std::string >();
				//	string model_path = "model.pt";
		//		torch::serialize::OutputArchive output_archive;
		//		f->save(output_archive);
		//		output_archive.save_to(saveNet);
				torch::save(f, saveNet);
			}
		 
			if (test_tensor->getY().sizes()[0] != 0)
			{

				torch::Tensor pred_test = f->forward(test_tensor);
				std::shared_ptr<Evaluate> test_eva = std::make_shared<Evaluate>(test_tensor->getY(), pred_test, test_tensor->dataType);
				prediction_error.push_back(test_eva);
				/*torch::Tensor out_data = torch::cat({ test_tensor->getY() , pred_test }, 1);
				std::ofstream out;
				Eigen::MatrixXd out_data_M = dtt::libtorch2eigen<double>(out_data);
				out.open("Prediction_testing.txt", std::ios::out);
				out << out_data_M << std::endl;
				out.close();
				*/  /*
				
				std::vector<int> shuffledID(test->phe.nind);
				#pragma omp parallel for
				for (int i = 0; i < shuffledID.size(); i++)
				{
					shuffledID[i] = i;
				}
				auto rng = std::default_random_engine{};
				rng.seed(seed);
				std::shuffle(std::begin(shuffledID), std::end(shuffledID), rng);
				std::stringstream ss_bug;
				ss << "\\\\\\\\\\\\\\\\\\\\\n" << "Covs\tMSE\tCor" << std::endl;
				for (int i = 0; i < test->cov.npar; i++)
				{
					CovData cov_shuffle = test->cov;
					for (size_t j = 0; j < shuffledID.size(); j++)
					{
						cov_shuffle.Covariates(j, i) = test->cov.Covariates(shuffledID[j], i);
					}
					std::shared_ptr<TensorData> test_tensor_shuffle = std::make_shared<TensorData>(test->phe, test->geno, cov_shuffle);
					torch::Tensor pred_test_shuffle = f->forward(test_tensor_shuffle);
					std::shared_ptr<Evaluate> test_eva_shuffle = std::make_shared<Evaluate>(test_tensor->getY(), pred_test_shuffle, test_tensor->dataType);
					ss << cov_shuffle.names[i] << "\t" << test_eva_shuffle->getMSE() << "\t" << test_eva_shuffle->getCor() << std::endl;
				}
				std::cout << ss.str() << std::endl;
				*/
	//			prediction_error.push_back(test.getAUC());
			}
			else
			{
				prediction_error.push_back(nullptr);
				//prediction_error.push_back(-9);
			}

			
			std::shared_ptr<Evaluate> train_eva = nullptr;
			torch::Tensor pred_train = f->forward(train_tensor);
			train_eva = std::make_shared< Evaluate>(train_tensor->getY(), pred_train, train_tensor->dataType);
			prediction_error.push_back(train_eva);
			///////////
			/*
			std::vector<int> shuffledID(train->phe.nind);
			#pragma omp parallel for
			for (int i = 0; i < shuffledID.size(); i++)
			{
				shuffledID[i] = i;
			}
			auto rng = std::default_random_engine{};
			rng.seed(seed);
			std::shuffle(std::begin(shuffledID), std::end(shuffledID), rng);
			std::stringstream ss_bug;
			ss << "\\\\\\\\\\\\\\\\\\\\\n" << "Covs\tMSE\tCor" << std::endl;
			for (int i = 0; i < train->cov.npar; i++)
			{
				CovData cov_shuffle = train->cov;
				for (size_t j = 0; j < shuffledID.size(); j++)
				{
					cov_shuffle.Covariates(j, i) = train->cov.Covariates(shuffledID[j], i);
				}
				std::shared_ptr<TensorData> train_tensor_shuffle = std::make_shared<TensorData>(train->phe, train->geno, cov_shuffle);
				torch::Tensor pred_train_shuffle = f->forward(train_tensor_shuffle);
				std::shared_ptr<Evaluate> train_eva_shuffle = std::make_shared<Evaluate>(train_tensor->getY(), pred_train_shuffle, train_tensor->dataType);
				ss << cov_shuffle.names[i] << "\t" << train_eva_shuffle->getMSE() << "\t" << train_eva_shuffle->getCor() << std::endl;
			}
			std::cout << ss.str() << std::endl;
			*/
			/*
			torch::Tensor out_data = torch::cat({ train_tensor->getY() , pred_train }, 1);
			Eigen::MatrixXd out_data_M = dtt::libtorch2eigen<double>(out_data);
			std::ofstream out;
			out.open("Prediction_training.txt", std::ios::out);
			out << out_data_M << std::endl;
			out.close();
			*/
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
