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


INITIALIZE_EASYLOGGINGPP

void readAlgrithomParameter(boost::program_options::variables_map programOptions, MinqueOptions& minque);

void ReadData(boost::program_options::variables_map programOptions, DataManager &dm);

void MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager &dm, std::string &out);

void BatchMINQUE(MinqueOptions &minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData & phe, std::vector<float>& variances, float & iterateTimes, int nsplit, int seed, int nthread, bool isecho);

void MINQUE(MinqueOptions &minque, std::vector<Eigen::MatrixXf> &Kernels, PhenoData &phe, std::vector<float> &variances, float &iterateTimes,bool isecho);

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
//////init logger
	el::Configurations conf;
	conf.setToDefault();
	conf.parseFromText(("*GLOBAL:\n  FILENAME = " + logfile+"\n  To_Standard_Output = false"));
	el::Loggers::reconfigureLogger("default", conf);
	std::cout << opt.print() << std::endl;
	LOG(INFO) << opt.print();
	DataManager dm;
	ReadData(programOptions, dm);
	std::vector<KernelData> kernelList; 
		//generate a built-in kernel or not
	if (programOptions.count("thread"))
	{
		int nthread = programOptions["thread"].as<int>();
		mkl_set_num_threads(nthread);
		omp_set_num_threads(nthread);
	}
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
			throw ("Error: Invalided kernel name \"" + kernelname+"\"");
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
				throw ("Error: cannot open the file [" + weightVFile + "] to read.");
			}
			int id = 0;
			while (!infile.eof())
			{
				std::string eleInVector;
				getline(infile, eleInVector);
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
			KernelWriter kw(dm.GetKernel()[i]);

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
			MINQUEAnalysis(programOptions, dm, result);
		}

	}
	
}

int main(int argc, const char *const argv[])
{
	clock_t t1 = clock();
	
	/////////////////////////////////////////////////////////////////////
	std::cout<<"\n"
		"@----------------------------------------------------------@\n"
		"|        KNN       |     v alpha 0.1.1  |    "<<__DATE__<<"   |\n"
		"|----------------------------------------------------------|\n"
		"|    Statistical Genetics and Statistical Learning Group   |\n"
		"@----------------------------------------------------------@\n"
		"\n";
	try
	{
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
	std::cout << "Total elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	return 1;
}

void ReadData(boost::program_options::variables_map programOptions, DataManager &dm)
{
	std::vector<std::string> GFile;
	bool isImpute = false;
	if (programOptions.count("phe"))
	{
		std::string reponsefile = programOptions["phe"].as < std::string >();
		std::cout << "Reading Phenotype from [" + reponsefile + "]." << std::endl;
		LOG(INFO) << "Reading Phenotype from [" + reponsefile + "].";
		clock_t t1 = clock();
		dm.readPhe(reponsefile);
		std::stringstream ss;
		ss << "Read Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		std::cout <<ss.str() << std::endl;
		LOG(INFO) << ss.str();
	}
	if (programOptions.count("kernel"))
	{
		std::string kernelfiles = programOptions["kernel"].as<std::string >();
		std::cout << "Reading kernel list from [" + kernelfiles + "]." << std::endl;
		LOG(INFO) << "Reading kernel list from [" + kernelfiles + "].";
		clock_t t1 = clock();
		dm.readKernel(kernelfiles);
		std::stringstream ss;
		ss << "Read kernel file Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
	}
	if (programOptions.count("mkernel"))
	{
		std::string mkernelfile = programOptions["mkernel"].as<std::string >();
		std::cout << "Reading kernel list from [" + mkernelfile + "]." << std::endl;
		LOG(INFO) << "Reading kernel list from [" + mkernelfile + "].";
		clock_t t1 = clock();
		dm.readmKernel(mkernelfile);
		std::stringstream ss;
		ss << "Read multi-kernel Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
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
		clock_t t1 = clock();
		dm.readGeno(GFile, isImpute);
		std::stringstream ss;
		ss << "Read Genotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str() << std::endl;
	}
	
}

void MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager &dm, std::string &result)
{
	bool GPU = false;
	if (programOptions.count("GPU"))
	{
		GPU = true;
	}
	PhenoData phe = dm.getPhenotype();
	std::vector<KernelData> kd = dm.GetKernel();
	MinqueOptions minopt;
	readAlgrithomParameter(programOptions, minopt);
	std::vector<float> VarComp;
	float iterateTimes = 0;
	bool isecho;
	if (GPU)
	{
		// 		cuMINQUE cuvarest;
		// 		cuvarest.importY(Response.data(), Response.size());
		// 		cuvarest.pushback_Vi(e.data(), e.rows());
		// 		for (int i = 0; i < Kmatrix.size(); i++)
		// 		{
		// 			cuvarest.pushback_Vi(Kmatrix[i].data(), Kmatrix[i].rows());
		// 		}
		// 		std::cout << "Starting MINQUE estimate using GPU" << std::endl;
		// 		clock_t t1 = clock();
		// 		cuvarest.estimate();
		// 		std::cout << fixed << setprecision(2) << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		// 		logout << setprecision(2) << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		// 
		// 		VarComp = cuvarest.GetTheta();
	}
	else
	{
		std::vector<Eigen::MatrixXf> Kernels;
		if (programOptions.count("alphaKNN"))
		{
			int alpha = programOptions["alphaKNN"].as<int>();
			KernelExpansion ks(kd, alpha);
			Kernels = ks.GetExtendMatrix();
		}
		else
		{
			for (int i=0;i<kd.size();i++)
			{
				Kernels.push_back(kd[i].kernelMatrix);
			}
		}
		if (programOptions.count("batch"))
		{
			int nthread = 10;
			int nsplit= programOptions["batch"].as<int>();
			int seed = 0;
			if (programOptions.count("thread"))
			{
				nthread = programOptions["thread"].as<int>();
			}
			if (programOptions.count("seed"))
			{
				seed = programOptions["seed"].as<int>();
			}
			isecho = false;
			if (programOptions.count("echo"))
			{
				isecho = programOptions["echo"].as<bool>();
			}
			BatchMINQUE(minopt, Kernels, phe, VarComp, iterateTimes,nsplit,seed, nthread, isecho);
		}
		else
		{
			isecho = true;
			if (programOptions.count("echo"))
			{
				isecho = programOptions["echo"].as<bool>();
			}
			
			MINQUE(minopt, Kernels, phe, VarComp, iterateTimes,isecho);
		}
		
	}
	ofstream out;
	out.open(result, ios::out);
	LOG(INFO) << "---Result----";
	out << "Source\tVariance" << std::endl;
	LOG(INFO) << "Source\tVariance";
//	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
	int i = 0;
	float VG = 0;
	float VP = 0;
	std::stringstream ss;
	for (; i < VarComp.size() - 1; i++)
	{
	//	std::cout << VarComp.at(i) << " ";
		int index = i + 1;
		ss.str("");
		ss << "V(G" << index << ")\t" << VarComp.at(i);
		out << ss.str() << std::endl;
		std::cout << ss.str() << std::endl;
		LOG(INFO) << ss.str();
		VG += VarComp.at(i);
	}
	VP = VG;
	ss.str("");
	ss << "V(e)\t" << VarComp.at(i);
	out << ss.str() << std::endl;
	std::cout << ss.str() << std::endl;
	LOG(INFO) << ss.str();
	VP += VarComp.at(i);
	ss.str("");
	ss << "Vp\t" << VP;
	out << ss.str() << std::endl;
	std::cout << ss.str() << std::endl;
	LOG(INFO) << ss.str();
	for (i = 0; i < VarComp.size() - 1; i++)
	{
		int index = i + 1;
		ss.str("");
		ss << "V(G" << index << ")/Vp\t" << VarComp.at(i) / VP;
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
	out.close();
}

void BatchMINQUE(MinqueOptions &minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData & phe, std::vector<float>& variances, float & iterateTimes, int nsplit,int seed, int nthread, bool isecho)
{
    int nkernel = Kernels.size();
	variances.clear();
	Batch b = Batch(Kernels, phe.Phenotype, nsplit, seed, true);
	b.start();
	std::vector<std::vector<Eigen::MatrixXf>> KernelsBatch;
	std::vector<Eigen::VectorXf> PheBatch;
	b.GetBatchKernels(KernelsBatch);
	b.GetBatchPhe(PheBatch);
	std::vector<float> time(KernelsBatch.size());
	std::vector<std::vector<float>> varsBatch(KernelsBatch.size());
	std::cout << "starting CPU MINQUE" << std::endl;
	clock_t time_total = clock();
	omp_set_num_threads(nthread);
	#pragma omp parallel for
	for (int i = 0; i < KernelsBatch.size(); i++)
	{
		clock_t t1 = clock();
		printf("Starting CPU-based analysis on thread %d\n", i);
		LOG(INFO) << "Starting CPU-based analysis on thread " << i;
		imnq varest;
		varest.isEcho(isecho);
		varest.setThreadId(i);
		varest.setOptions(minque);
		//		varest.setLogfile(logout);
		varest.importY(PheBatch[i]);
		Eigen::MatrixXf e(PheBatch[i].size(), PheBatch[i].size());
		e.setIdentity();
		for (int j = 0; j < KernelsBatch[i].size(); j++)
		{
			varest.pushback_Vi(KernelsBatch[i][j]);
		}
		varest.pushback_Vi(e);
		try
		{
			varest.estimate();
			varsBatch[i].resize(varest.getvcs().size());
			Eigen::VectorXf::Map(&varsBatch[i][0], varest.getvcs().size()) = varest.getvcs();
			time[i] = varest.getIterateTimes();
			float elapse = (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000;
			printf("The thread %d is finished, elapse time %.3f ms\n", i, elapse);
			LOG(INFO) << "The thread " << i << " is finished, elapse time " << elapse << " ms";
		}
		catch (const std::exception& err)
		{
			stringstream ss;
			ss << "[Warning]: The thread " << i << " is interrupt, because " << err.what();
			printf("%s\n", ss.str().c_str());
			LOG(WARNING)<< ss.str();
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
		}
	}
	std::cout << "CPU Elapse Time : " << (clock() - time_total) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
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
		auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i < -999; });
		if (itzero != (*it).end())
		{
			it = varsBatch.erase(it);
		}
		else
		{
			++it;
		}
	}
	for (int i = 0; i < nkernel + 1; i++)
	{
		float sum = 0;
		for (int j = 0; j < varsBatch.size(); j++)
		{
			sum += varsBatch[j][i];
		}
		variances.push_back(sum / float(varsBatch.size()));
	}
	iterateTimes = accumulate(time.begin(), time.end(), 0.0) / time.size(); ;
}

void MINQUE(MinqueOptions & minque, std::vector<Eigen::MatrixXf>& Kernels, PhenoData & phe, std::vector<float>& variances, float & iterateTimes,bool isecho)
{
	imnq varest;
	varest.setOptions(minque);
	varest.isEcho(isecho);
	//		varest.setLogfile(logout);
	varest.importY(phe.Phenotype);
	for (int i = 0; i < Kernels.size(); i++)
	{
		varest.pushback_Vi(Kernels[i]);
	}
	Eigen::MatrixXf e(phe.fid_iid.size(), phe.fid_iid.size());
	e.setIdentity();
	varest.pushback_Vi(e);
	std::cout << "starting CPU MINQUE " << std::endl;
	clock_t t1 = clock();
	varest.estimate();
	std::cout << " --- Completed" << std::endl;
	std::cout << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	variances.resize(varest.getvcs().size());
	Eigen::VectorXf::Map(&variances[0], varest.getvcs().size()) = varest.getvcs();
	iterateTimes = varest.getIterateTimes();
}

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
				logic_error emsg("The parameter inverse is not correct, please check it. More detail --help");
				throw std::exception(emsg);
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
				logic_error emsg("The parameter inverse is not correct, please check it. More detail --help");
				throw std::exception(emsg);
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
				throw std::exception(logic_error("The parameter ginverse is not correct, please check it. More detail --help"));
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
				throw std::exception(logic_error("The parameter ginverse is not correct, please check it. More detail --help"));
			}
		}
	}
}