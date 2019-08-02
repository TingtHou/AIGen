// MINQUE.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include "pch.h"
#include <iostream>
#include "ToolKit.h"
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <time.h>
#include <iomanip>
#include <io.h>
#include <random>
#include "LinearRegression.h"
#include "Random.h"
#include "PlinkReader.h"
#include <vector>
#include <stdio.h>
#include "Options.h"
#include <iostream>
#include "CheckMode.h"
#include "CommonFunc.h"
#include "KernelManage.h"
#include "DataManager.h"
#include "imnq.h"
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <map>
#include <boost/serialization/map.hpp>
#include <boost/serialization/split_member.hpp>
#include "KernelGenerator.h"

void readAlgrithomParameter(boost::program_options::variables_map programOptions, MinqueOptions &minque)
{
	if (programOptions.count("iterate"))
	{
		minque.iterate = programOptions["iterate"].as<int>();
	}
	if (programOptions.count("tolerance"))
	{
		minque.tolerance = programOptions["tolerance"].as<double>();
	}
	if (programOptions.count("pseudo"))
	{
		minque.allowPseudoInverse = programOptions["pseudo"].as<bool>();
	}
	if (programOptions.count("inverse"))
	{
		std::string Decomposition = programOptions["inverse"].as < std::string >();
		transform(Decomposition.begin(), Decomposition.end(), Decomposition.begin(), tolower);
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
				throw ("The parameter inverse is not correct, please check it. More detail --help");
				break;
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
				throw ("The parameter inverse is not correct, please check it. More detail --help");
			}
		}
		
		
	}
	if (programOptions.count("ginverse"))
	{
		std::string Decomposition = programOptions["ginverse"].as < std::string >();
		transform(Decomposition.begin(), Decomposition.end(), Decomposition.begin(), tolower);
		if (isNum(Decomposition))
		{
			int number;
			std::istringstream iss(Decomposition);
			iss >> number;
			switch (number)
			{
			case 2:
				minque.MatrixDecomposition = 2;
				break;
			case 3:
				minque.MatrixDecomposition = 3;
				break;
			default:
				throw ("The parameter ginverse is not correct, please check it. More detail --help");
				break;
			}
		}
		else
		{
			if (Decomposition == "qr")
			{
				minque.MatrixDecomposition = 2;

			}
			else if (Decomposition == "svd")
			{
				minque.MatrixDecomposition = 3;
			}
			else
			{
				throw ("The parameter ginverse is not correct, please check it. More detail --help");
			}
		}
	}
}
void ReadData(boost::program_options::variables_map programOptions, DataManager &dm, LOG *logout);

void MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager &dm, LOG *log, std::string &out);


int main(int argc, const char *const argv[])
{
	printf("\n"
		"@----------------------------------------------------------@\n"
		"|        KNN       |     v alpha 0.1    |   19/July/2019   |\n"
		"|----------------------------------------------------------|\n"
		"|    Statistical Genetics and Statistical Learning Group   | \n"
		"@----------------------------------------------------------@\n"
		"\n");
	string result = "result.txt";
	string logfile = "result.log";
	LOG *logout = nullptr;
	try
	{
		Options opt(argc, argv);
		boost::program_options::variables_map programOptions = opt.GetOptions();
		if (1 == argc || programOptions.count("help"))
		{
			std::cout << opt.GetDescription() << std::endl;
			return 1;
		}
		if (programOptions.count("version"))
		{
			std::cout << "Kernel Based Neural Network Software alpha 0.1" << std::endl;
			return 1;

		}
		if (programOptions.count("out"))
		{
			result = programOptions["out"].as < std::string >();
			std::string basename = GetBaseName(result);
			std::string ParentPath = GetParentPath(result);
			int positionDot = basename.rfind('.');
			logfile = ParentPath +"/"+basename.substr(0,positionDot) + ".log";
			programOptions.erase("out");
		}
		if (programOptions.count("log"))
		{
			logfile = programOptions["log"].as < std::string >();
			programOptions.erase("log");
		}
		logout = new LOG(logfile);
		
		DataManager dm;
		ReadData(programOptions, dm, logout);
		std::vector<KernelData> kernelList;\
		//generate a built-in kernel or not
		if (programOptions.count("make-kernel"))
		{
			int kerneltype;
			std::string kernelname= programOptions["make-kernel"].as < std::string >();
			if (isNum(kernelname))
			{
				switch (stoi(kernelname))
				{
				case 0:
					kerneltype = CAR;
					break;
				case 1:
					kerneltype = Identity;
					break;
				case 2:
					kerneltype = Product;
					break;
				case 3:
					kerneltype = Ploymonial;
					break;
				case 4:
					kerneltype = Gaussian;
					break;
				case 5:
					kerneltype = IBS;
					break;
				default:
					throw ("Error: The kernel name [" + kernelname + "] cannot be identified.");
					break;
				}
			}
			else
			{
				transform(kernelname.begin(), kernelname.end(), kernelname.begin(), tolower);
				if (kernelname == "car")
				{
					kerneltype = CAR;
				}
				else if (kernelname == "identity")
				{
					kerneltype = Identity;
				}
				else if (kernelname == "product")
				{
					kerneltype = Product;
				}
				else if (kernelname == "polynomial")
				{
					kerneltype = Ploymonial;
				}
				else if (kernelname == "gaussian")
				{
					kerneltype = Gaussian;
				}
				else if (kernelname == "ibs")
				{
					kerneltype = IBS;
				}
				else
				{
					throw ("Error: The kernel name [" + kernelname + "] cannot be identified.");
				}
			}
			double weight= programOptions["weight"].as < double >();
			double constant= programOptions["constant"].as < double >();
			double deg = programOptions["deg"].as < double >();
			double sigmma=programOptions["sigma"].as < double >();
			GenoData gd = dm.getGenotype();
			if (programOptions.count("std"))
			{
				logout->write("Standardize the genotype matrix", true);
				stdSNPmv(gd.Geno);
			}
			logout->write("Starting generate a " +kernelname+" kernel from input genotype", true);
			KernelGenerator kernelGenr(gd, kerneltype, weight, constant, deg, sigmma);
			kernelList.push_back(kernelGenr.getKernel());
		}
		//The built-in kernel will overwrite the kernel from file
		if (kernelList.size())
		{
			dm.SetKernel(kernelList);
		}
		//if the phenotype is inputed, the estimation will be started.
		if (dm.getPhenotype().fid_iid.size()!=0)
		{
			dm.match();
			if (!programOptions.count("skip"))
			{
				MINQUEAnalysis(programOptions, dm, logout, result);
			}

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
		logout->close();
		delete logout;
	}
	catch (string &e)
	{
		logout->write(e, true);
		logout->close();
		delete logout;
//		std::cout << e << std::endl;
	}
	catch (const std::exception &err)
	{
		logout->write(err.what(), true);
		logout->close();
		delete logout;
	}
	//get now time
	time_t now = time(0);
	//format time to string;
	char dt[256];
	ctime_s(dt,256, &now);
	std::cout << "\n\nAnalysis finished: " << dt << std::endl;
}

void ReadData(boost::program_options::variables_map programOptions, DataManager &dm, LOG *logout)
{
	std::vector<std::string> GFile;
	if (programOptions.count("phe"))
	{
		std::string reponsefile = programOptions["phe"].as < std::string >();
		logout->write("Reading Phenotype from [" + reponsefile + "].",true);
		clock_t t1 = clock();
		dm.readPhe(reponsefile);
		std::stringstream ss;
		ss << "Read Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		logout->write(ss.str(), true);
	}
	if (programOptions.count("kernel"))
	{
		std::string kernelfiles = programOptions["kernel"].as<std::string >();
		logout->write("Reading kernel list from [" + kernelfiles + "].", true);
		clock_t t1 = clock();
		dm.readKernel(kernelfiles);
		std::stringstream ss;
		ss << "Read kernel file Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		logout->write(ss.str(), true);
	}
	if (programOptions.count("mkernel"))
	{
		std::string mkernelfile = programOptions["mkernel"].as<std::string >();
		logout->write("Reading kernel list from [" + mkernelfile + "].", true);
		clock_t t1 = clock();
		dm.readmKernel(mkernelfile);
		std::stringstream ss;
		ss << "Read multi-kernel Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		logout->write(ss.str(), true);
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
	if (GFile.size()!=0)
	{
		clock_t t1 = clock();
		dm.readGeno(GFile,false);
		std::stringstream ss;
		ss << "Read Genotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
		logout->write(ss.str(), true);
	}

}

void MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager &dm, LOG *logout, std::string &result)
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

	std::vector<double> VarComp;
	Eigen::MatrixXd e(phe.fid_iid.size(), phe.fid_iid.size());
	e.setIdentity();
	int iterateTimes = 0;
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
		imnq varest;
		varest.setOptions(minopt);
		varest.importY(phe.Phenotype);
		varest.setLogfile(logout);
		for (int i = 0; i < kd.size(); i++)
		{
			varest.pushback_Vi(kd[i].kernelMatrix);
		}
		varest.pushback_Vi(e);
		std::cout << "starting CPU MINQUE" << std::endl;
		clock_t t1 = clock();
		varest.estimate();
		std::cout << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		VarComp.resize(varest.getvcs().size());
		Eigen::VectorXd::Map(&VarComp[0], varest.getvcs().size()) = varest.getvcs();
		iterateTimes = varest.getIterateTimes();
	}
	ofstream out;
	out.open(result, ios::out);
	logout->write("---Result----", false);
	out << "Source\tVariance" << std::endl;
	logout->write("Source\tVariance", false);
//	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
	int i = 0;
	double VG = 0;
	double VP = 0;
	std::stringstream ss;
	for (; i < VarComp.size() - 1; i++)
	{
	//	std::cout << VarComp.at(i) << " ";
		int index = i + 1;
		ss.str("");
		ss << "V(G" << index << ")\t" << VarComp.at(i);
		out << ss.str() << std::endl;
		logout->write(ss.str(), false);
		VG += VarComp.at(i);
	}
	VP = VG;
	ss.str("");
	ss << "V(e)\t" << VarComp.at(i);
	out << ss.str() << std::endl;
	logout->write(ss.str(), false);
	VP += VarComp.at(i);
	ss.str("");
	ss << "Vp\t" << VP;
	out << ss.str() << std::endl;
	logout->write(ss.str(), false);
	for (i = 0; i < VarComp.size() - 1; i++)
	{
		int index = i + 1;
		ss.str("");
		ss << "V(G" << index << ")/Vp\t" << VarComp.at(i) / VP;
		out << ss.str() << std::endl;
		logout->write(ss.str(), false);
	}
	ss.str("");
	ss << "Sum of V(G)/Vp\t" << VG / VP;
	out << ss.str() << std::endl;
	logout->write(ss.str(), false);
	ss.str("");
	ss << "Iterate Times:\t" << iterateTimes + 1;
	out << ss.str() << std::endl;
	logout->write(ss.str(), false);
	out.close();
}