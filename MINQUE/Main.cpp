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
#include <map>
#include <boost/serialization/map.hpp>
#include <boost/serialization/split_member.hpp>
#include "KernelCompute.h"

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
				std::cout << "The parameter inverse is not correct, please check it. More detail --help" << std::endl;
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
				std::cout << "The parameter inverse is not correct, please check it. More detail --help" << std::endl;
				exit(0);
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
				std::cout << "The parameter altinverse is not correct, please check it. More detail --help" << std::endl;
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
				std::cout << "The parameter altinverse is not correct, please check it. More detail --help" << std::endl;
				exit(0);
			}
		}
	}
}

void MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager dm, LOG *log, ofstream &out);


int main(int argc, const char *const argv[])
{
	ofstream out;
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
			exit(0);
		}
		if (programOptions.count("version"))
		{
			std::cout << "Kernel Based Neural Network Software alpha 0.1" << std::endl;
			exit(0);
		}
		if (programOptions.count("check"))
		{
			CheckMatrixInverseMode();
			exit(0);
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
			programOptions.erase("out");
		}
		logout = new LOG(logfile);
		out.open(result, ios::out);
		DataManager dm(programOptions, logout);
		dm.read();
		if (programOptions.count("recode"))
		{
			std::vector<KernelData> kernelList = dm.GetKernel();
			for (int i = 0; i < kernelList.size(); i++)
			{
				KernelWriter kw(kernelList[i]);
				std::string outname= programOptions["recode"].as < std::string >();
				kw.writeText(outname);
			}
		}
		if (programOptions.count("make-kbin"))
		{
			std::vector<KernelData> kernelList = dm.GetKernel();
			bool isfloat = true;
			if (programOptions.count("precision"))
			{
				isfloat= programOptions["precision"].as < int >();
			}
			for (int i = 0; i < kernelList.size(); i++)
			{
				KernelWriter kw(kernelList[i]);
				std::string outname = programOptions["make-kbin"].as < std::string >();
				kw.setprecision(isfloat);
				kw.write(outname);
			}
		}
		if (!programOptions.count("skip"))
		{
			MINQUEAnalysis(programOptions, dm, logout, out);
		}
		out.close();
		delete logout;
	}
	catch (string &e)
	{
		logout->write(e, true);
		out.close();
		delete logout;
//		std::cout << e << std::endl;
	}

}

void MINQUEAnalysis(boost::program_options::variables_map programOptions, DataManager dm, LOG *logout, ofstream &out)
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
	logout->write("---Result----", false);
	out << "Source\tVariance" << std::endl;
	logout->write("Source\tVariance", false);
	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
	int i = 0;
	double VG = 0;
	double VP = 0;
	std::stringstream ss;
	for (; i < VarComp.size() - 1; i++)
	{
		std::cout << VarComp.at(i) << " ";
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
}