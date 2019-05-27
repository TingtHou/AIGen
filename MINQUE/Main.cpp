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
#include "imnq.h"
#include <boost/algorithm/string.hpp>
void readResponseFile(string resopnsefile, Eigen::VectorXd &Response,ofstream &logout);
void readKernelFiles(std::vector<string> kernels, std::vector<Eigen::MatrixXd> &KMatrix, int nind, ofstream &logout);

void readProgramOptions(boost::program_options::variables_map programOptions, ofstream &out, ofstream &logout, Eigen::VectorXd &Response, std::vector<Eigen::MatrixXd> &Kmatrix)
{
	std::vector<string> kernelfiles;
	std::string reponsefile;
	std::string result = "result.txt";
	std::string logfile = "log.txt";
	
	if (programOptions.count("kernel"))
	{
		kernelfiles = programOptions["kernel"].as<std::vector<std::string> >();
	}
	if (programOptions.count("phe"))
	{
		reponsefile = programOptions["phe"].as < std::string >();
	}
	if (programOptions.count("out"))
	{
		result = programOptions["out"].as < std::string >();
	}
	if (programOptions.count("log"))
	{
		logfile = programOptions["log"].as < std::string >();
	}
	logout.open(logfile, ios::out);
	out.open(result, ios::out);
	readResponseFile(reponsefile, Response,logout);
	readKernelFiles(kernelfiles, Kmatrix, Response.size(),logout);
}

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
	if (programOptions.count("altinverse"))
	{
		std::string Decomposition = programOptions["altinverse"].as < std::string >();
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

int main(int argc, const char *const argv[])
{

	ofstream out;
	ofstream logout;
	bool GPU=false;
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
	
	if (programOptions.count("GPU"))
	{
		GPU = true;
	}

	std::time_t currenttime = std::time(0);
	char tAll[255];
 	tm Tm;
	localtime_s(&Tm, &currenttime);
	std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S", &Tm);
	logout << tAll << std::endl;
	Eigen::VectorXd Response;
	std::vector<Eigen::MatrixXd> Kmatrix;
	readProgramOptions(programOptions, out, logout, Response, Kmatrix);
	MinqueOptions minopt;
	readAlgrithomParameter(programOptions, minopt);
	std::vector<double> VarComp;
 	Eigen::MatrixXd e(Response.size(), Response.size());
	e.setIdentity();
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
		varest.importY(Response);
		
		for (int i = 0; i < Kmatrix.size(); i++)
		{
			varest.pushback_Vi(Kmatrix[i]);
		}
		varest.pushback_Vi(e);
		std::cout << "starting CPU MINQUE" << std::endl;
		clock_t t1 = clock();
		varest.estimate();
		std::cout << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		VarComp.resize(varest.getvcs().size());
		Eigen::VectorXd::Map(&VarComp[0], varest.getvcs().size()) = varest.getvcs();
	}
	out << "Source\tVariance" << std::endl;
	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
	int i = 0;
	double VG = 0;
	for (; i < VarComp.size()-1; i++)
	{
		std::cout << VarComp.at(i) << " ";
		int index = i + 1;
		out<<"V(G"<<index<<")\t"<< VarComp.at(i) << std::endl;
		VG += VarComp.at(i);
	}
	out << "V(e)\t" << VarComp.at(i) << std::endl;
	double varrsp = Variance(Response);
	out << "Vp\t" << varrsp << std::endl;
	for (i=0; i < VarComp.size() - 1; i++)
	{
		std::cout << VarComp.at(i) << " ";
		int index = i + 1;
		out << "V(G" << index << ")/Vp\t" << VarComp.at(i)/ varrsp << std::endl;
	}
	out << "Sum of V(G)/Vp\t" << varrsp / varrsp << std::endl;
	logout << std::endl;
	logout.flush();
	logout.close();
	out.close();
}
void readResponseFile(string resopnsefile, Eigen::VectorXd &Response, ofstream &logout)
{
	ifstream infile;
	string str;
	std::vector<double> yvector;
	yvector.clear();
	clock_t t1 = clock();
	std::cout << "Reading Phenotype from " << resopnsefile << std::endl;
	logout << "Reading Phenotype from " << resopnsefile << std::endl;
	infile.open(resopnsefile);
	while (getline(infile, str))
	{
		boost::algorithm::trim(str);
		if (str.empty())
		{
			continue;
		}
		vector<string> strVec;
		boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"));
		yvector.push_back(stod(strVec[2]));
	}
	infile.close();
	int nind = yvector.size();
	std::cout << nind << " Total" << std::endl;
	Response= Eigen::VectorXd(nind);
	for (int i = 0; i < nind; i++)
	{
		Response[i] = yvector[i];
	}
	std::cout << "Reading Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	logout << "Reading Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	logout.flush();
}

void readKernelFiles(std::vector<string> kernels, std::vector<Eigen::MatrixXd> &KMatrix, int nind, ofstream &logout)
{
	for (int i=0;i<kernels.size();i++)
	{
		ifstream infile;
		Eigen::MatrixXd Kernel(nind, nind);
		Kernel.setZero();
		infile.open(kernels.at(i));
		string str = "";
		int id = 0;
		int ncols = 0;
		std::cout << "Reading kernel matrix from " << kernels.at(i) << std::endl;
		logout << "Reading kernel matrix from " << kernels.at(i) << std::endl;
		clock_t t1 = clock();
		while (getline(infile, str))
		{
			std::vector<std::string> tmp;
			ToolKit::Stringsplit(str, tmp, "\t");
			if (!ncols)
			{
				ncols = tmp.size();
			}
			for (int i = 0; i < ncols; i++)
			{
				Kernel(id, i) = stod(tmp[i]);
			}
			id++;
			std::cout << fixed << setprecision(2) << "\r\rReading Kernel Matrix: " << (double)id * 100 / (double)nind << "%";
		}
		infile.close();
		std::cout << std::endl;
		std::cout << "Reading kernel matrix elapse time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		logout << "Reading kernel matrix elapse time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		KMatrix.push_back(Kernel);
		logout.flush();
	}
}

//Testing
// void test()
// {
// 	// Testing 
// //	InverseTest();
// //	RandomModel(3000);
// 	//boostProgramOptionsRoutine(argc, argv);
// // 	PlinkReader plreader;
// // 	plreader.test();
// // 	cuToolkit::cuGetGPUinfo();
// // 	std::vector<std::string> files;
// // 	std::string path = "C:\\Users\\Hou59\\source\\repos\\tingtHou\\Kernel-Based-Neural-Network\\NData\\N02";
// // 	getFiles(path, files);
// // 	for (int i=0;i<files.size();i++)
// // 	{
// // 		MinqueSample(files.at(i));
// // 	}
// 
// }

// void RandomModel(int nind)
// {
// //	int nind = 3000;
// 	Eigen::VectorXd Y(nind);
// 	Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(nind, 1);
// 	Eigen::MatrixXd W2 = Eigen::MatrixXd::Random(nind, 1);
// 	Eigen::MatrixXd W3 = Eigen::MatrixXd::Random(nind, 1);
// 	Eigen::MatrixXd e(nind, nind);
// 	Eigen::MatrixXd V1(nind, nind);
// 	Eigen::MatrixXd V2(nind, nind);
// 	//	Eigen::MatrixXd V3(nind, nind);
// 	e.setIdentity();
// 	std::vector<double> edouble;
// 	Random rad(1);
// 	V1.setZero();
// 	V2.setZero();
// 	for (int i = 0; i < Y.size(); i++)
// 	{
// 		V1(i, i) = W1(i, 0)*W1(i, 0);
// 		V2(i, i) = W2(i, 0)*W2(i, 0);
// 		//	V3(i, i) = W3(i, 0)*W3(i, 0);
// 		Y(i) = W1(i, 0)*rad.Normal(0, 1.3) + W2(i, 0)*rad.Normal(0, 1.5) + /*W3(i, 0)*rad.Normal(0, 1.5) + */rad.Normal(0, 1);
// 		edouble.push_back(Y[i]);
// 	}
// 	std::cout << "means: " << ave(edouble) << "\nVariance: " << variance(edouble) << std::endl;
// 
// 	cuMINQUE varest;
// 	varest.importY(Y.data(), nind);
// 	varest.pushback_Vi(e.data(), nind);
// 	varest.pushback_Vi(V1.data(), nind);
// 	varest.pushback_Vi(V2.data(), nind);
// 	std::cout << "starting GPU MINQUE" << std::endl;
// 	clock_t t1 = clock();
// 	varest.estimate();
// 	std::cout << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
// 	std::vector<double> theta = varest.GetTheta();
// 	for (int i = 0; i < theta.size(); i++)
// 	{
// 		std::cout << theta.at(i) << " ";
// 	}
// 	std::cout << std::endl;
// 	MINQUE min;
// 	min.importY(Y);
// 	min.pushback_Vi(e);
// 	min.pushback_Vi(V1);
// 	min.pushback_Vi(V2);
// 	std::cout << "starting CPU MINQUE" << std::endl;
// 	clock_t t2 = clock();
// 	min.estimate();
// 	std::cout << "CPU Elapse Time : " << (clock() - t2) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	std::cout << fixed << setprecision(4) << "Estimated Variances: " << min.Gettheta().transpose() << std::endl;
// 	Eigen::VectorXd Real(3);
// 	Real << 1, 1.3*1.3, 1.5*1.5;
// 	std::cout << "Expected Variances: " << Real.transpose() << std::endl;
// 	std::cout << "Bias: ";
// 	for (int i = 0; i < min.Gettheta().size(); i++)
// 	{
// 		std::cout << abs(Real[i] - min.Gettheta()[i]) / Real[i] << "\t";
// 	}
// 	std::cout << std::endl;
// }
// 
// 
// void getFiles(string path, std::vector<string>& files)
// {
// 	//文件句柄
// 	intptr_t hFile = 0;
// 	//文件信息
// 	struct _finddata_t fileinfo;
// 	string p;
// 	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
// 	{
// 		do
// 		{
// 			//如果是目录,迭代之
// 			//如果不是,加入列表
// 			if ((fileinfo.attrib &  _A_SUBDIR))
// 			{
// 				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
// 					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
// 			}
// 		} while (_findnext(hFile, &fileinfo) == 0);
// 		_findclose(hFile);
// 	}
// }
// 
// 
// void MinqueSample(string folder)
// {
// 
// 	std::string stime;
// 	std::stringstream strtime;
// 	std::time_t currenttime = std::time(0);
// 	char tAll[255];
// 	tm Tm;
// 	localtime_s(&Tm, &currenttime);
// 	std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S", &Tm);
// //	std::string folder = "../data2/001.dvp/";
// 	std::string Yfile = folder + "\\rsp.txt";
// 	std::string Fixedfile = folder + "\\fix.txt";
// 	std::string fitfile = folder + "\\fit.txt";
// 	std::string fitrfile = folder + "\\vcs.txt";
// 	std::string FixedDM = folder + "\\xmx.txt";
// 	std::string L1file = folder + "\\knl/LN1.txt";
// 	std::string L2file = folder + "\\knl/LN2.txt";
//  //	std::string L3file = folder + "\\knl/LN3.txt";
// 	std::string logfile = folder + "\\log.txt";
// 	ofstream oufile;
// 	oufile.open(logfile, ios::out);
// 	oufile << tAll << std::endl;
// 	oufile.flush();
// 	ifstream infile;
// 	string str;
// 	std::vector<double> yvector;
// 	yvector.clear();
// 	clock_t t1 = clock();
// 	std::string header = "";
// 	///////////////////////////////////////////////////////////////////////////////////
// 	std::cout << "Reading fit file: ";
// 	infile.open(fitfile);
// 	getline(infile, header);
// 	getline(infile, str);
// 	infile.close();
// 	std::vector<std::string> tmp;
// 	ToolKit::Stringsplit(str, tmp, "\t");
// 	Eigen::VectorXd fited(tmp.size());
// 	for (int i = 0; i < tmp.size(); i++)
// 	{
// 		fited(i) = stod(tmp[i]);
// 	}
// 	std::cout << fited.transpose() << std::endl;
// 	/////////////////////////////////////////////////////////////////////////////////////////
// 	std::cout << "Reading Phenotype: ";
// 	infile.open(Yfile);
// 	while (getline(infile, str))
// 	{
// 		yvector.push_back(stod(str));
// 	}
// 	infile.close();
// 	int nind = yvector.size();
// 	std::cout << nind<<" Total" << std::endl;
// 	Eigen::VectorXd Y(nind);
// 	for (int i = 0; i < nind; i++)
// 	{
// 		Y[i] = yvector[i];
// 	}
// 	oufile << "Reading Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	///////////////////////////////////////////////////////////////////////////////////////
// 	infile.open(Fixedfile);
// 	str = "";
// 	getline(infile, str);
// 	getline(infile, str);
// 	infile.close();
// 	tmp.clear();
// 	ToolKit::Stringsplit(str, tmp, "\t");
// 	Eigen::VectorXd FxCov(tmp.size());
// 	for (int i=0;i<tmp.size();i++)
// 	{
// 		FxCov(i) = stod(tmp[i]);
// 	}
// 	std::cout << fixed << setprecision(4) <<"Expected fixed covariates: " << FxCov.transpose() << std::endl;
// 	/////////////////////////////////////////////////////////
// 	infile.open(fitrfile);
// 	str = "";
// 	getline(infile, str);
// 	getline(infile, str);
// 	infile.close();
// 	tmp.clear();
// 	ToolKit::Stringsplit(str, tmp, "\t");
// 	Eigen::VectorXd RnCov(tmp.size());
// 	for (int i = 0; i < tmp.size(); i++)
// 	{
// 		RnCov(i) = stod(tmp[i]);
// 	}
// 	std::cout << fixed << setprecision(4) << "Expected Random covariates: " << RnCov.transpose() << std::endl;
// 	/////////////////////////////////////////////////////////////////////////
// 
// 	std::cout << "Reading fixed covariates design matrix: ";
// 	Eigen::MatrixXd FXDM(nind, FxCov.size());
// 	double **X;
// 	X = (double **)malloc(nind * sizeof(double*));
// 	FXDM.setZero();
// 	infile.open(FixedDM);
// 	str = "";
// 	int id = 0;
// 	int ncols = 0;
// 	t1 = clock();
// 	while (getline(infile, str))
// 	{
// 		std::vector<std::string> tmp;
// 		ToolKit::Stringsplit(str, tmp, "\t");
// 		if (!ncols)
// 		{
// 			ncols = tmp.size();
// 		}
// 		X[id] = (double *)malloc(ncols * sizeof(double));
// 		for (int i = 0; i < ncols; i++)
// 		{
// 			FXDM(id, i) = stod(tmp[i]);
// 			X[id][i] = stod(tmp[i]);
// 		}
// 		id++;
// 		std::cout << fixed << setprecision(2) << "\r\rReading fixed covariates design matrix: " << (double)id * 100 / (double)nind << "%";
// 	}
// 	infile.close();
// 	std::cout << std::endl;
// 	oufile << "Reading fixed covariates design matrix Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	//////////////////////////////////////////////////////////////////////////////////////
// 	Eigen::VectorXd Fixed(ncols);
// // 	LinearRegression lr(Y.data(), X, false, nind, nind, ncols);
// // 	lr.MLE();
// 	for (int i = 0; i < ncols; i++) {
// 		Fixed[i] = fited[i];
//  	}
// 	std::cout  << fixed << setprecision(4) << "Estimated fixed effect: " << Fixed.transpose() << std::endl;
// 	oufile << fixed << setprecision(4) << "Estimated fixed effect: " << Fixed.transpose() << std::endl;
// 	for (int i=0;i<nind;i++)
// 	{
// 		free(X[i]);
// 	}
// 	free(X);
// 	Y = Y - FXDM * Fixed;
// /////////////////////////////////////////////////////////////
// 	Eigen::MatrixXd LN1(nind, nind);
// 	LN1.setZero();
// 	infile.open(L1file);
// 	str = "";
// 	id = 0;
// 	ncols = 0;
// 	std::cout << "Reading LN1 Matrix: ";
// 	t1 = clock();
// 	while (getline(infile, str))
// 	{
// 		std::vector<std::string> tmp;
// 		ToolKit::Stringsplit(str, tmp, "\t");
// 		if (!ncols)
// 		{
// 			ncols = tmp.size();
// 		}
// 		for (int i = 0; i < ncols; i++)
// 		{
// 			LN1(id, i) = stod(tmp[i]);
// 		}
// 		id++;
// 		std::cout << fixed << setprecision(2)<< "\r\rReading LN1 Matrix: " << (double)id * 100 / (double)nind << "%";
// 	}
// 	infile.close();
// 	std::cout << std::endl;
// 	oufile << "Reading Reading LN1 Matrix Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 
// 	////////////////////////////////////////
// 	
// 	Eigen::MatrixXd LN2(nind, nind);
// 	LN2.setZero();
// 	infile.open(L2file);
// 	str = "";
// 	id = 0;
// 	ncols = 0;
// 	std::cout << "Reading LN2 Matrix: ";
// 	t1 = clock();
// 	while (getline(infile, str))
// 	{
// 		vector<string> tmp;
// 		ToolKit::Stringsplit(str, tmp, "\t");
// 		if (!ncols)
// 		{
// 			ncols = tmp.size();
// 		}
// 		for (int i = 0; i < ncols; i++)
// 		{
// 			LN2(id, i) = stod(tmp[i]);
// 		}
// 		id++;
// 		std::cout << fixed << setprecision(2) << "\r\rReading LN2 Matrix: " << (double)id * 100 / (double)nind << "%";
// 	}
// 	infile.close();
// 	oufile << "Reading Reading LN2 Matrix Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	std::cout << std::endl;
// 	
// 	/////////////////////////////////////////////////////
// // 	Eigen::MatrixXd LN3(nind, nind);
// // 	LN3.setZero();
// // 	infile.open(L3file);
// // 	str = "";
// // 	id = 0;
// // 	ncols = 0;
// // 	std::cout << "Reading LN3 Matrix: ";
// // 	t1 = clock();
// // 	while (getline(infile, str))
// // 	{
// // 		vector<string> tmp;
// // 		ToolKit::Stringsplit(str, tmp, "\t");
// // 		if (!ncols)
// // 		{
// // 			ncols = tmp.size();
// // 		}
// // 		for (int i = 0; i < ncols; i++)
// // 		{
// // 			LN3(id, i) = stod(tmp[i]);
// // 		}
// // 		id++;
// // 		std::cout << fixed << setprecision(2) << "\r\rReading LN3 Matrix: " << (double)id * 100 / (double)nind << "%";
// // 	}
// // 	infile.close();
// // 	oufile << "Reading Reading LN3 Matrix Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// // 
// // 	std::cout << std::endl;
// // 	oufile.flush();
// 	
// 	//////////////////////////////////////////////////
// 
// 	cuMINQUE cuvarest;
// 	std::cout << "Push back Y" << std::endl;
// 	cuvarest.importY(Y.data(), Y.size());
// 	Eigen::MatrixXd IdN(Y.size(), Y.size());
// 	IdN.setIdentity();
// 	cuvarest.pushback_Vi(IdN.data(),Y.size());
// 	std::cout << "Push back LN1 matrix" << std::endl;
// 	cuvarest.pushback_Vi(LN1.data(), LN1.rows());
// 	std::cout << "Push back LN2 matrix" << std::endl;
// 	cuvarest.pushback_Vi(LN2.data(), LN2.rows());
// // 	std::cout << "Push back LN3 matrix" << std::endl;
// // 	cuvarest.push_back_Vi(LN3.data(), LN3.rows());
// 	std::cout << "starting GPU MINQUE" << std::endl;
// 	t1 = clock();
// 	cuvarest.estimate();
// 	std::cout << fixed << setprecision(2) << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	oufile << setprecision(2) << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
// 	oufile << fixed << setprecision(4) << "Estimated Variances: ";
// 	std::vector<double> theta = cuvarest.GetTheta();
// 	for (int i = 0; i < theta.size(); i++)
// 	{
// 		std::cout << theta.at(i) << " ";
// 		oufile << theta.at(i) << " ";
// 	}
// 	std::cout << std::endl;
// 	oufile << std::endl;
// 	oufile.flush();
// 	///////////////////////////////////////////////
// 	MINQUE varest;
// 	varest.importY(Y);
// // 	Eigen::MatrixXd IdN(Y.size(), Y.size());
// // 	IdN.setIdentity();
// 	varest.pushback_Vi(IdN);
// 	varest.pushback_Vi(LN1);
//  	varest.pushback_Vi(LN2);
//  //	varest.pushback_Vi(LN3);
// 	std::cout << "starting CPU MINQUE" << std::endl;
// 	t1 = clock();
// 	varest.estimate();
// 	std::cout << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	oufile << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	std::cout << fixed << setprecision(4)<< "Estimated Variances: " << varest.Gettheta().transpose() << std::endl;
// 	oufile << fixed << setprecision(4) << "Estimated Variances: " << varest.Gettheta().transpose() << std::endl;
// 	oufile.flush();
// 	////////////////////////////////////////////
// 	Eigen::VectorXd Randomed = varest.Gettheta();
// 	Eigen::VectorXd Finalest(Fixed.size()+Randomed.size());
// 	Finalest << Fixed,
// 		Randomed;
// 	Eigen::VectorXd Expected(Finalest.size());
// 	Expected << FxCov,
// 		RnCov;
// 	std::cout << "Expected effect by File: " << Expected.transpose() << std::endl;
// 	oufile << "Expected effect by File: " << Expected.transpose() << std::endl;
// 	std::cout << "Estimated effect by MIN: " << Finalest.transpose() << std::endl;
// 	oufile << "Estimated effect by MIN: " << Finalest.transpose() << std::endl;
// 	std::cout << "Estimated effect in File:" << fited.transpose() << std::endl;
// 	oufile << "Estimated effect by File: " << fited.transpose() << std::endl;
// 	std::cout << "Bias with File: ";
// 	oufile << "Bias with File: ";
// 	for (int i = 0; i < fited.size(); i++)
// 	{
// 		std::cout << abs(Finalest[i] - fited[i]) / fited[i] << "\t";
// 		oufile << abs(Finalest[i] - fited[i]) / fited[i] << "\t";
// 		oufile.flush();
// 	}
// 	std::cout << std::endl;
// 	oufile << std::endl;
// 	std::cout << "Bias with Expect: ";
// 	oufile << "Bias with Expect: ";
// 	for (int i = 0; i < Expected.size(); i++)
// 	{
// 		std::cout << abs(Finalest[i] - Expected[i]) / Expected[i] << "\t";
// 		oufile << abs(Finalest[i] - Expected[i]) / Expected[i] << "\t";
// 		oufile.flush();
// 	}
// 	oufile << std::endl;
// 	std::cout << std::endl;
// 	oufile.close();
// 	std::cout << std::endl;
// }
// 
// 
// 	


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
