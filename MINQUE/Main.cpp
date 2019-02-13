// MINQUE.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "ToolKit.h"
#include "minque.h"
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <time.h>
#include "cuToolkit.h"
#include "cuMINQUE.h"
#include <iomanip>
#include <io.h>


void MinqueSample(string folder);
void getFiles(string path, vector<string>& files);
int main()
{
	vector<string> files;
	string path = "C:\\Users\\Hou59\\source\\repos\\TingtHou\\Kernel-Based-Neural-Network\\data3";
	getFiles(path, files);
	for (int i=0;i<files.size();i++)
	{
		MinqueSample(files.at(i));
	}
	
}


void getFiles(string path, vector<string>& files)
{
	//文件句柄
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


void MinqueSample(string folder)
{

	std::string stime;
	std::stringstream strtime;
	std::time_t currenttime = std::time(0);
	char tAll[255];
	tm Tm;
	localtime_s(&Tm, &currenttime);
	std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S", &Tm);
//	std::string folder = "../data2/001.dvp/";
	std::string Yfile = folder + "\\rsp.txt";
	std::string L1file = folder + "\\knl/LN1.txt";
	std::string L2file = folder + "\\knl/LN2.txt";
	std::string L3file = folder + "\\knl/LN3.txt";
	std::string logfile = folder + "\\log.txt";
	ofstream oufile;
	oufile.open(logfile, ios::out);
	oufile << tAll << std::endl;
	oufile.flush();
	ifstream infile;
	infile.open(Yfile);
	string str;
	std::vector<double> yvector;
	yvector.clear();
	clock_t t1 = clock();
	std::cout << "Reading Phenotype: ";
	
	while (getline(infile, str))
	{
		yvector.push_back(stod(str));
	}
	infile.close();
	int nind = yvector.size();
	std::cout << nind<<" Total" << std::endl;
	Eigen::VectorXd Y(nind);
	for (int i = 0; i < nind; i++)
	{
		Y[i] = yvector[i];
	}
	oufile << "Reading Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
/////////////////////////////////////////////////////////////
	Eigen::MatrixXd LN1(nind, nind);
	LN1.setZero();
	infile.open(L1file);
	str = "";
	int id = 0;
	int ncols = 0;
	std::cout << "Reading LN1 Matrix: ";
	t1 = clock();
	while (getline(infile, str))
	{
		vector<string> tmp;
		ToolKit::Stringsplit(str, tmp, "\t");
		if (!ncols)
		{
			ncols = tmp.size();
		}
		for (int i = 0; i < ncols; i++)
		{
			LN1(id, i) = stod(tmp[i]);
		}
		id++;
		std::cout << fixed << setprecision(2)<< "\r\rReading LN1 Matrix: " << (double)id * 100 / (double)nind << "%";
	}
	infile.close();
	std::cout << std::endl;
	oufile << "Reading Reading LN1 Matrix Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;

	////////////////////////////////////////
	Eigen::MatrixXd LN2(nind, nind);
	LN2.setZero();
	infile.open(L2file);
	str = "";
	id = 0;
	ncols = 0;
	std::cout << "Reading LN2 Matrix: ";
	t1 = clock();
	while (getline(infile, str))
	{
		vector<string> tmp;
		ToolKit::Stringsplit(str, tmp, "\t");
		if (!ncols)
		{
			ncols = tmp.size();
		}
		for (int i = 0; i < ncols; i++)
		{
			LN2(id, i) = stod(tmp[i]);
		}
		id++;
		std::cout << fixed << setprecision(2) << "\r\rReading LN2 Matrix: " << (double)id * 100 / (double)nind << "%";
	}
	infile.close();
	oufile << "Reading Reading LN2 Matrix Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	std::cout << std::endl;
	/////////////////////////////////////////////////////
	Eigen::MatrixXd LN3(nind, nind);
	LN3.setZero();
	infile.open(L3file);
	str = "";
	id = 0;
	ncols = 0;
	std::cout << "Reading LN3 Matrix: ";
	t1 = clock();
	while (getline(infile, str))
	{
		vector<string> tmp;
		ToolKit::Stringsplit(str, tmp, "\t");
		if (!ncols)
		{
			ncols = tmp.size();
		}
		for (int i = 0; i < ncols; i++)
		{
			LN3(id, i) = stod(tmp[i]);
		}
		id++;
		std::cout << fixed << setprecision(2) << "\r\rReading LN3 Matrix: " << (double)id * 100 / (double)nind << "%";
	}
	infile.close();
	oufile << "Reading Reading LN3 Matrix Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;

	std::cout << std::endl;
	oufile.flush();
	//////////////////////////////////////////////////
	cuMINQUE cuvarest;
	std::cout << "Push back Y" << std::endl;
	cuvarest.import_Y(Y.data(), Y.size());
	std::cout << "Push back LN1 matrix" << std::endl;
	cuvarest.push_back_Vi(LN1.data(), LN1.rows());
	std::cout << "Push back LN2 matrix" << std::endl;
	cuvarest.push_back_Vi(LN2.data(), LN2.rows());
	std::cout << "Push back LN matrix" << std::endl;
	cuvarest.push_back_Vi(LN3.data(), LN3.rows());
	std::cout << "starting GPU MINQUE" << std::endl;
	t1 = clock();
	cuvarest.estimate();
	std::cout << fixed << setprecision(2) << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	oufile << setprecision(2) << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
	oufile << fixed << setprecision(4) << "Estimated Variances: ";
	std::vector<double> theta = cuvarest.GetTheta();
	for (int i = 0; i < theta.size(); i++)
	{
		std::cout << theta.at(i) << " ";
		oufile << theta.at(i) << " ";
	}
	std::cout << std::endl;
	oufile << std::endl;
	oufile.flush();
	///////////////////////////////////////////////
	MINQUE varest;
	varest.importY(Y);
	varest.pushback_Vi(LN1);
	varest.pushback_Vi(LN2);
	varest.pushback_Vi(LN3);
	std::cout << "starting CPU MINQUE" << std::endl;
	t1 = clock();
	varest.estimate();
	std::cout << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	oufile << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	std::cout << fixed << setprecision(4)<< "Estimated Variances: " << varest.Gettheta().transpose() << std::endl;
	oufile << fixed << setprecision(4) << "Estimated Variances: " << varest.Gettheta().transpose() << std::endl;
	oufile.flush();
	oufile.close();
}


	


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
