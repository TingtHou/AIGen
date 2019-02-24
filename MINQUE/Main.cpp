// MINQUE.cpp : This file contains the 'main' function. Program execution begins and ends there.

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
#include <random>
#include "eigenmvn.h"
#include "Random.h"
//#include "cuMatrix.cuh"
//extern "C" void cuMatrixTrace(const double *Mat, double *sum, int ncols);
void MinqueSample(string folder);
void getFiles(string path, vector<string>& files);

double ave(double a[], int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += a[i];
	return sum / n;
} //平均值，没什么好说的?
//方差要用的是平均值，不是中值； 
double ave(std::vector<double> a)
{
	int sum = 0;
	for (int i = 0; i < a.size(); i++)
		sum += a[i];
	return sum / (double)a.size();
} //平均值，没什么好说的?
//方差要用的是平均值，不是中值； 


double variance(std::vector<double> a)
{
	double sum = 0;
	double average = ave(a);//函数调用！！不许有【】！！！不许有int和double ！！ 
	for (int i = 0; i < a.size(); i++)
		sum = (a[i] - average)*(a[i] - average);
	return sum / (double)a.size();
}

double Normaldist(double mean, double var)
{
	std::default_random_engine e; //引擎
	std::normal_distribution<double> n(mean, var); //均值, 方差
	return n(e);
}

template <class Iter> typename Iter::value_type cov(const Iter &x, const Iter &y)
{
	double sum_x = std::accumulate(std::begin(x), std::end(x), 0.0);
	double sum_y = std::accumulate(std::begin(y), std::end(y), 0.0);

	double mx = sum_x / x.size();
	double my = sum_y / y.size();

	double accum = 0.0;

	for (auto i = 0; i < x.size(); i++)
	{
		accum += (x.at(i) - mx) * (y.at(i) - my);
	}

	return accum / (x.size() - 1);
}
extern "C" void cuMatrixTrace(const double *A, const int nrows, double *Result);
int main()
{

	int nind = 1000;
	Eigen::VectorXd Y(nind);
	Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(nind,1);
	Eigen::MatrixXd W2= Eigen::MatrixXd::Random(nind, 1);
	Eigen::MatrixXd W3= Eigen::MatrixXd::Random(nind, 1);
	Eigen::MatrixXd e(nind, nind);
	Eigen::MatrixXd V1(nind, nind);
	Eigen::MatrixXd V2(nind, nind);
//	Eigen::MatrixXd V3(nind, nind);
	e.setIdentity();
	std::vector<double> edouble;
	Random rad(1);
	V1.setZero();
	for (int i=0;i<Y.size();i++)
	{
		V1(i, i) = W1(i, 0)*W1(i, 0);
		V2(i, i) = W2(i, 0)*W2(i, 0);
	//	V3(i, i) = W3(i, 0)*W3(i, 0);
		Y(i) =W1(i,0)*rad.Normal(0, 1.3)+ W2(i, 0)*rad.Normal(0, 1.5)+ /*W3(i, 0)*rad.Normal(0, 1.5) + */rad.Normal(0, 1);
		edouble.push_back(Y[i]);
	}
	std::cout << "means: " << ave(edouble) << "\nVariance: " << cov(edouble,edouble) << std::endl;

	cuMINQUE varest;
	varest.import_Y(Y.data(),nind);
	varest.push_back_Vi(e.data(),nind);
	varest.push_back_Vi(V1.data(),nind);
	varest.push_back_Vi(V2.data(), nind);
	std::cout << "starting GPU MINQUE" << std::endl;
	clock_t t1 = clock();
	varest.estimate();
	std::cout << "GPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
 	std::cout << fixed << setprecision(4) << "Estimated Variances: ";
	std::vector<double> theta = varest.GetTheta();
	for (int i = 0; i < theta.size(); i++)
	{
		std::cout << theta.at(i) << " ";
	}
	std::cout << std::endl;
	MINQUE min;
	min.importY(Y);
	min.pushback_Vi(e);
	min.pushback_Vi(V1);
	min.pushback_Vi(V2);
//	min.pushback_Vi(V3);
	std::cout << "starting CPU MINQUE" << std::endl;
	clock_t t2 = clock();
	min.estimate();
	std::cout << "CPU Elapse Time : " << (clock() - t2) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	std::cout << fixed << setprecision(4) << "Estimated Variances: " << min.Gettheta().transpose() << std::endl;
	Eigen::VectorXd Real(3);
	Real << 1 ,1.3*1.3, 1.5*1.5;
	std::cout << "Estimated Variances: " << Real.transpose() << std::endl;
	std::cout << "Bias: ";
	for (int i=0; i<min.Gettheta().size();i++)
	{
		std::cout << abs(Real[i] - min.Gettheta()[i]) / Real[i] << "\t";
	}
	std::cout << std::endl;
// 	Eigen::MatrixXd b = Eigen::MatrixXd::Random(4000, 4000);
// 	clock_t t1 = clock();
// 	Eigen::MatrixXd c=a*b;
// 	std::cout << "Matrix multiplication Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
// 	cuToolkit::cuGetGPUinfo();
// 	vector<string> files;
// 	string path = "..//data3";
// 	getFiles(path, files);
// 	for (int i=0;i<files.size();i++)
// 	{
// 		MinqueSample(files.at(i));
// 	}
	
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
// 
// 	cuMINQUE cuvarest;
// 	std::cout << "Push back Y" << std::endl;
// 	cuvarest.import_Y(Y.data(), Y.size());
//	Eigen::MatrixXd IdN(Y.size(), Y.size());
//	IdN.setIdentity();
//	cuvarest.pushback_Vi(IdN.data());
// 	std::cout << "Push back LN1 matrix" << std::endl;
// 	cuvarest.push_back_Vi(LN1.data(), LN1.rows());
// 	std::cout << "Push back LN2 matrix" << std::endl;
// 	cuvarest.push_back_Vi(LN2.data(), LN2.rows());
// 	std::cout << "Push back LN3 matrix" << std::endl;
// 	cuvarest.push_back_Vi(LN3.data(), LN3.rows());
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
	///////////////////////////////////////////////
	MINQUE varest;
	varest.importY(Y);
	Eigen::MatrixXd IdN(Y.size(), Y.size());
	IdN.setIdentity();
	varest.pushback_Vi(IdN);
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
