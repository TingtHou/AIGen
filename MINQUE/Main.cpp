// MINQUE.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "ToolKit.h"
#include "MINQUE.h"
#include <string>
#include <fstream>
#include <Eigen/Dense>
int main()
{
// 	Eigen::MatrixXd a = Eigen::MatrixXd::Random(5, 5);
// 	std::cout << "\\\\\\\\\Matrixxd Version\\\\\\\\\\\\\\" << std::endl;
// 	std::cout << a << std::endl;
// 	std::cout << "\\\\\\\\\\Vector Version\\\\\\\\\\\\" << std::endl;
// 	vector<vector<double>> a_inv(5);
// 	for (int i=0;i<5;i++)
// 	{
// 		a_inv[i].resize(5);
// 		for (int j=0;j<5;j++)
// 		{
// 			a_inv[i][j] = a(i, j);
// 		}
// 	}
// 	for (int i = 0; i < 5; i++)
// 	{
// 		for (int j = 0; j < 5; j++)
// 		{
// 			std::cout << a_inv[i][j] << "\t";
// 		}
// 		std::cout << std::endl;
// 	}
// 
// 	MatFunc::LUinversion(a_inv, 1e-20);
// 	std::cout << "\\\\\\\\\\Inversion Version\\\\\\\\\\\\" << std::endl;
// 	for (int i = 0; i < 5; i++)
// 	{
// 		for (int j = 0; j < 5; j++)
// 		{
// 			std::cout << a_inv[i][j] << "\t";
// 		}
// 		std::cout << std::endl;
// 	}
// 	std::cout << "\\\\\\\\\\Inversion Version\\\\\\\\\\\\" << std::endl;
// 	Eigen::MatrixXd ainv(5, 5);
// 	for (int i = 0; i < 5; i++)
// 	{
// 		for (int j = 0; j < 5; j++)
// 		{
// 			ainv(i, j) = a_inv[i][j];
// 			
// 		}
// 	}
// 	std::cout << a * ainv << std::endl;
	ifstream infile;
	infile.open("../data/rsp.txt");
	string str;
	vector<double> Y;
	while (getline(infile, str))
	{
		Y.push_back(stod(str));
	}
	infile.close();
	int nind = Y.size();
	double **ULN1 = (double **)malloc(nind * sizeof(double*));
	infile.open("../data/LN1.txt");
	str="";
	int id = 0;
	int ncols=0;
	while (getline(infile, str))
	{
		vector<string> tmp;
		ToolKit::Stringsplit(str, tmp, "\t");
		if (!ncols)
		{
			ncols = tmp.size();
		}
		ULN1[id] = (double *)malloc(ncols* sizeof(double));
		for (int i=0;i<ncols;i++)
		{
			ULN1[id][i] = stod(tmp[i]);
		}
		id++;
	}
	infile.close();
	vector<vector<double>> LN1;
	ToolKit::ArraytoVector(ULN1, nind, ncols, LN1, true);
	////////////////////////////////
	double **ULN2 = (double **)malloc(nind * sizeof(double*));
	infile.open("../data/LN2.txt");
	str = "";
	id = 0;
	ncols = 0;
	while (getline(infile, str))
	{
		vector<string> tmp;
		ToolKit::Stringsplit(str, tmp, "\t");
		if (!ncols)
		{
			ncols = tmp.size();
		}
		ULN2[id] = (double *)malloc(ncols * sizeof(double));
		for (int i = 0; i < ncols; i++)
		{
			ULN2[id][i] = stod(tmp[i]);
		}
		id++;
	}
	infile.close();
	vector<vector<double>> LN2;
	ToolKit::ArraytoVector(ULN2, nind, ncols, LN2, true);

	////////////////////////////
	double **ULN3 = (double **)malloc(nind * sizeof(double*));
	infile.open("../data/LN3.txt");
	str = "";
	id = 0;
	ncols = 0;
	while (getline(infile, str))
	{
		vector<string> tmp;
		ToolKit::Stringsplit(str, tmp, "\t");
		if (!ncols)
		{
			ncols = tmp.size();
		}
		ULN3[id] = (double *)malloc(ncols * sizeof(double));
		for (int i = 0; i < ncols; i++)
		{
			ULN3[id][i] = stod(tmp[i]);
		}
		id++;
	}
	infile.close();
	vector<vector<double>> LN3;
	ToolKit::ArraytoVector(ULN3, nind, ncols, LN3, true);

	/////////////////////////////////

	MINQUE varest;
	varest.import_data(Y);
	varest.push_back_Vi(LN1);
	varest.push_back_Vi(LN2);
	varest.push_back_Vi(LN3);
	std::cout << "starting MINQUE" << std::endl;
	varest.MIVQUE();
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
