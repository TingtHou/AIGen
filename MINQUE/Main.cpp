// MINQUE.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "ToolKit.h"
#include "MINQUE.h"
#include <string>
#include <fstream>
int main()
{
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
	double **U = (double **)malloc(nind * sizeof(double*));
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
		U[id] = (double *)malloc(ncols* sizeof(double));
		for (int i=0;i<ncols;i++)
		{
			U[id][i] = stod(tmp[i]);
		}
		id++;
	}
	infile.close();
	vector<vector<double>> LN1;
	ToolKit::ArraytoVector(U, nind, ncols, LN1, false);
	MINQUE varest;
	varest.import_data(Y);
	varest.push_back_U(LN1);
	varest.var_est();
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
