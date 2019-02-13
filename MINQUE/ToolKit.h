#pragma once
#include <vector>
#include <Eigen/Dense>
using namespace std;
class ToolKit
{
public:
	static void ArraytoVector(double ** a, int n, int m, vector<vector<double>>& v, bool Transpose = false);
	static void Array2toArrat1(double **a, int n, int m, double *b, bool Colfirst=true);
	static void Vector2toArray1(vector<vector<double>>& v, double *b, bool Colfirst = true);
	static void Stringsplit(string &org, vector<string> & splited, string delim);

};

