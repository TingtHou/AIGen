#include "pch.h"
#include "ToolKit.h"

void ToolKit::ArraytoVector(double ** a, int n, int m, vector<vector<double>>& v, bool Transpose)
{
	v.clear();
	if (Transpose)
	{
		v.resize(m);
		for (int i = 0; i < m; i++)
		{
			v[i].resize(n);
			for (int j = 0; j < n; j++)
			{
				v[i][j] = a[j][i];
			}
		}
	}
	else
	{
		v.resize(n);
		for (int i = 0; i < n; i++)
		{
			v[i].resize(m);
			for (int j = 0; j < m; j++)
			{
				v[i][j] = a[i][j];
			}
		}

	}
}


void ToolKit::Array2toArrat1(double **a, int n, int m, double *b, bool Colfirst)
{
	int k = 0;
	if (Colfirst)
	{
		for (int i=0;i<m;i++)
		{
			for (int j=0;j<n;j++)
			{
				b[k++] = a[j][i];
			}
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				b[k++] = a[i][j];
			}
		}
	}
}

void ToolKit::Vector2toArray1(vector<vector<double>>& v, double * b, bool Colfirst)
{
	int n = v.size();
	int m = v[0].size();
	int k = 0;
	if (Colfirst)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				b[k++] = v[j][i];
			}
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				b[k++] = v[i][j];
			}
		}
	}
}

void ToolKit::Stringsplit(string & org, vector<string>& splited, string delim)
{
	splited.clear();
	size_t current;
	size_t next = -1;
	do
	{
		current = next + 1;
		next = org.find_first_of(delim, current);
		splited.push_back(org.substr(current, next - current));
	} while (next != string::npos);
}

