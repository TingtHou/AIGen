#include "pch.h"
#include "MINQUE.h"
#include <iostream>
MINQUE::MINQUE()
{
	_seed = -5251949;
	_n = 0;
	_r = 0;
	_U_c = 0;
}

void MINQUE::import_data(const vector<double> &Y)
{
	if (Y.empty()) throw("\"Y\" is a empty vector! mixed_model::import");
	_Y.clear();
	_Y = Y;
	_n = _Y.size();
	// initialize _Uu_c
	_Uu_c.clear();
	_Uu_c.push_back(_n); // for matrix I
}

void MINQUE::push_back_U(const vector< vector<double> > &Ui)
{
	long Ui_c = Ui.size();
	if (Ui_c == 0) throw("Empty Ui! mixed_model::push_back_U");
	if (Ui[0].size() != _n) throw("The row number of \"Ui\" and \"_n\" mismatch! MINQUE::push_back_U");
	_U.insert(_U.end(), Ui.begin(), Ui.end());
	_Uu_c.insert(_Uu_c.end() - 1, Ui_c);//random pars + Identity Matrix
	_U_c = _U.size(); // update _U_c
	_r = _Uu_c.size() - 1; // update _r

	long i = 0, j = 0, k = 0;
	vector< vector<double> > U_v(Ui_c);
	vector< vector<long> > U_p(Ui_c);
	vector<long> U_r(Ui_c);
	for (i = 0; i < Ui_c; i++)
	{
		for (j = 0; j < _n; j++)
		{
			if (abs(Ui[i][j])>1e-10)
			{
				U_v[i].push_back(Ui[i][j]);
				U_p[i].push_back(j);
				U_r[i]++;
			}
		}
	}
	_U_v.insert(_U_v.end(), U_v.begin(), U_v.end());
	_U_p.insert(_U_p.end(), U_p.begin(), U_p.end());
	_U_r.insert(_U_r.end(), U_r.begin(), U_r.end());
}

void MINQUE::push_back_Vi(const vector<vector<double>>& Vi)
{
	double lamda = 1;
	if (_Vi.size() != _n)
	{
		_Vi.clear();
		_Vi.resize(_n);
		for (int i = 0; i < _n; i++)
		{
			_Vi[i].resize(_n);
			for (int j = 0; j < _n; j++) 
				_Vi[i][j] = 0.0;
		}
	}
	for (int i=0;i<_n;i++)
	{
		for (int j=0;j<_n;j++)
		{
			_Vi[i][j] = lamda * Vi[i][j];
		}
	}
	_r++;
}

void MINQUE::init_Vi()
{
	long i = 0, j = 0;

	if (_Vi.size() != _n)
	{
		_Vi.clear();
		_Vi.resize(_n);
		for (i = 0; i < _n; i++) _Vi[i].resize(_n);
		for (i = 0; i < _n; i++) _Vi[i][i] = 1.0;
	}
	else
	{
		for (i = 0; i < _n - 1; i++)
		{
			_Vi[i][i] = 1.0;
			for (j = i + 1; j < _n; j++) _Vi[i][j] = _Vi[j][i] = 0.0;
		}
		_Vi[_n - 1][_n - 1] = 1.0;
	}
}

void MINQUE::init_Vi(const vector<double>& prior)
{
	long i = 0, j = 0;

	if (_Vi.size() != _n)
	{
		_Vi.clear();
		_Vi.resize(_n);
		for (i = 0; i < _n; i++) _Vi[i].resize(_n);
		for (i = 0; i < _n; i++) _Vi[i][i] = 1.0 / prior[_r];
	}
	else
	{
		for (i = 0; i < _n - 1; i++)
		{
			_Vi[i][i] = 1.0 / prior[_r];
			for (j = i + 1; j < _n; j++) _Vi[i][j] = _Vi[j][i] = 0.0;
		}
		_Vi[_n - 1][_n - 1] = 1.0 / prior[_r];
	}
}



void MINQUE::Minque_1()
{
	long i = 0, j = 0, k = 0, l = 0, m = 0;
	double d_buf = 0.0;
	double lamda = 0.0;

	for (i = 0; i < _r + 1; i++) _alpha[i] = 1.0;
	if (_Vi.size() != _n)
	{
		vector<double> Viu(_n);

		init_Vi();

		for (i = 0; i < _r; i++)
		{
			for (j = 0; j < _Uu_c[i]; j++, m++)
			{
				for (k = 0; k < _n; k++)
				{
					Viu[k] = 0.0;
					//_U_r number of non-zero element for each random effect
					//_U_p id of non-zero element for each random effect
					for (l = 0; l < _U_r[m]; l++) Viu[k] += _Vi[k][_U_p[m][l]] * _U_v[m][l];
				}

				lamda = 1.0;
				for (k = 0; k < _U_r[m]; k++) lamda += _U_v[m][k] * Viu[_U_p[m][k]];
				for (k = 0; k < _n; k++)
				{
					d_buf = Viu[k] / lamda;
					for (l = k; l < _n; l++) _Vi[l][k] = _Vi[k][l] -= d_buf * Viu[l];
				}
			}
		}
	}
	

	HR_equation();

	for (j = 0; j < _r + 1; j++)
	{
		std::cout << _var_comp[j] << "\t";
// 		long seed = -5251949;
// 		if (_var_comp[j] < 1.0e-08*fabs(_Y[0])) _var_comp[j] = fabs(_Y[0])*0.00001*StatFunc::UniformDev(0.0, 1.0, seed);
	}
	std::cout << std::endl;
}

void MINQUE::MINQUE_alpha(const vector<double>& prior)
{
	long j = 0;

	_alpha.clear();
	_alpha = prior;

	// calculate Vi
	morrision_Vi(prior);

	HR_equation();

	for (j = 0; j < _r + 1; j++)
	{
		std::cout << _var_comp[j] << "\t";
//		long seed = -5251949;
//		if (_var_comp[j] < 1.0e-08*fabs(_Y[0])) _var_comp[j] = fabs(_Y[0])*0.00001*StatFunc::UniformDev(0.0, 1.0, seed);
	}
}

void MINQUE::morrision_Vi(const vector<double>& prior)
{
	long i = 0, j = 0, k = 0, l = 0, m = 0;
	double d_buf = 0.0;
	double ut_Vi_u = 0.0;
	vector<double> Vi_u(_n);

	// initialize Vi matrix
	init_Vi(prior);

	for (i = 0; i < _r; i++)
	{
		for (j = 0; j < _Uu_c[i]; j++, m++)
		{
			for (k = 0; k < _n; k++)
			{
				Vi_u[k] = 0.0;
				for (l = 0; l < _U_r[m]; l++) Vi_u[k] += _Vi[k][_U_p[m][l]] * _U_v[m][l];
			}

			ut_Vi_u = 0.0;
			for (k = 0; k < _U_r[m]; k++) ut_Vi_u += _U_v[m][k] * Vi_u[_U_p[m][k]];
			ut_Vi_u = prior[i] / (1.0 + prior[i] * ut_Vi_u);
			for (k = 0; k < _n; k++)
			{
				d_buf = Vi_u[k] * ut_Vi_u;
				for (l = k; l < _n; l++) _Vi[l][k] = _Vi[k][l] -= d_buf * Vi_u[l];
			}
		}
	}
}

void MINQUE::HR_equation()
{
	long i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, h = 0;
	double d_buf = 0.0;
	vector< vector<double> > Q_Uv;

	// calculate _Q
	calcu_Q();

	// initialization
	_Yt_Q_U.clear();
	_Yt_Q_U.resize(_r + 1);
	_q.clear();
	_q.resize(_r + 1);
	_H.clear();
	_H.resize(_r + 1);
	for (i = 0; i < _r + 1; i++) _H[i].resize(_r + 1);

	for (i = 0; i < _r + 1; i++)
	{
		calcu_Q_U(Q_Uv, i, m);

		calcu_q(Q_Uv, i);

		// calculate _H
		n = m - _Uu_c[i];
		for (j = i; j < _r + 1; j++)
		{
			if (j < _r)
			{
				for (k = 0; k < _Uu_c[j]; k++, n++)
				{
					for (l = 0; l < _Uu_c[i]; l++)
					{
						d_buf = 0.0;
						for (h = 0; h < _U_r[n]; h++) d_buf += _U_v[n][h] * Q_Uv[l][_U_p[n][h]];
						_H[i][j] += d_buf * d_buf;
					}
				}
			}
			else
			{
				for (k = 0; k < _Uu_c[i]; k++)
				{
					for (l = 0; l < _n; l++) _H[i][j] += Q_Uv[k][l] * Q_Uv[k][l];
				}
			}
			_H[j][i] = _H[i][j];
		}
	}
	// inversion
	MatFunc::LUinversion(_H, 1.0e-08);

	// calculate variance componments
	_var_comp.clear();
	_var_comp.resize(_r + 1);
	for (j = 0; j < _r + 1; j++)
	{
		for (k = 0; k < _r + 1; k++) _var_comp[j] += _H[j][k] * _q[k];
	}
}



void MINQUE::MIVQUE()
{
	Minque_1();
	vector<double> var_comp = _var_comp;
	MINQUE_alpha(var_comp);

}

MINQUE::~MINQUE()
{
}

void MINQUE::calcu_Q()
{
	long i = 0, j = 0, k = 0, l = 0;
	double d_buf = 0.0;

	// calculate Q
	if (_Q.size() != _n)
	{
		_Q.clear();
		_Q.resize(_n);
		for (i = 0; i < _n; i++) _Q[i].resize(_n);
	}
// 	if (_X_c > 1)
// 	{
// 		//calculate the inverse of X*Vi*X
// 		Xt_Vi_X_inverse();
// 
// 		for (i = 0; i < _n; i++)
// 		{
// 			for (j = 0; j <= i; j++)
// 			{
// 				_Q[i][j] = 0.0;
// 				for (k = 0; k < _X_c; k++)
// 				{
// 					d_buf = 0.0;
// 					for (l = 0; l < _X_c; l++) d_buf += _Vi_X[i][l] * _Xt_Vi_X[l][k];
// 					_Q[i][j] += d_buf * _Vi_X[j][k];
// 				}
// 				_Q[i][j] = _Vi[i][j] - _Q[i][j];
// 				_Q[j][i] = _Q[i][j]; // because Q is a symmetrix matrix
// 			}
// 		}
// 	}
	double sum = 0.0;
	for (i = 0; i < _n; i++)
	{
		for (j = 0; j < _n; j++) sum += _Vi[i][j];
	}

	vector<double> vd_buf(_n);
	for (i = 0; i < _n; i++)
	{
		vd_buf[i] = 0.0;
		for (j = 0; j < _n; j++) vd_buf[i] += _Vi[i][j];
	}

	for (i = 0; i < _n; i++)
	{
		d_buf = vd_buf[i] / sum;
		for (j = i; j < _n; j++) _Q[j][i] = _Q[i][j] = _Vi[i][j] - d_buf * vd_buf[j];
	}

}

void MINQUE::calcu_Q_U(vector< vector<double> > &Q_Uv, long u, long &m)
{
	long j = 0, k = 0, l = 0;

	Q_Uv.clear();
	if (u < _r)
	{
		Q_Uv.resize(_Uu_c[u]);
		for (j = 0; j < _Uu_c[u]; j++, m++)
		{
			Q_Uv[j].resize(_n);
			for (k = 0; k < _n; k++)
			{
				for (l = 0; l < _U_r[m]; l++) Q_Uv[j][k] += _Q[k][_U_p[m][l]] * _U_v[m][l];
			}
		}
	}
	else Q_Uv = _Q;
}

void MINQUE::calcu_q(const vector< vector<double> > &Q_Uv, long u)
{
	long j = 0, k = 0;

	_q[u] = 0.0;
	_Yt_Q_U[u].clear();
	_Yt_Q_U[u].resize(_Uu_c[u]);
	for (j = 0; j < _Uu_c[u]; j++)
	{
		for (k = 0; k < _n; k++) _Yt_Q_U[u][j] += _Y[k] * Q_Uv[j][k];
		_q[u] += _Yt_Q_U[u][j] * _Yt_Q_U[u][j];
	}
}


