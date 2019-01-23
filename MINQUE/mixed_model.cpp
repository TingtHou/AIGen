#include "mixed_model.h"

mixed_model::mixed_model()
{
	_seed=-5251949;
	_n=0;
	_X_c=0;
	_p=0;
	_r=0;
	_U_c=0;
	_var_est_mtd=MINQ1_mtd;
	_fix_est_mtd=GLS_mtd;
	_rand_pre_mtd=LUP_mtd;
}

mixed_model::~mixed_model()
{
	
}

void mixed_model::import_data(const vector<double> &Y)
{
	if(Y.empty()) throw("\"Y\" is a empty vector! mixed_model::import");
	_Y.clear();
	_Y=Y;
	_n=_Y.size();
	
	// initialize _Uu_c
	_Uu_c.clear();
	_Uu_c.push_back(_n); // for matrix I
}

void mixed_model::import_data(const vector< vector<double> > &Y)
{

}

void mixed_model::import_cross(const long cross_cate)
{
	
}

void mixed_model::import_exp_des(const long rep_num, const long env_num)
{

}

void mixed_model::push_back_HomoEffects(const vector< vector< vector<long> > > &FixEffBuf, long QTLNum, long EffectSN)
{

}

void mixed_model::push_back_X(const vector<double> &X_col)
{				
	if(X_col.size()!=_n) throw("The size of \"X_col\" and \"_n\" mismatch! mixed_model::push_back_X");
	
	_X.push_back(X_col);
	_X_c=_X.size(); // update _X_c
}

void mixed_model::push_back_X(const vector< vector<double> > &X)
{
	long X_c=X.size();
	if(X_c==0) throw("\"X\" is empty! mixed_model::push_back_X");
	if(X[0].size()!=_n) throw("The size of \"X\" and \"_n\" mismatch! mixed_model::push_back_X");
	
	long i=0;
	for(i=0; i<X_c; i++) _X.push_back(X[i]);
	_X_c=_X.size(); // update _X_c
}

void mixed_model::push_back_U(const vector< vector<double> > &Ui)
{
	long Ui_c=Ui.size();
	if(Ui_c==0) throw("Empty Ui! mixed_model::push_back_U");
	if(Ui[0].size()!=_n) throw("The row number of \"Ui\" and \"_n\" mismatch! mixed_model::push_back_U");
	_U.insert(_U.end(), Ui.begin(), Ui.end());
	_Uu_c.insert(_Uu_c.end()-1, Ui_c);
	_U_c=_U.size(); // update _U_c
	_r=_Uu_c.size()-1; // update _r
	
	long i=0, j=0, k=0;
	vector< vector<double> > U_v(Ui_c);
	vector< vector<long> > U_p(Ui_c);
	vector<long> U_r(Ui_c);
	for(i=0; i<Ui_c; i++)
	{
		for(j=0; j<_n; j++)
		{
			if( CommFunc::FloatNotEqual(Ui[i][j], 0.0) )
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

void mixed_model::pop_back_X(long size)
{
	if(size>_X_c) throw("\"size\" exceed the size of \"_X\"! mixed_model::pop_back_X");
	
	long i=0;
	for(i=0; i<size; i++) _X.pop_back();
	_X_c=_X.size(); // update _X_c
}

void mixed_model::pop_back_U(long size)
{
	if(size>_r) throw("\"size\" exceed the size of \"_U\"! mixed_model::pop_back_U");
	
	long i=0;
	long erased_col_num=0;
	for(i=0; i<size; i++)
	{
		erased_col_num+=*(_Uu_c.end()-2);
		_Uu_c.erase(_Uu_c.end()-2);
	}
	
	for(i=0; i<erased_col_num; i++)
	{
		_U.pop_back();
		_U_v.pop_back();
		_U_p.pop_back();
		_U_r.pop_back();
	}
	_U_c=_U.size(); // update _U_c
	_r=_Uu_c.size()-1; // update _r
}

void mixed_model::sparse_mat()
{
	long i=0, j=0;
	
	_U_r.clear();
	_U_p.clear();
	_U_v.clear();
	_U_r.resize(_U_c);
	_U_p.resize(_U_c);
	_U_v.resize(_U_c);	
	for(i=0; i<_U_c; i++)
	{
		for(j=0; j<_n; j++)
		{
			if( CommFunc::FloatNotEqual(_U[i][j], 0.0) )
			{
				_U_v[i].push_back(_U[i][j]);
				_U_p[i].push_back(j);
				_U_r[i]++;
			}
		}
	}
}

void mixed_model::Xt_Vi_X_inverse()
{
	long i=0, j=0, k=0;
	
	// calculate Vi*X
	_Vi_X.clear();
	_Vi_X.resize(_n);
	for(i=0; i<_n; i++)
	{
		_Vi_X[i].resize(_X_c);
		for(j=0; j<_X_c; j++)
		{
			for(k=0; k<_n; k++) _Vi_X[i][j]+=_Vi[i][k]*_X[j][k];
		}
	}
	
	//calculate Xt*Vi*X
	_Xt_Vi_X.clear();
	_Xt_Vi_X.resize(_X_c);
	for(i=0; i<_X_c; i++)
	{
		_Xt_Vi_X[i].resize(_X_c);
		for(j=0; j<=i; j++)
		{
			for(k=0; k<_n; k++) _Xt_Vi_X[i][j]+=_Vi_X[k][j]*_X[i][k];
			_Xt_Vi_X[j][i]=_Xt_Vi_X[i][j]; //because Xt*Vi*X is a symmetrix matrix 
		}
	}
	
	//calculate the inverse of X*Vi*X
	if( !_Xt_Vi_X.empty() ) MatFunc::SVDInversion(_Xt_Vi_X, 1.0e-08);
}

void mixed_model::calcu_Q()
{
	long i=0, j=0, k=0, l=0;
	double d_buf=0.0;
	
	// calculate Q
	if( _Q.size() != _n )
	{
		_Q.clear();
		_Q.resize(_n);
		for(i=0; i<_n; i++) _Q[i].resize(_n);
	}
	if(_X_c>1)
	{
		//calculate the inverse of X*Vi*X
		Xt_Vi_X_inverse();
		
		for(i=0; i<_n; i++)
		{
			for(j=0; j<=i; j++)
			{
				_Q[i][j]=0.0;
				for(k=0; k<_X_c; k++)
				{
					d_buf=0.0;
					for(l=0; l<_X_c; l++) d_buf+=_Vi_X[i][l]*_Xt_Vi_X[l][k];
					_Q[i][j]+=d_buf*_Vi_X[j][k];
				}
				_Q[i][j]=_Vi[i][j]-_Q[i][j];
				_Q[j][i]=_Q[i][j]; // because Q is a symmetrix matrix
			}
		}
	}
	else
	{
		double sum=0.0;
		for(i=0; i<_n; i++)
		{
			for(j=0; j<_n; j++) sum+=_Vi[i][j];
		}
		
		vector<double> vd_buf(_n);
		for(i=0; i<_n; i++)
		{
			vd_buf[i]=0.0;
			for(j=0; j<_n; j++) vd_buf[i]+=_Vi[i][j];
		}
		
		for(i=0; i<_n; i++)
		{
			d_buf=vd_buf[i]/sum;
			for(j=i; j<_n; j++) _Q[j][i]=_Q[i][j]=_Vi[i][j]-d_buf*vd_buf[j];
		}
	}
}

void mixed_model::calcu_Q_U(vector< vector<double> > &Q_Uv, long u, long &m)
{
	long j=0, k=0, l=0;
	
	Q_Uv.clear();
	if(u<_r)
	{
		Q_Uv.resize(_Uu_c[u]);
		for(j=0; j<_Uu_c[u]; j++, m++)
		{
			Q_Uv[j].resize(_n);
			for(k=0; k<_n; k++)
			{
				for(l=0; l<_U_r[m]; l++) Q_Uv[j][k]+=_Q[k][ _U_p[m][l] ]*_U_v[m][l];
			}
		}
	}
	else Q_Uv=_Q;
}

void mixed_model::calcu_q(const vector< vector<double> > &Q_Uv, long u)
{
	long j=0, k=0;
	
	_q[u]=0.0;
	_Yt_Q_U[u].clear();
	_Yt_Q_U[u].resize(_Uu_c[u]);
	for(j=0; j<_Uu_c[u]; j++)
	{
		for(k=0; k<_n; k++) _Yt_Q_U[u][j]+=_Y[k]*Q_Uv[j][k];
		_q[u]+=_Yt_Q_U[u][j]*_Yt_Q_U[u][j];
	}
}

void mixed_model::HR_equation()
{
	long i=0, j=0, k=0, l=0, m=0, n=0, h=0;
	double d_buf=0.0;
	vector< vector<double> > Q_Uv;
	
	// calculate _Q
	calcu_Q();
	
	// initialization
	_Yt_Q_U.clear();
	_Yt_Q_U.resize(_r+1);
	_q.clear(); 
	_q.resize(_r+1);
	_H.clear();
	_H.resize(_r+1);
	for(i=0; i<_r+1; i++) _H[i].resize(_r+1);
	
	for(i=0; i<_r+1; i++)
	{
		calcu_Q_U(Q_Uv, i, m);
		
		calcu_q(Q_Uv, i);
		
		// calculate _H
		n=m-_Uu_c[i];
		for(j=i; j<_r+1; j++)
		{
			if(j<_r)
			{
				for(k=0; k<_Uu_c[j]; k++, n++)
				{
					for(l=0; l<_Uu_c[i]; l++)
					{
						d_buf=0.0;
						for(h=0; h<_U_r[n]; h++) d_buf+=_U_v[n][h]*Q_Uv[l][_U_p[n][h]];
						_H[i][j]+=d_buf*d_buf;
					}
				}
			}
			else
			{
				for(k=0; k<_Uu_c[i]; k++)
				{
					for(l=0; l<_n; l++) _H[i][j]+=Q_Uv[k][l]*Q_Uv[k][l];
				}
			}
			_H[j][i]=_H[i][j];
		}
	}
	
	MatFunc::SVDInversion(_H, 1.0e-08);
	
	// calculate variance componments
	_var_comp.clear();
	_var_comp.resize(_r+1);
	for(j=0; j<_r+1; j++)
	{
		for(k=0; k<_r+1; k++) _var_comp[j]+=_H[j][k]*_q[k];
	}
}

void mixed_model::init_Vi()
{
	long i=0, j=0;
	
	if(_Vi.size()!=_n)
	{
		_Vi.clear();
		_Vi.resize(_n);
		for(i=0; i<_n; i++) _Vi[i].resize(_n);
		for(i=0; i<_n; i++) _Vi[i][i]=1.0;
	}
	else
	{
		for(i=0; i<_n-1; i++)
		{
			_Vi[i][i]=1.0;
			for(j=i+1; j<_n; j++) _Vi[i][j]=_Vi[j][i]=0.0;
		}
		_Vi[_n-1][_n-1]=1.0;
	}
}

void mixed_model::init_Vi(const vector<double> &prior)
{
	long i=0, j=0;
	
	if(_Vi.size()!=_n)
	{
		_Vi.clear();
		_Vi.resize(_n);
		for(i=0; i<_n; i++) _Vi[i].resize(_n);
		for(i=0; i<_n; i++) _Vi[i][i]=1.0/prior[_r];
	}
	else
	{
		for(i=0; i<_n-1; i++)
		{
			_Vi[i][i]=1.0/prior[_r];
			for(j=i+1; j<_n; j++) _Vi[i][j]=_Vi[j][i]=0.0;
		}
		_Vi[_n-1][_n-1]=1.0/prior[_r];
	}
}

void mixed_model::MINQUE_0()
{
	long i=0, j=0;
	
	_alpha.clear();
	_alpha.resize(_r+1);
	for(i=0; i<_r; i++) _alpha[i]=0.0;
	_alpha[_r]=1.0;
	
	init_Vi();
	
	HR_equation();
	
	for(j=0; j<_r+1; j++)
	{
		long seed=-5251949;
		if( _var_comp[j]<1.0e-08*fabs(_Y[0]) ) _var_comp[j]=fabs(_Y[0])*0.00001*StatFunc::UniformDev(0.0, 1.0, seed);
	}
}

void mixed_model::MINQUE_1()
{
	long i=0, j=0, k=0, l=0, m=0;
	double d_buf=0.0;
	double lamda=0.0;
	vector<double> Viu(_n);
	
	_alpha.clear();
	_alpha.resize(_r+1);
	for(i=0; i<_r+1; i++) _alpha[i]=1.0;
	
	init_Vi();
	
	for(i=0; i<_r; i++)
	{
		for(j=0; j<_Uu_c[i]; j++, m++)
		{
			for(k=0; k<_n; k++)
			{
				Viu[k]=0.0;
				for(l=0; l<_U_r[m]; l++) Viu[k]+=_Vi[k][_U_p[m][l]]*_U_v[m][l];
			}
			
			lamda=1.0;
			for(k=0; k<_U_r[m]; k++) lamda+=_U_v[m][k]*Viu[_U_p[m][k]];
			for(k=0; k<_n; k++)
			{
				d_buf=Viu[k]/lamda;
				for(l=k; l<_n; l++) _Vi[l][k]=_Vi[k][l]-=d_buf*Viu[l];
			}
		}
	}
	
	HR_equation();
	
	for(j=0; j<_r+1; j++)
	{
		long seed=-5251949;
		if( _var_comp[j]<1.0e-08*fabs(_Y[0]) ) _var_comp[j]=fabs(_Y[0])*0.00001*StatFunc::UniformDev(0.0, 1.0, seed);
	}
}

void mixed_model::morrision_Vi(const vector<double> &prior)
{
	long i=0, j=0, k=0, l=0, m=0;
	double d_buf=0.0;
	double ut_Vi_u=0.0;
	vector<double> Vi_u(_n);
	
	// initialize Vi matrix
	init_Vi(prior);
	
	for(i=0; i<_r; i++)
	{
		for(j=0; j<_Uu_c[i]; j++, m++)
		{
			for(k=0; k<_n; k++)
			{
				Vi_u[k]=0.0;
				for(l=0; l<_U_r[m]; l++) Vi_u[k]+=_Vi[k][_U_p[m][l]]*_U_v[m][l];
			}
			
			ut_Vi_u=0.0;
			for(k=0; k<_U_r[m]; k++) ut_Vi_u+=_U_v[m][k]*Vi_u[_U_p[m][k]];
			ut_Vi_u=prior[i]/(1.0+prior[i]*ut_Vi_u);
			for(k=0; k<_n; k++)
			{
				d_buf=Vi_u[k]*ut_Vi_u;
				for(l=k; l<_n; l++) _Vi[l][k]=_Vi[k][l]-=d_buf*Vi_u[l];
			}
		}
	}
}

void mixed_model::MINQUE_alpha(const vector<double> &prior)
{
	long j=0;
	
	_alpha.clear();
	_alpha=prior;
	
	// calculate Vi
	morrision_Vi(prior);
	
	HR_equation();
	
	for(j=0; j<_r+1; j++)
	{
		long seed=-5251949;
		if( _var_comp[j]<1.0e-08*fabs(_Y[0]) ) _var_comp[j]=fabs(_Y[0])*0.00001*StatFunc::UniformDev(0.0, 1.0, seed);
	}
}

void mixed_model::MIVQUE()
{
	MINQUE_1();
	vector<double> var_comp=_var_comp;
	MINQUE_alpha(var_comp);
}

void mixed_model::REML()
{
	long j=0, k=0;
	bool converge=false;
	
	vector<double> var_comp(_r+1);
	for(j=0; j<_r+1; j++) var_comp[j]=1.0;
	
	for(k=0; k<200; k++)
	{	
		MINQUE_alpha(var_comp);
	
		converge=true;
		for(j=0; j<_r+1; j++)
		{
			if( _var_comp[j]>1.0e-08*fabs(_Y[0]) && fabs((_var_comp[j]-var_comp[j])/var_comp[j]) > 0.01 ) converge=false;
		}
		if(converge) break;
		var_comp=_var_comp;
	}
}

void mixed_model::EM()
{
	bool converge=false;
	long j=0, k=0, l=0, m=0;
	double d_buf=0.0;
	vector< vector<double> > Q_Uv;
	vector< vector<double> > Ut_Vi_U;
	
	vector<double> var_comp(_r+1);
	for(j=0; j<_r+1; j++) var_comp[j]=1.0;
	
	_Yt_Q_U.clear();
	_Yt_Q_U.resize(_r+1);
	_q.clear(); 
	_q.resize(_r+1);
	_H.clear();
	_H.resize(_r+1);
	_var_comp.clear();
	_var_comp.resize(_r+1);	
	for(k=0; k<200; k++) // restrict the iteration times
	{
		morrision_Vi(var_comp);
		calcu_Q();
		for(j=0, m=0; j<_r+1; j++)
		{
			if(j<_r) Ut_M_U(_Vi, Ut_Vi_U, j, m);
			else
			{
				Ut_Vi_U.clear();
				Ut_Vi_U=_Vi;
			}
			
			calcu_Q_U(Q_Uv, j, m);
			calcu_q(Q_Uv, j);
			
			d_buf=var_comp[j]*var_comp[j]*_q[j];
			for(l=0; l<_Uu_c[j]; l++)
			{
				d_buf+=var_comp[j]*(1.0-var_comp[j]*Ut_Vi_U[l][l]);
			}
			_var_comp[j]=d_buf/(double)_Uu_c[j];
		}
		
		// determine whether the EM process i converge or not
		converge=true;
		for(j=0; j<_r+1; j++) 
		{
			if( fabs((_var_comp[j]-var_comp[j])/var_comp[j]) > 0.01 ) converge=false;
		}
		if(converge) break;
		
		//update var_comp
		var_comp=_var_comp;
	}
}

void mixed_model::Ut_M_U(const vector< vector<double> > &M, vector< vector<double> > &Ut_M_U, long u, long m)
{
	long i=0, j=0, k=0, l=0;
	
	// calculate M_U
	vector< vector<double> > M_U(_Uu_c[u]);
	for(i=0, l=m; i<_Uu_c[u]; i++, l++)
	{
		M_U[i].resize(_n);
		for(j=0; j<_n; j++)
		{
			for(k=0; k<_U_r[l]; k++) M_U[i][j]+=M[j][ _U_p[l][k] ]*_U_v[l][k];
		}
	}
	
	// calculate Ut_M_U
	Ut_M_U.clear();
	Ut_M_U.resize(_Uu_c[u]);
	for(i=0; i<_Uu_c[u]; i++) Ut_M_U[i].resize(_Uu_c[u]);
	for(i=0, l=m; i<_Uu_c[u]; i++, l++)
	{
		for(j=i; j<_Uu_c[u]; j++)
		{
			for(k=0; k<_U_r[l]; k++) Ut_M_U[i][j]+=M_U[j][ _U_p[l][k] ]*_U_v[l][k];
			Ut_M_U[j][i]=Ut_M_U[i][j]; 
		}
	}
}

void mixed_model::var_est()
{
	if(_var_est_mtd==MINQ0_mtd) MINQUE_0();
	else if(_var_est_mtd==MINQ1_mtd) MINQUE_1();
	else if(_var_est_mtd==REML_mtd) REML();
	else if(_var_est_mtd==EM_mtd) EM();
	else if(_var_est_mtd==MIVQ_mtd) MIVQUE();
	else throw("Illegal method option of variance estimation!");
}

void mixed_model::GLS(bool calcu_prob)
{
	long j=0, k=0, l=0;
	double d_buf=0.0;
	
	if(_var_est_mtd!=REML_mtd && _var_est_mtd!=EM_mtd) morrision_Vi(_var_comp);

	Xt_Vi_X_inverse();
	
	_b.clear();
	_b.resize(_X_c);
	for(j=0; j<_X_c; j++)
	{
		for(k=0; k<_n; k++)
		{
			d_buf=0.0;
			for(l=0; l<_X_c; l++) d_buf+=_Xt_Vi_X[j][l]*_Vi_X[k][l];
			_b[j]+=d_buf*_Y[k];		
		}
	}

	// calculate the probability of significance for all the fixed effects
	if(calcu_prob)
	{
		_SE_b.clear();
		_SE_b.resize(_X_c);
		_t_val_b.clear();
		_t_val_b.resize(_X_c);
		_prob_b.clear();
		_prob_b.resize(_X_c);
		for(j=0; j<_X_c; j++)
		{
			if(_Xt_Vi_X[j][j]<0.0) _SE_b[j]=0.0;
			else _SE_b[j]=sqrt(_Xt_Vi_X[j][j]);
			
			if( CommFunc::FloatEqual(_b[j], 0.0) ) { _SE_b[j]=0.0; _t_val_b[j]=0.0; _prob_b[j]=1.0; continue; }
			else if( CommFunc::FloatEqual(_SE_b[j], 0.0) ) _t_val_b[j]=103.56;
			else  _t_val_b[j]= _b[j]/_SE_b[j];
			_prob_b[j] = StatFunc::t_prob( (double)(_n-_X_c), _t_val_b[j]);
		}
	}
}

void mixed_model::calcu_V(vector< vector<double> > &V, const vector<double> &sigma_square) const
{
	long i=0, j=0, k=0, l=0, m=0, n=0;
	
	if(V.size()!=_n)
	{
		V.clear();
		V.resize(_n);
		for(i=0; i<_n; i++)
		{
			V[i].resize(_n);
			V[i][i]=sigma_square[_r];
		}
	}
	else
	{
		for(i=0; i<_n-1; i++)
		{
			V[i][i]=sigma_square[_r];
			for(j=i+1; j<_n; j++) V[j][i]=V[i][j]=0.0;
		}
		V[_n-1][_n-1]=sigma_square[_r];
	}
	
	m=0;
	for(i=0; i<_r; i++)
	{
		for(j=0; j<_n; j++)
		{
			for(k=0; k<=j; k++)
			{
				for(l=0, n=m; l<_Uu_c[i]; l++, n++) V[j][k]+=_U[n][j]*_U[n][k]*sigma_square[i];
				V[k][j]=V[j][k];
			}
		}
		m+=_Uu_c[i];
	}
}

void mixed_model::OLS(bool calcu_prob)
{
	long i=0, j=0, k=0, l=0;
	vector< vector<double> > buf;
	vector< vector<double> > var_b;
	vector< vector<double> > Xt_X;
	
	MatFunc::At_A(_X, Xt_X);
	if(!Xt_X.empty()) MatFunc::SVDInversion(Xt_X, 1.0e-08);
	
	_b.clear();
	_b.resize(_X_c);
	buf.clear();
	buf.resize(_X_c);
	for(j=0; j<_X_c; j++)
	{
		buf[j].resize(_n);
		for(k=0; k<_n; k++)
		{
			for(l=0; l<_X_c; l++) buf[j][k]+=Xt_X[j][l]*_X[l][k];
			_b[j]+=buf[j][k]*_Y[k];
		}
	}
	
	// calculate the probability of significance for all the fixed effects
	if(calcu_prob)
	{
		calcu_V(_V, _var_comp);
		MatFunc::At_B_A(buf, _V, var_b); 
		_SE_b.clear();
		_SE_b.resize(_X_c);
		_t_val_b.clear();
		_t_val_b.resize(_X_c);
		_prob_b.clear();
		_prob_b.resize(_X_c);
		for(j=0; j<_X_c; j++)
		{
			if( var_b[j][j] < 0.0 ) _SE_b[j]=0.0;
			else _SE_b[j]=sqrt(var_b[j][j]);
			
			if( CommFunc::FloatEqual(_b[j], 0.0) ) { _SE_b[j]=0.0; _t_val_b[j]=0.0; _prob_b[j]=1.0; continue; }
			else if( CommFunc::FloatEqual(_SE_b[j], 0.0) ) _t_val_b[j]=103.56;
			else  _t_val_b[j]= _b[j]/_SE_b[j];
			_prob_b[j] = StatFunc::t_prob( (double)(_n-_X_c), _t_val_b[j]);
		}
	}
}

void mixed_model::fixed_eff_est(bool calcu_prob)
{
	if(_fix_est_mtd==OLS_mtd) OLS(calcu_prob);
	else if(_fix_est_mtd=GLS_mtd) GLS(calcu_prob);
	else throw("Illegal method option for fixed effect estimation!");
}

void mixed_model::BLUP()
{
	long i=0, j=0, k=0, m=0;
	vector< vector<double> > Q_Uv;
	
	if(_var_est_mtd!=REML_mtd && _var_est_mtd!=EM_mtd)
	{
		morrision_Vi(_var_comp);
		calcu_Q();
	}
	
	_e.clear();
	_e.resize(_r+1);
	for(i=0; i<_r+1; i++)
	{
		_e[i].resize(_Uu_c[i]);
		calcu_Q_U(Q_Uv, i, m);
		for(j=0; j<_Uu_c[i]; j++)
		{
			for(k=0; k<_n; k++) _e[i][j]+=_var_comp[i]*Q_Uv[j][k]*_Y[k];
		}
	}
}

void mixed_model::LUP()
{
	long i=0, j=0;
	
	_e.clear();
	_e.resize(_r+1);
	for(i=0; i<_r+1; i++)
	{
		_e[i].resize(_Uu_c[i]);
		for(j=0; j<_Uu_c[i]; j++) _e[i][j]=_alpha[i]*_Yt_Q_U[i][j]; 
	}
}

void mixed_model::AUP()
{
	long i=0, j=0, k=0;
	
	vector<double> karba(_r+1);
	for(i=0; i<_r+1; i++)
	{
		for(j=0; j<_Uu_c[i]; j++) karba[i]+=_Yt_Q_U[i][j]*_Yt_Q_U[i][j];
		karba[i]=sqrt((_Uu_c[i]-1)*_var_comp[i]/karba[i]) / _alpha[i];
	}
	
	LUP();
	
	for(i=0; i<_r+1; i++)
	{
		for(j=0; j<_Uu_c[i]; j++) _e[i][j]*=karba[i];
	}
}

void mixed_model::random_eff_pre()
{
	if(_rand_pre_mtd==BLUP_mtd) BLUP();
	else if(_rand_pre_mtd==LUP_mtd) LUP();
	else if(_rand_pre_mtd==AUP_mtd) AUP();
	else throw("Illegal method option for random prediction!");
}

double mixed_model::calcu_L()
{
	double d_buf=0.0;
	
	// calculate (y-Xb)t*Vi*(y-Xb)
	long i=0;
	vector<double> Xb;
	vector<double> Y_Xb(_n);
	
	var_est();
	fixed_eff_est(true);
	MatFunc::mat_multiply_vec(_X, _b, Xb);
	for(i=0; i<_n; i++) Y_Xb[i]=_Y[i]-Xb[i];
	d_buf=0.0-MatFunc::At_B_A(Y_Xb, _Vi);
	d_buf+=calcu_lnV();
	
	return d_buf;
}

double mixed_model::calcu_lnV()
{
	long i=0;
	vector<long> indx(_n);
	double d=0.0;
	
	MatFunc::ludcmp(_Vi, indx, d);
	d=0.0;
	for(i=0; i<_n; i++)
	{ 
		_Vi[i][i]=fabs(_Vi[i][i]);
		d+=log(_Vi[i][i]);
	}
	
	return d;
}

void mixed_model::set_method(long var_est_mtd, long fix_est_mtd, long rand_pre_mtd)
{
	_var_est_mtd=var_est_mtd;
	_fix_est_mtd=fix_est_mtd;
	_rand_pre_mtd=rand_pre_mtd;
}

void mixed_model::jknf_test(bool jack_var_comp, bool jack_b, bool jack_e, long jack_size)
{
	long i=0, j=0, k=0, l=0, g=0;
	long n=_n;
	double d_g=0.0, ave=0.0;
	vector<double> Y=_Y;
	vector< vector<double> > X=_X; 
	vector< vector<double> > U=_U;
	vector<double> var_comp;
	vector< vector<double> > pseduo_var_comp;
	vector<double> b;
	vector< vector<double> > pseduo_b;
	vector< vector<double> > e;
	vector< vector< vector<double> > > pseduo_e;
	
	// pre-jackknife
	var_est();
	
	if(jack_var_comp) var_comp=_var_comp;
	
	if(jack_b)
	{
		fixed_eff_est(false);
		b=_b;
	}
	else fixed_eff_est(true);
	
	if(jack_e)
	{
		random_eff_pre();
		e=_e;
	}
	
	// Set total Process for QTModel project
//	_process_point=1;
//	g_progressBar->setTotalSteps(int(n/jack_size)+1);

	// Set total Process for QTLNetwork project
	#ifdef _QTLNETWORK_GUI
	long ProcessPoint=0, TtlProcess=0;
	for(i=0; i<n; i+=jack_size) TtlProcess++;
	SendMessage(g_ParentWnd, WM_UPDATE_PROGRESSBAR, 1, TtlProcess);
	#endif

	for(i=0; i<n; i+=jack_size)
	{
		if(_Y.begin()+i+jack_size < _Y.end() ) _Y.erase(_Y.begin()+i, _Y.begin()+i+jack_size);
		else _Y.erase(_Y.begin()+i, _Y.end());
		_n=_Y.size(); // update _n
		
		for(j=0; j<_X_c; j++)
		{
			if(_X[j].begin()+i+jack_size < _X[j].end() ) _X[j].erase(_X[j].begin()+i, _X[j].begin()+i+jack_size);
			else _X[j].erase(_X[j].begin()+i, _X[j].end());
		}
		
		for(j=0; j<_U_c; j++)
		{
			if(_U[j].begin()+i+jack_size < _U[j].end() ) _U[j].erase(_U[j].begin()+i, _U[j].begin()+i+jack_size);
			else _U[j].erase(_U[j].begin()+i, _U[j].end());
		}
		_Uu_c[_r]=_n; // change _Uu_c, because of the jackknife procedure		
		
		sparse_mat();
		
		// check whether a effect is missing, if so drop this sample
		bool drop_flag=true;
		for(j=0; j<_X.size(); j++)
		{
			for(k=0; k<_n; k++) 
			{
				if( CommFunc::FloatNotEqual(_X[j][k], 0.0) ) { drop_flag=false; break; }
			}
		}
		for(j=0; j<_U_r.size(); j++)
		{
			if(_U_r[j]!=0) { drop_flag=false; break; }
		}
		
		if(!drop_flag)
		{
			var_est();
			if(jack_var_comp) pseduo_var_comp.push_back(_var_comp);
			if(jack_b)
			{
				fixed_eff_est(false);
				pseduo_b.push_back(_b);
			}
			if(jack_e)
			{
				random_eff_pre();
				pseduo_e.push_back(_e);
			}
			g++;
		}
		
		_Y.clear();
		_X.clear();
		_U.clear();
		_Y=Y;
		_X=X;
		_U=U;
		
		// Set Process for QTModel project
//		g_progressBar->setProgress(_process_point++);

		// Set Process for QTLNetwork project
		#ifdef _QTLNETWORK_GUI
		SendMessage(g_ParentWnd,WM_UPDATE_PROGRESSBAR,0,++ProcessPoint);
		#endif

	}
	if(g<5) throw("Too few jackknife sampling times!");
	d_g=(double)g;
	
	// recovery
	if(jack_var_comp) _var_comp=var_comp; // recovery _var_comp
	if(jack_b) _b=b; // recovery _b
	if(jack_e) _e=e; // recovery _e
	
	if(jack_var_comp)
	{
		_SE_var_comp.clear();
		_SE_var_comp.resize(_r+1);
		_prob_var_comp.clear();
		_prob_var_comp.resize(_r+1);
		for(i=0; i<_r+1; i++)
		{
			ave=0.0;
			for(j=0; j<g; j++) ave+=pseduo_var_comp[j][i];
			ave/=d_g;
			for(j=0; j<g; j++) _SE_var_comp[i]+=(pseduo_var_comp[j][i]-ave)*(pseduo_var_comp[j][i]-ave);
			_SE_var_comp[i]=sqrt( _SE_var_comp[i]*(d_g-1.0)/d_g );
			
			if( CommFunc::FloatEqual(var_comp[i], 0.0) ) { _SE_var_comp[i]=0.0; _prob_var_comp[i]=1.0; continue; }
			else if( CommFunc::FloatEqual(_SE_var_comp[i], 0.0) ) _prob_var_comp[i]=103.56;
			else _prob_var_comp[i]=fabs( (d_g*var_comp[i]-(d_g-1.0)*ave) / _SE_var_comp[i] );
			_prob_var_comp[i]=StatFunc::t_prob(d_g-1.0, _prob_var_comp[i]);
		}		
	}
	
	if(jack_b)
	{
		_SE_b.clear();
		_SE_b.resize(_X_c);
		_t_val_b.clear();
		_t_val_b.resize(_X_c);
		_prob_b.clear();
		_prob_b.resize(_X_c);
		for(i=0; i<_X_c; i++)
		{
			ave=0.0;
			for(j=0; j<g; j++) ave+=pseduo_b[j][i];
			ave/=d_g;
			for(j=0; j<g; j++) _SE_b[i]+=(pseduo_b[j][i]-ave)*(pseduo_b[j][i]-ave);
			_SE_b[i]=sqrt( _SE_b[i]*(d_g-1.0)/d_g );			
			
			if( CommFunc::FloatEqual(b[i], 0.0) ) { _SE_b[i]=0.0; _t_val_b[i]=0.0; _prob_b[i]=1.0; continue; }
			else if( CommFunc::FloatEqual(_SE_b[i], 0.0) ) _t_val_b[i]=103.56;
			else _t_val_b[i]=fabs( (d_g*b[i]-(d_g-1.0)*ave) / _SE_b[i] );
			_prob_b[i]=StatFunc::t_prob(d_g-1.0, _t_val_b[i]);
		}
	}
	
	if(jack_e)
	{
		_SE_e.clear();
		_prob_e.clear();
		_SE_e.resize(_r);
		_prob_e.resize(_r);
		for(i=0; i<_r; i++)
		{
			_SE_e[i].resize(_Uu_c[i]);
			_prob_e[i].resize(_Uu_c[i]);
			for(j=0; j<_Uu_c[i]; j++)
			{
				ave=0.0;
				for(k=0; k<g; k++) ave+=pseduo_e[k][i][j];
				ave/=d_g;
				for(k=0; k<g; k++) _SE_e[i][j]+=(pseduo_e[k][i][j]-ave)*(pseduo_e[k][i][j]-ave);
				_SE_e[i][j]=sqrt( _SE_e[i][j]*(d_g-1.0)/d_g );
				
				if( CommFunc::FloatEqual(e[i][j], 0.0) ) { _SE_e[i][j]=0.0; _prob_e[i][j]=1.0; continue; }
				else if( CommFunc::FloatEqual(_SE_e[i][j], 0.0) ) _prob_e[i][j]=103.56;
				else _prob_e[i][j]=fabs( (d_g*e[i][j]-(d_g-1.0)*ave) / _SE_e[i][j]);
				_prob_e[i][j]=StatFunc::t_prob(d_g-1.0, _prob_e[i][j]);
			}
		}
	}
}

void mixed_model::MCMC(bool calcu_var, long smpl_size)
{
	
	/*clock_t start, end;
	double useTime;
	start=clock();
	*/
	
	long i=0, j=0, k=0, l=0;
	
	vector< vector<double>::iterator > W;
	vector< vector<double> > WtW;
	long N=CalcuWtW(W, WtW);
	

	/*end=clock();
	useTime=(double)(end-start);
	cout<<"CalcuWtW! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();
*/


	// initialize d
	vector<double> d(N);
	for(i=0; i<N; i++)
	{
		for(j=0; j<_n; j++) d[i]+=W[i][j]*_Y[j];
	}
	
	/*end=clock();
	useTime=(double)(end-start);
	cout<<"initialize d! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();
*/

	// set prior value
	set_method(MIVQ_mtd, GLS_mtd, LUP_mtd);

	/*end=clock();
	useTime=(double)(end-start);
	cout<<"set_method! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();
*/
	

	var_est();

	/*end=clock();
	useTime=(double)(end-start);
	cout<<"var_est! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();*/


	fixed_eff_est(false);

	/*end=clock();
	useTime=(double)(end-start);
	cout<<"fixed_eff_est! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();
*/

	random_eff_pre();
	
	/*end=clock();
	useTime=(double)(end-start);
	cout<<"random_eff_pre! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();
*/
	

	//set an arbitrary value
	/*long seed=-5251949;
	_var_comp.resize(_r+1);
	_b.resize(_X_c);
	_e.resize(_r+1);
	for(int i=0;i<_r+1;i++)
		_e[i].resize(_Uu_c[i]);
	for(int i=0;i<_r+1;i++)
		_var_comp[i]=1;
	for(int i=0;i<_X_c;i++)
		_b[i] = StatFunc::UniformDev(0, 1, seed);
	for(int i=0;i<_r+1;i++)
	{
		for(int j=0;j<_Uu_c[i];j++)
			_e[i][j]=StatFunc::UniformDev(0, 1, seed);
	}*/

	// initialize wii	
	vector<double> wii(N);
	for(i=0; i<N; i++) wii[i]=WtW[i][i];
	vector<double> wii_o=wii; // backup the original wii
	update_wii(wii, wii_o, _var_comp);
	
	/*end=clock();
	useTime=(double)(end-start);
	cout<<"update_wii! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();*/


	// initialize beta and var_comp
	vector<double> var_comp=_var_comp;
	vector<double> beta=_b;
	for(i=0; i<_r; i++) beta.insert(beta.end(), _e[i].begin(), _e[i].end());

	// Set total Process for QTLNetwork project
	#ifdef _QTLNETWORK_GUI
	long ProcessPoint=0;
	SendMessage(g_ParentWnd, WM_UPDATE_PROGRESSBAR, 1, 11);	
	#endif
	
	// Gibbs sampling
	long warm_up=smpl_size;
	long chain_length=smpl_size*11;
	double d_buf=0.0;
	vector<double> beta_mean(N), beta_var(N), beta_smpl;
	vector< vector<double> > beta_mean_smpl(N), beta_var_smpl(N), var_comp_smpl(_r+1);

	/*end=clock();
	useTime=(double)(end-start);
	cout<<"pre GIBBS! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();
*/
	for(i=0; i<chain_length; i++)
	{
		// Gibbs sampling for beta
		for(j=0; j<N; j++)
		{
			beta_mean[j]=0.0;
			for(k=0; k<N; k++)
			{
				if(k!=j) beta_mean[j]+=WtW[j][k]*beta[k];
			}
			beta_mean[j]=(d[j]-beta_mean[j])/wii[j];
			beta_var[j]=var_comp[_r]/wii[j];
			beta[j]=beta_mean[j]+sqrt(beta_var[j])*StatFunc::gasdev(_seed);
		}

		if(i>=warm_up && i%10==0)
		{
			for(j=0; j<N; j++)
			{
				beta_mean_smpl[j].push_back(beta_mean[j]);
				beta_var_smpl[j].push_back(beta_var[j]);
			}
		}

		// Set process for QTLNetwork project
		#ifdef _QTLNETWORK_GUI
	    if(i%smpl_size==0) SendMessage(g_ParentWnd, WM_UPDATE_PROGRESSBAR, 0, ++ProcessPoint);		
		#endif
		
		// Gibbs sampling for var_comp
		// residual variance
		if(!calcu_var) continue;
		var_comp[_r]=0.0;
		for(j=0; j<_n; j++)
		{
			d_buf=_Y[j];
			for(k=0; k<N; k++) d_buf-=W[k][j]*beta[k];
			var_comp[_r]+=d_buf*d_buf;
		}
		var_comp[_r]/=StatFunc::chidev(_seed, _Uu_c[_r]+2);
		// variances of other random effect
		for(j=0, k=_X_c; j<_r; j++)
		{
			var_comp[j]=0.0;
			for(l=0; l<_Uu_c[j]; l++, k++) var_comp[j]+=beta[k]*beta[k];
			var_comp[j]/=StatFunc::chidev(_seed, _Uu_c[j]+2);
		}
		update_wii(wii, wii_o, var_comp);
		if(i>=warm_up && i%10==0)
		{
			for(j=0; j<_r+1; j++) var_comp_smpl[j].push_back(var_comp[j]);
		}
	}
	
	
	/*end=clock();
	useTime=(double)(end-start);
	cout<<" GIBBS! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();*/

	// calculate bayesian p value
	double d_smpl_size=(double)smpl_size;
	double z=0.0, min=0.0, max=0.0, h=0.0, space_itvl=0.0;
	// calculte P value of variance component
	_SE_var_comp.resize(_r+1);
	_prob_var_comp.resize(_r+1);
	for(i=0; i<_r+1 && calcu_var; i++)
	{
		_var_comp[i]=0.0;
		for(j=0; j<smpl_size; j++) _var_comp[i]+=var_comp_smpl[i][j];
		_var_comp[i]/=d_smpl_size;
		
		_SE_var_comp[i]=0.0;
		for(j=0; j<smpl_size; j++) _SE_var_comp[i]+=(var_comp_smpl[i][j]-var_comp[i])*(var_comp_smpl[i][j]-var_comp[i]);
		_SE_var_comp[i]=sqrt(_SE_var_comp[i]/d_smpl_size);
		
		if( CommFunc::FloatEqual(_var_comp[i], 0.0) ) { _SE_var_comp[i]=0.0; _prob_var_comp[i]=1.0; continue; }
		else if( CommFunc::FloatEqual(_SE_var_comp[i], 0.0) ) _prob_var_comp[i]=103.56;
		else _prob_var_comp[i]=fabs(_var_comp[i]/_SE_var_comp[i]);
		_prob_var_comp[i]=StatFunc::t_prob(d_smpl_size-1.0, _prob_var_comp[i]);
	}
	
	/*end=clock();
	useTime=(double)(end-start);
	cout<<" calculte P value of variance component! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();*/

	// calculte P value of fixed effect
	_SE_b.resize(_X_c);
	_t_val_b.resize(_X_c);
	_prob_b.resize(_X_c);
	for(i=0; i<_X_c; i++)
	{
		_b[i]=0.0;
		for(j=0; j<smpl_size; j++) _b[i]+=beta_mean_smpl[i][j];
		_b[i]/=d_smpl_size;
		
		_SE_b[i]=0.0;
		for(j=0; j<smpl_size; j++) _SE_b[i]+=beta_var_smpl[i][j];
//		for(j=0; j<smpl_size; j++) _SE_b[i]+=pow(beta_mean_smpl[i][j]-_b[i], 2);
		_SE_b[i]=sqrt(_SE_b[i]/d_smpl_size);

		if( CommFunc::FloatEqual(_b[i], 0.0) ) { _SE_b[i]=0.0; _t_val_b[i]=0.0; _prob_b[i]=1.0; continue; }
		else if( CommFunc::FloatEqual(_SE_b[i], 0.0) ) _t_val_b[i]=103.56;
		else _t_val_b[i]=fabs(_b[i]/_SE_b[i]);
		_prob_b[i]=StatFunc::t_prob(d_smpl_size, _t_val_b[i]);
	}
	
	/*end=clock();
	useTime=(double)(end-start);
	cout<<" calculte P value of fixed effect! Time elapsed "<<useTime/60000<<"min."<<endl;
	start=clock();*/


	// calculte P value of random effect
	_SE_e.resize(_r);
	_prob_e.resize(_r);
	for(i=0, l=_X_c; i<_r; i++)
	{
		_SE_e[i].resize(_Uu_c[i]);
		_prob_e[i].resize(_Uu_c[i]);
		for(j=0; j<_Uu_c[i]; j++, l++)
		{
			_e[i][j]=0.0;
			for(k=0; k<smpl_size; k++) _e[i][j]+=beta_mean_smpl[l][k];
			_e[i][j]/=d_smpl_size;
			
			_SE_e[i][j]=0.0;
			for(k=0; k<smpl_size; k++) _SE_e[i][j]+=beta_var_smpl[l][k];
//			for(k=0; k<smpl_size; k++) _SE_e[i][j]+=pow(beta_var_smpl[l][k]-_e[i][j],2);
			_SE_e[i][j]=sqrt(_SE_e[i][j]/d_smpl_size);
			
			if( CommFunc::FloatEqual(_e[i][j], 0.0) ) { _SE_e[i][j]=0.0; _prob_e[i][j]=1.0; continue; }
			else if( CommFunc::FloatEqual(_SE_e[i][j], 0.0) ) _prob_e[i][j]=103.56;
			else _prob_e[i][j]=fabs(_e[i][j]/_SE_e[i][j]);
			_prob_e[i][j]=StatFunc::t_prob(d_smpl_size, _prob_e[i][j]);
		}
	}

	/*end=clock();
	useTime=(double)(end-start);
	cout<<" calculte P value of random effect! Time elapsed "<<useTime/60000<<"min."<<endl;
	*/
}

long mixed_model::CalcuWtW(vector< vector<double>::iterator > &W, vector< vector<double> > &WtW)
{
	// combine _X and _U to construct a new coefficient matrix W
	long i=0, j=0, k=0;
	for(i=0; i<_X_c; i++) W.push_back(_X[i].begin());
	for(i=0; i<_U_c; i++) W.push_back(_U[i].begin());

	// initialize W
	long N=_X_c+_U_c;
	WtW.resize(N);
	for(i=0; i<N; i++) WtW[i].resize(N);
	for(i=0; i<N; i++)
	{
		for(j=i; j<N; j++)
		{
			for(k=0; k<_n; k++) WtW[i][j]+=W[i][k]*W[j][k];
			WtW[j][i]=WtW[i][j]; // WtW is symmetric 
		}
	}

	return N;
}

void mixed_model::update_wii(vector<double> &wii, const vector<double> &wii_o, const vector<double> &var_comp)
{
	long i=0, j=0;
	long pos=_X_c;
	double d_buf=0.0;
	for(i=0; i<_r; i++)
	{
		for(j=0; j<_Uu_c[i]; j++, pos++) wii[pos]=wii_o[pos]+var_comp[_r]/var_comp[i];
	}
}

double mixed_model::qsimp(const double a, const double b, const long smpl_size, const vector<double> &smpl, const double h)
{
	const long JMAX=20;
	const double EPS=1.0e-08;
	long j;
	double s,st,ost=0.0,os=0.0;
	
	for (j=0;j<JMAX;j++) {
		st=trapzd(a,b,j+1, smpl_size, smpl, h);
		s=(4.0*st-ost)/3.0;
		if (j > 5)
			if (fabs(s-os) < EPS*fabs(os) ||
				(s == 0.0 && os == 0.0)) return s;
			os=s;
			ost=st;
	}
	throw("Too many steps in routine qsimp!");
	return 0.0;
}

double mixed_model::trapzd(const double a, const double b, const long n, const long smpl_size, const vector<double> &smpl, const double h)
{
	double x,tnm,sum,del;
	static double s;
	long it,j;
	
	if (n == 1) {
		return (s=0.5*(b-a)*(kernel_func(a, smpl_size, smpl, h)+kernel_func(b, smpl_size, smpl, h)));
	} else {
		for (it=1,j=1;j<n-1;j++) it <<= 1;
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for (sum=0.0,j=0;j<it;j++,x+=del) sum += kernel_func(x, smpl_size, smpl, h);
		s=0.5*(s+(b-a)*sum/tnm);
		return s;
	}
}

double mixed_model::kernel_func(const double z, const long smpl_size, const vector<double> &smpl, const double h)
{
	long i=0;
	double i_buf=0.0, p=0.0, h_square=h*h;
	for(i=0; i<smpl_size; i++)
	{
		i_buf=z-smpl[i];
		p+=exp(-0.5*i_buf*i_buf/h_square);
	}
	
	return p/((double)smpl_size*h*2.506628274631);
}

double mixed_model::Miu() const
{
	return _b[0];
}

double mixed_model::Ve() const
{
	return _var_comp[_r];
}

void mixed_model::var_cmp(vector<double> &var, vector<double> &var_P_val) const
{
	if(_var_comp.empty()) throw("No variance found! mixed_model::var_cmp");
	if(_prob_var_comp.empty()) throw("No variance probability found! mixed_model::var_cmp");
	var=_var_comp;
	var_P_val=_prob_var_comp;
}

void mixed_model::var_cmp(vector<double> &var) const
{
	if(_var_comp.empty()) throw("No variance found! mixed_model::var_cmp");
	var=_var_comp;
}

void mixed_model::var_cmp(vector< vector<double> > &var) const
{

}

void mixed_model::b(vector<double> &b) const
{
	if(_b.empty()) throw("No b found! mixed_model::b");
	b=_b;
}

void mixed_model::b(vector<double> &b, vector<double> &b_P_val) const
{
	if(_b.empty()) throw("No b found! mixed_model::b");
	if(_prob_b.empty()) throw("No b probability found! mixed_model::b");
	b=_b;
	b_P_val=_prob_b;
}

void mixed_model::b(double &b, double &b_SE, double &b_Prob, long pos) const
{
	if(pos>_X_c-1) throw("\"pos\" exceed the size of \"_b\"! mixed_model::output_b");

	b=_b[_X_c-1-pos];
	b_SE=_SE_b[_X_c-1-pos];
	b_Prob=_prob_b[_X_c-1-pos];
}

void mixed_model::e(vector< vector<double> > &e) const
{
	if(_e.empty()) throw("No e found! mixed_model::e");
	e=_e;
}

void mixed_model::e(long pos, vector<double> &e) const
{
	if(pos>_r) throw("r is too large, no random effect in this position! mixed_model::e");
	e=_e[_r-pos];
}

void mixed_model::e(vector< vector<double> > &e, vector< vector<double> > &e_P_val) const
{
	if(_e.empty()) throw("No e found! mixed_model::e");
	if(_prob_e.empty()) throw("No e probability found! mixed_model::e");
	e=_e;
	e_P_val=_prob_e;
}

void mixed_model::e(vector<double> &e, vector<double> &e_SE, vector<double> &e_Prob, long pos) const
{
	if(pos>_r) throw("pos exceed the size of _e! mixed_model::output_e");
	
	e.clear();
	e_SE.clear();
	e_Prob.clear();
	e=_e[_r-pos];
	e_SE=_SE_e[_r-pos];
	e_Prob=_prob_e[_r-pos];
}

void mixed_model::FixedEffect(vector<double> &b, vector<double> &b_SE, vector<double> &b_Prob, long QTL, long FixedEffSN) const
{

}

void mixed_model::RandEffect(vector<double> &e, vector<double> &e_SE, vector<double> &e_Prob, long QTL, long RandEffSN) const
{

}

void mixed_model::output_var_comp(std::ostream &os, long pos, long precision, long width, bool output_SE) const
{
	if(pos>_r) throw("\"pos\" exceed the size of \"_var_comp\"! mixed_model::output_var_comp");
	
	if(_var_comp[_r-pos]>fabs(_Y[0])*0.00001)
	{
		os<<setw(width)<<_var_comp[_r-pos];
		if(output_SE) os<<setw(width)<<_SE_var_comp[_r-pos];
		os<<setprecision(6)<<setw(width)<<_prob_var_comp[_r-pos]<<setprecision(precision);
	}
	else
	{
		os<<setw(width)<<0.0;
		if(output_SE) os<<setw(width)<<0.0;
		os<<setprecision(6)<<setw(width)<<1.0<<setprecision(precision);
	}
}

void mixed_model::output_b(std::ostream &os, long pos, long precision, long width, bool output_SE, bool output_t_val) const
{
	if(pos>_X_c-1) throw("\"pos\" exceed the size of \"_b\"! mixed_model::output_b");
	
	os<<setw(width)<<_b[_X_c-1-pos];
	if(output_SE) os<<setw(width)<<_SE_b[_X_c-1-pos];
	if(output_t_val) os<<setw(width)<<_t_val_b[_X_c-1-pos];
	os<<setprecision(6)<<setw(width)<<_prob_b[_X_c-1-pos]<<setprecision(precision);
}

void mixed_model::output_e(std::ostream &os, long pos, long precision, long width, bool output_SE) const
{
	if(pos>_r) throw("pos exceed the size of _e! mixed_model::output_e");
	
	long i=0;
	for(i=0; i<_Uu_c[_r-pos]; i++)
	{
		os<<setw(width)<<_e[_r-pos][i];
		if(output_SE) os<<setw(width)<<_SE_e[_r-pos][i];
		os<<setprecision(6)<<setw(width)<<_prob_e[_r-pos][i]<<setprecision(precision);
	}
}

void mixed_model::EffGrpSN(vector<long> &EffGrp, long QTL, long EffSN) const
{

}

void mixed_model::calcu_V()
{
	calcu_V(_V, _var_comp);
}

void mixed_model::contrast(const vector<double> &c_vec, double &C_val, double &SE, double &P_val)
{
	long i=0;
	
	C_val=0.0;
	for(i=0; i<_n; i++) C_val+=_Y[i]*c_vec[i];
	SE=sqrt( fabs(MatFunc::At_B_A(c_vec, _V)) );
	if( CommFunc::FloatEqual(C_val, 0.0) ) P_val=0.0;
	else if( CommFunc::FloatEqual(SE, 0.0) ) P_val=103.56;
	else P_val=fabs(C_val/SE);
	P_val=StatFunc::t_prob((double)(_n-_X_c), P_val);
}

void mixed_model::output(std::ostream &os) const
{
	long i=0, j=0;
	os<<"_n: "<<_n<<endl;
	os<<"_r: "<<_r<<endl;
	os<<"_Y: ";
	for(i=0; i<_Y.size(); i++) os<<_Y[i]<<" ";
	os<<endl;
	os<<"_X_c: "<<_X_c<<endl;
	os<<"_X"<<endl;
	if(!_X.empty()) MatFunc::output(_X, os, 2);
	os<<"_r: "<<_r<<endl;
	os<<"_U: "<<endl;
	if(!_U.empty()) MatFunc::output(_U, os, 2);
	os<<"_U_v: "<<endl;
	if(!_U_v.empty()) MatFunc::output(_U_v, os, 2);
	os<<"_U_p: "<<endl;
	if(!_U_p.empty()) MatFunc::output(_U_p, os, 2);
	os<<"_U_r: "<<endl;
	if(!_U_r.empty()) for(i=0; i<_U_r.size(); i++) os<<_U_r[i]<<" ";
	os<<endl;
	os<<"_Uu_c: "<<endl;
	if(!_Uu_c.empty()) for(i=0; i<_Uu_c.size(); i++) os<<_Uu_c[i]<<" ";
	os<<endl;
	os<<"_U_c: "<<_U_c<<endl;
	
	os<<"_var_est_mtd: "<<_var_est_mtd<<endl;;
	os<<"_fix_est_mtd: "<<_fix_est_mtd<<endl;;
	os<<"_rand_est_mtd: "<<_rand_pre_mtd<<endl;;
}

void mixed_model::OutlierAnalysis(double alpha)
{
	bool Stop=false;
	while(!Stop)
	{
		vector< vector<double>::iterator > W;
		vector< vector<double> > WtW;
		long N=CalcuWtW(W, WtW);
		double RandWtW=MatFunc::SVDInversion(WtW, 1.0e-08);
		
		set_method(MIVQ_mtd, GLS_mtd, BLUP_mtd);
		var_est();
		random_eff_pre();
		
		long i=0;
		double DF=(double)_n-RandWtW;
		vector<double> t_Val(_n), P_Val(_n);
		for(i=0; i<_n; i++)
		{
			t_Val[i]=_e[_r][i]/(_var_comp[_r]*sqrt(_Q[i][i]));
			P_Val[i]=StatFunc::t_prob(DF, t_Val[i]);
		}
		double Threshold=StatFunc::ControlFDR(P_Val, alpha, false);
		if(Threshold<0.0) break;

		long Pos=0;
		double d_Buf=0.0;
		for(i=0; i<_n; i++)
		{
			if(P_Val[i]>Threshold && P_Val[i]>d_Buf)  Pos=i;
		}
		RemoveOneObs(Pos);
	}
}

void mixed_model::RemoveOneObs(long Pos)
{
	long i=0, j=0;
	for(i=0; i<_X_c; i++) _X[i].erase(_X[i].begin()+Pos);
	for(i=0; i<_U_c; i++) _U[i].erase(_U[i].begin()+Pos);
	_n--;
}
