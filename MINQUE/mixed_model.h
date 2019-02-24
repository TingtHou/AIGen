// mixed_model.h
// Auther: Jian Yang 
// Date: 28/4/04
// Copyright @ Institute of Bioinformatics, Zhejiang University, China

// _X, _U, and _QU are column matrix

#ifndef _MIXED_MODEL_H_
#define _MIXED_MODEL_H_

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#ifdef _QTLNETWORK_GUI
#include "GlobVarb.h"
#endif

#include <fstream>
#include <iostream>
#include "MatFunc.h"
using namespace std;

class mixed_model  
{
public:
	// MINQUE0, MINQUE_1, REML, EM and  MINQUE_alpha
	enum var_est_mtd { MINQ0_mtd=0, MINQ1_mtd=1, REML_mtd=2, EM_mtd=3, MINQ_alpha_mtd=4, MIVQ_mtd=5 };

	// OLS square and GLS
	enum fix_est_mtd { OLS_mtd=0, GLS_mtd=1 };

	// BLUP, LUP and AUP
	enum rand_pre_mtd { BLUP_mtd=0, LUP_mtd=1, AUP_mtd=2 };

public:
	// constructor and deconstructor
	mixed_model();
	virtual ~mixed_model();

	void import_data(const vector<double> &Y);
	virtual void import_data(const vector< vector<double> > &Y);
	virtual void import_exp_des(const long rep_num, const long env_num);
	virtual void import_cross(const long cross_cate);

	// add or erase element(s) of _X including _X_v, _X_p, _X_r
	// add or erase element(s) of _U including _U_v, _U_p, _U_r and _U_c
	virtual void push_back_HomoEffects(const vector< vector< vector<long> > > &FixEffBuf, long QTLNum, long EffectSN);
	void push_back_X(const vector<double> &X_col);
	virtual void push_back_X(const vector< vector<double> > &X);
	void push_back_U(const vector< vector<double> > &Ui);
	void pop_back_X(long size=1);
	void pop_back_U(long size=1);

	// set the methods of variance estimation, fixed effect estimation and random effect prediction
	void set_method(long var_est_mtd=MINQ1_mtd, long fix_est_mtd=GLS_mtd, long rand_pre_mtd=LUP_mtd);

	// implement
	virtual void var_est();
	//virtual void fixed_eff_est(bool calcu_prob=true);
	void random_eff_pre();

	void jknf_test(bool jack_var_comp, bool jack_b, bool jack_e, long jack_size);

	virtual void MCMC(bool calcu_var, long smpl_size);

	// calculate the V matrix
	void calcu_V();

	// do linear contrast
	void contrast(const vector<double> &c_vec, double &C_val, double &SE, double &P_val);

	// calculate the likelihood function using estimated parameters
	// calculate -ln|V|-(y-Xb)t*Vi*(y-Xb)
	double calcu_L();
	double calcu_lnV();

	// return result
	double Miu() const;
	double Ve() const;
	void var_cmp(vector<double> &var) const;
	void var_cmp(vector<double> &var, vector<double> &var_P_val) const;
	virtual void var_cmp(vector< vector<double> > &var) const;
	void b(vector<double> &b) const;
	void b(vector<double> &b, vector<double> &b_P_val) const;
	virtual void b(double &b, double &b_SE, double &b_Prob, long pos) const;
	void e(vector< vector<double> > &e) const;
	void e(long r, vector<double> &e) const;
	void e(vector< vector<double> > &e, vector< vector<double> > &e_P_val) const;
	virtual void e(vector<double> &e, vector<double> &e_SE, vector<double> &e_Prob, long pos) const;
	virtual void FixedEffect(vector<double> &b, vector<double> &b_SE, vector<double> &b_Prob, long QTL, long FixedEffSN) const;
	virtual void RandEffect(vector<double> &e, vector<double> &e_SE, vector<double> &e_Prob, long QTL, long RandEffSN) const;
//	virtual void EffGrpSN(vector<long> &EffGrp, long QTL, long EffSN) const;

	// output
	// pos: the distance between target element and the last element
	void output(std::ostream &os=std::cout) const;
	void output_var_comp(std::ostream &os, long pos, long precision, long width, bool output_SE=false) const;
	void output_b(std::ostream &os, long pos, long precision, long width, bool output_SE=false, bool output_t_val=false) const;
	void output_e(std::ostream &os, long pos, long precision, long width, bool output_SE=false) const;

	// Outlier deletion analysis
	void OutlierAnalysis(double alpha);

protected:
	void sparse_mat();

	// initialize the _Vi matrix
	void init_Vi();
	void init_Vi(const vector<double> &prior);

	// calculate the inverse of the Variance-Covariance matrix using Morrision method
	void morrision_Vi(const vector<double> &prior);

	// calculate the matrix Q*U 
	void calcu_Q();

	// return the Xt_Vi_X matrix
	void Xt_Vi_X_inverse();

	// calculate Q_U
	void calcu_Q_U(vector< vector<double> > &Q_Uv, long u, long &m);

	// calculate q
	virtual void calcu_q(const vector< vector<double> > &Q_Uv, long u);

	// the equation proposed by Hartley and Rao: H*[var]=q
	virtual void HR_equation();

	void Ut_M_U(const vector< vector<double> > &M, vector< vector<double> > &Ut_M_U, long u, long m);

	// caluclate the matrix V
	void calcu_V(vector< vector<double> > &V, const vector<double> &sigma_square) const;

	// calculate the variance componments
	void REML(); // The maximum iteration time is 10
	virtual void MINQUE_0();
	virtual void MINQUE_1();
	void MINQUE_alpha(const vector<double> &prior);
	void MIVQUE();
	void EM(); // restricted iteration times is 10

	// fixed effect estimation
//	virtual void GLS(bool calcu_prob); // General Least Square (GLS)
	void OLS(bool calcu_prob);	// Ordinary Least Square (OLS)

	// random effect prediction
	void BLUP(); // Best Linear Unbiased Estimation
	void LUP(); // Linear Unbiased Estimation
	void AUP(); // Adjusted Unbiased Estimation

	// MCMC related
	long CalcuWtW(vector< vector<double>::iterator > &W, vector< vector<double> > &WtW);
	void update_wii(vector<double> &wii, const vector<double> &wii_o, const vector<double> &var_comp);
	double kernel_func(const double z, const long smpl_size, const vector<double> &smpl, const double h);
	double qsimp(const double a, const double b, const long smpl_size, const vector<double> &smpl, const double h);
	double trapzd(const double a, const double b, const long n, const long smpl_size, const vector<double> &smpl, const double h);

	// Outlier analysis
	void RemoveOneObs(long Pos);
protected:
	long _seed;

	long _n; // the observation number
	vector<double> _Y; // observations

	// coefficient matrices
	vector< vector<double> > _X; // full _X matrix
	long _X_c; // column number of _X
	long _p;
	vector< vector<double> > _U; // excluding the last identity matrix
	vector< vector<double> > _U_v;
	vector< vector<long> >_U_p;
	vector<long> _U_r;
	vector<long> _Uu_c;
	long _U_c;
	long _r;

	vector<double> _alpha;  // alpha for MINQUE

	vector< vector<double> > _V; // calculated by the estimated variance components
	vector< vector<double> > _Vi;
	vector< vector<double> > _Vi_X;
	vector< vector<double> > _Xt_Vi_X;
	vector< vector<double> > _Q;
	vector< vector<double> > _Yt_Q_U;
	vector< vector<double> > _H;
	vector<double> _q;
	vector<double> _var_comp;
	vector<double> _SE_var_comp;
	vector<double> _prob_var_comp;
	vector<double> _b;
	vector<double> _SE_b;
	vector<double> _t_val_b;
	vector<double> _prob_b;
	vector< vector<double> > _e;
	vector< vector<double> > _SE_e;
	vector< vector<double> > _prob_e;

	long _cross_cate;

	// method options
	long _var_est_mtd;
	long _fix_est_mtd;
	long _rand_pre_mtd;

	// process
	long _total_process; 
	long _process_point;

};

#endif