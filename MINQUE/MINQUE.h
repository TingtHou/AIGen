#pragma once
#include <vector>
#include <Eigen/Dense>
#include "MatFunc.h"
using namespace std;
class MINQUE
{
public:
	MINQUE();
	void push_back_U(const vector< vector<double> > &Ui);
	void import_data(const vector<double> &Y);
	void SetGPU(bool isGPU);
	void var_est();
	~MINQUE();
private:
	void calcu_Q();
	void calcu_Q_U(vector< vector<double> > &Q_Uv, long u, long &m);
	void calcu_q(const vector< vector<double> > &Q_Uv, long u);
	void HR_equation();
	void init_Vi();
	vector<double> _alpha;  // alpha for MINQUE
	vector< vector<double> > _U; // excluding the last identity matrix
	vector< vector<double> > _U_v;
	vector< vector<long> >_U_p;
	vector<long> _U_r;
	vector<long> _Uu_c;
	long _U_c;
	long _r;
	vector<double> _q;
	vector< vector<double> > _Vi;
	vector< vector<double> > _Q;
	vector< vector<double> > _Yt_Q_U;
	vector< vector<double> > _H;
	vector<double> _var_comp;
	long _seed;
	long _n; // the observation number
	vector<double> _Y; // observations


};

