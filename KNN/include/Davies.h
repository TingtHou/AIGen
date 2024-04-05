#pragma once
#define UseDouble 0             /* all floating point double */


#include <setjmp.h>
#include <cmath>

using std::atan;
using std::exp;
using std::sin;
using std::log;
using std::fabs;
using std::sqrt;
using std::floor;
typedef int BOOL;





class  Davies
{
public:
	 Davies();
	~ Davies();
	void  qfc(double* lb1, double* nc1, int* n1, int* r1, double* sigma, double* c1, int* lim1, double* acc, double* trace, int* ifault, double* res);
private:
	 double sigsq, lmax, lmin, mean, c;
	 double intl, ersm;
	 int count, r, lim;  BOOL ndtsrt, fail;
	 int* n, * th;  double* lb, * nc;
	 jmp_buf env;
	 double pi = 3.14159265358979;
	 double log28 = .0866;  /*  log(2.0) / 8.0  */

	 double exp1(double x);
	 void counter(void);
	 double square(double x) { return x * x; };
	 double cube(double x) { return x * x * x; };
	 double  log1(double x, BOOL first);
	 void order(void);
	 double   errbd(double u, double* cx);
	 double  ctff(double accx, double* upn);
	 double truncation(double u, double tausq);
	 void findu(double* utx, double accx);
	 void integrate(int nterm, double interv, double tausq, BOOL mainx);
	 double cfe(double x);

};


/*

*/


