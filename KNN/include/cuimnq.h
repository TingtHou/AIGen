#pragma once
#include "CommonFunc.h"
#include "cuMinqueBase.h"
#include "cuMINQUE1.h"

class cuimnq :
	public cuMinqueBase
{
public:
	void estimateVCs();										//starting estimation
	void setOptions(MinqueOptions mnqoptions);				//
	int getIterateTimes();
	void isEcho(bool isecho);
private:
	float tol = 1e-5; //convergence tolerence (def=1e-5)
	int itr = 20;	//iterations allowed (def=20)

	int initIterate = 0;
	bool isecho = true;
	bool MINQUE1 = false;
private:
	void Iterate();
};

