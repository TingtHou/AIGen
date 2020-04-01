#pragma once
#include "cuMinqueBase.h"
class cuMINQUE1:
	public cuMinqueBase
{
public:
	cuMINQUE1(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse);
	~cuMINQUE1();
	void estimateVCs();
private:
};


