#pragma once
#include "cuMinqueBase.h"

class cuMINQUE0 :
	public cuMinqueBase
{
public:
	cuMINQUE0(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse);
	~cuMINQUE0();
	void estimateVCs();
private:
};

