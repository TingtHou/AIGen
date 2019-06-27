#include "pch.h"
#include "CommonFunc.h"
#include "ToolKit.h"
#include <boost/algorithm/string.hpp>
int Inverse(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse)
{
	int status = 0;
	bool statusInverse;
	switch (DecompositionMode)
	{
	case Cholesky:
	{
		statusInverse = ToolKit::Inv_Cholesky(Ori_Matrix, Inv_Matrix);
		status += !statusInverse;
		if (!statusInverse&&allowPseudoInverse)
		{
			if (AltDecompositionMode==SVD)
			{
				statusInverse = ToolKit::Inv_SVD(Ori_Matrix, Inv_Matrix, allowPseudoInverse);
			}
			if (AltDecompositionMode==QR)
			{
				statusInverse = ToolKit::Inv_QR(Ori_Matrix, Inv_Matrix, allowPseudoInverse);
			}
			status += !statusInverse;
		}
	}
		break;
	case LU:
	{
		statusInverse = ToolKit::Inv_LU(Ori_Matrix, Inv_Matrix);
		status += !statusInverse;
		if (!statusInverse&&allowPseudoInverse)
		{
			status += 1;
			if (AltDecompositionMode == SVD)
			{
				statusInverse = ToolKit::Inv_SVD(Ori_Matrix, Inv_Matrix, allowPseudoInverse);
			}
			if (AltDecompositionMode == QR)
			{
				statusInverse = ToolKit::Inv_QR(Ori_Matrix, Inv_Matrix, allowPseudoInverse);
			}
			status += !statusInverse;
		}
	}
	break;
	case QR:
	{
		statusInverse = ToolKit::Inv_QR(Ori_Matrix, Inv_Matrix,allowPseudoInverse);
		status += !statusInverse;
	}
	break;
	case SVD:
	{
		statusInverse = ToolKit::Inv_SVD(Ori_Matrix, Inv_Matrix, allowPseudoInverse);
		status += !statusInverse;
	}
	break;
	}
	return status;
}

double Variance(Eigen::VectorXd & Y)
{
	double meanY = mean(Y);
	double variance = 0;
	int nind = Y.size();
	for (int i=0;i<nind;i++)
	{
		variance += (Y[i] - meanY)*(Y[i] - meanY);
	}
	variance /= (double)nind;
	return variance;
}

double mean(Eigen::VectorXd & Y)
{
	double sumY = Y.sum();
	double nind = Y.size();
	return sumY/nind;
}

double isNum(std::string line)
{
	stringstream sin(line);
	double d;
	char c;
	if (!(sin >> d))
		return false;
	if (sin >> c)
		return false;
	return true;
}


std::string GetBaseName(std::string pathname)
{
	std::vector<string> splitstr;
	boost::split(splitstr, pathname, boost::is_any_of("/\\"), boost::token_compress_on);
	return splitstr.at(splitstr.size() - 1);
}

std::string GetParentPath(std::string pathname)
{
	std::vector<string> splitstr;
	boost::split(splitstr, pathname, boost::is_any_of("/\\"), boost::token_compress_on);
	splitstr.pop_back();
	std::string ParentPath = boost::join(splitstr, "/");
	return ParentPath==""?".":ParentPath;
}