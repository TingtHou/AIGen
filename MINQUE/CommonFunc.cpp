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
//Sample variance
double Variance(Eigen::VectorXd & Y)
{
	double meanY = mean(Y);
	double variance = 0;
	int nind = Y.size();
	for (int i=0;i<nind;i++)
	{
		variance += (Y[i] - meanY)*(Y[i] - meanY);
	}
	variance /= (double)(nind-1);
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


void stripSameCol(Eigen::MatrixXd & Geno)
{
	std::vector<int> repeatID;
	repeatID.clear();
	for (int i = 0; i < Geno.cols(); i++)
	{
		double *test = Geno.col(i).data();
		std::vector<double> rowi(test, test + Geno.col(i).size());
		std::sort(rowi.begin(), rowi.end());
		auto it = std::unique(rowi.begin(), rowi.end());
		rowi.erase(it, rowi.end());
		int len = rowi.size();
		if (len == 1)
		{
			repeatID.push_back(i);
		}
	}
	if (repeatID.empty())
	{
		return;
	}
	Eigen::MatrixXd tmpGeno = Geno;
	Geno.resize(tmpGeno.rows(), tmpGeno.cols() - repeatID.size());
	int j = 0;
	for (int i = 0; i < tmpGeno.cols(); i++)
	{
		if (std::find(repeatID.begin(), repeatID.end(), i) != repeatID.end())
		{
			continue;
		}
		Geno.col(j++) = tmpGeno.col(i);
	}

}

void stdSNPmv(Eigen::MatrixXd & Geno)
{
	int ncol = Geno.cols();
	int nrow = Geno.rows();
	Eigen::MatrixXd tmpGeno = Geno;
	Geno.setZero();
	for (int i = 0; i < ncol; i++)
	{
		Eigen::VectorXd Coli(nrow);
		Coli << tmpGeno.col(i);
		double means = mean(Coli);
		double sd = std::sqrt(Variance(Coli));
		Coli -= means * Eigen::VectorXd::Ones(nrow);
		Coli /= sd;
		Geno.col(i) << Coli;
	}
}
