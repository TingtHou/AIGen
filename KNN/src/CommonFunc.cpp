#include "../include/CommonFunc.h"

int Inverse(Eigen::MatrixXf & Ori_Matrix, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse)
{
	int status = 0;
	bool statusInverse;
	switch (DecompositionMode)
	{
	case Cholesky:
	{
		float det;
		statusInverse = ToolKit::comput_inverse_logdet_LDLT_mkl(Ori_Matrix);
	//	statusInverse = ToolKit::Inv_Cholesky(Ori_Matrix);
		status += !statusInverse;
 		if (!statusInverse&&allowPseudoInverse)
 		{
			if (AltDecompositionMode == SVD)
 			{
 				statusInverse = ToolKit::comput_inverse_logdet_SVD_mkl(Ori_Matrix);
 			}
 			if (AltDecompositionMode == QR)
 			{
 				statusInverse = ToolKit::comput_inverse_logdet_QR_mkl(Ori_Matrix);
 			}
 			status += !statusInverse;
 		}
	}
	break;
	case LU:
	{
		float det;
		statusInverse = ToolKit::comput_inverse_logdet_LU_mkl(Ori_Matrix);
		status += !statusInverse;
 		if (!statusInverse&&allowPseudoInverse)
 		{
			if (AltDecompositionMode == SVD)
 			{
 				statusInverse = ToolKit::comput_inverse_logdet_SVD_mkl(Ori_Matrix);
 			}
			if (AltDecompositionMode == QR)
			{
 				statusInverse = ToolKit::comput_inverse_logdet_QR_mkl(Ori_Matrix);
 			}
 			status += !statusInverse;
 		}
	}
	break;
	case QR:
	{
		statusInverse = ToolKit::comput_inverse_logdet_QR_mkl(Ori_Matrix);
		status += !statusInverse;
	}
	break;
	case SVD:
	{
		statusInverse = ToolKit::comput_inverse_logdet_SVD_mkl(Ori_Matrix);
		status += !statusInverse;
	}
	break;
 	}
	return status;
}
int Inverse(Eigen::MatrixXd & Ori_Matrix, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse)
{
	int status = 0;
	bool statusInverse;
	switch (DecompositionMode)
	{
	case Cholesky:
	{
		float det;
		statusInverse = ToolKit::comput_inverse_logdet_LDLT_mkl(Ori_Matrix);
		//	statusInverse = ToolKit::Inv_Cholesky(Ori_Matrix);
		status += !statusInverse;
		if (!statusInverse && allowPseudoInverse)
		{
			if (AltDecompositionMode == SVD)
			{
				statusInverse = ToolKit::comput_inverse_logdet_SVD_mkl(Ori_Matrix);
			}
			if (AltDecompositionMode == QR)
			{
				statusInverse = ToolKit::comput_inverse_logdet_QR_mkl(Ori_Matrix);
			}
			status += !statusInverse;
		}
	}
	break;
	case LU:
	{
		float det;
		statusInverse = ToolKit::comput_inverse_logdet_LU_mkl(Ori_Matrix);
		status += !statusInverse;
		if (!statusInverse && allowPseudoInverse)
		{
			if (AltDecompositionMode == SVD)
			{
				statusInverse = ToolKit::comput_inverse_logdet_SVD_mkl(Ori_Matrix);
			}
			if (AltDecompositionMode == QR)
			{
				statusInverse = ToolKit::comput_inverse_logdet_QR_mkl(Ori_Matrix);
			}
			status += !statusInverse;
		}
	}
	break;
	case QR:
	{
		statusInverse = ToolKit::comput_inverse_logdet_QR_mkl(Ori_Matrix);
		status += !statusInverse;
	}
	break;
	case SVD:
	{
		statusInverse = ToolKit::comput_inverse_logdet_SVD_mkl(Ori_Matrix);
		status += !statusInverse;
	}
	break;
	}
	return status;
}
//Sample variance
float Variance(Eigen::VectorXf & Y)
{
	float meanY = mean(Y);
	float variance = 0;
	int nind = Y.size();
	for (int i=0;i<nind;i++)
	{
		variance += (Y[i] - meanY)*(Y[i] - meanY);
	}
	variance /= (float)(nind-1);
	return variance;
}

float mean(Eigen::VectorXf & Y)
{
	float sumY = Y.sum();
	float nind = Y.size();
	return sumY/nind;
}

float isNum(std::string line)
{
	stringstream sin(line);
	float d;
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


void stripSameCol(Eigen::MatrixXf & Geno)
{
	std::vector<int> repeatID;
	repeatID.clear();
	for (int i = 0; i < Geno.cols(); i++)
	{
		float *test = Geno.col(i).data();
		std::vector<float> rowi(test, test + Geno.col(i).size());
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
	Eigen::MatrixXf tmpGeno = Geno;
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

void stdSNPmv(Eigen::MatrixXf & Geno)
{
	int ncol = Geno.cols();
	int nrow = Geno.rows();
	Eigen::MatrixXf tmpGeno = Geno;
	Geno.setZero();
	for (int i = 0; i < ncol; i++)
	{
		Eigen::VectorXf Coli(nrow);
		Coli << tmpGeno.col(i);
		float means = mean(Coli);
		float sd = std::sqrt(Variance(Coli));
		Coli -= means * Eigen::VectorXf::Ones(nrow);
		Coli /= sd;
		Geno.col(i) << Coli;
	}
}

void set_difference(boost::bimap<int, std::string>& map1, boost::bimap<int, std::string>& map2, std::vector<std::string>& overlap)
{
	for (auto it=map1.left.begin();it!=map1.left.end();it++)
	{
		if (map2.right.count(it->second))
		{
			overlap.push_back(it->second);
		}
	}
}

void GetSubMatrix(Eigen::MatrixXf & oMatrix, Eigen::MatrixXf & subMatrix, std::vector<int> rowIds, std::vector<int> colIDs)
{
	for (int i=0;i<rowIds.size();i++)
	{
		for (int j=0;j<colIDs.size();j++)
		{
			subMatrix(i, j) = oMatrix(rowIds[i], colIDs[j]);
		}
	}
}

void GetSubVector(Eigen::VectorXf & oVector, Eigen::VectorXf & subVector, std::vector<int> IDs)
{
	for (int i=0;i<IDs.size();i++)
	{
		subVector(i) = oVector(IDs[i]);
	}
}
