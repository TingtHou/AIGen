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

void GetSubMatrix(Eigen::MatrixXf& oMatrix, Eigen::MatrixXf& subMatrix, std::vector<int> rowIds)
{
	for (int i = 0; i < rowIds.size(); i++)
	{
		for (int j = 0; j < oMatrix.cols(); j++)
		{
			subMatrix(i, j) = oMatrix(rowIds[i], j);
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

float Cor(Eigen::VectorXf& Y1, Eigen::VectorXf& Y2)
{
	assert(Y1.size() == Y2.size());
	int nind = Y1.size();
	double upper = 0;
	double lower = 0;
	Eigen::VectorXd Y1_double = Y1.cast<double>();
	Eigen::VectorXd Y2_double = Y2.cast<double>();
	Eigen::VectorXd Y1_Y2 = Y1_double.cwiseProduct(Y2_double);
	double Y1_sum = Y1_double.sum();
	double Y2_sum = Y2_double.sum();
	double Y1_squre_sum = Y1_double.cwiseProduct(Y1_double).sum();
	double Y2_squre_sum = Y2_double.cwiseProduct(Y2_double).sum();
	double Y1_Y2_sum = Y1_Y2.sum();
	upper = nind * Y1_Y2_sum - Y1_sum * Y2_sum;
	lower = std::sqrt(nind * Y1_squre_sum - Y1_sum * Y1_sum) * std::sqrt(nind * Y2_squre_sum - Y2_sum * Y2_sum);
	float cor = (float)upper / lower;
	return cor;
}

float AUC(Eigen::VectorXf& Response, Eigen::VectorXf& Predictor)
{
	return 0.0f;
}

ROC::ROC(Eigen::VectorXf& Response, Eigen::VectorXf& Predictor)
{
	this->Response = Response;
	this->Predictor = Predictor;
	assert(Response.size() == Predictor.size());
	nind = Response.size();
	init();
	Calc();
	AUC();
}

void ROC::init()
{
	thresholds.resize(502);
	Specificity.resize(502);
	Sensitivity.resize(502);
	thresholds.setZero();
	Specificity.setZero();
	Sensitivity.setZero();
	float mins = Predictor.minCoeff();
	float maxs = Predictor.maxCoeff();
	step = (maxs- mins) / 500;
	for (int i = 1; i < thresholds.size()-1; i++)
	{
		thresholds[i] = mins + step * i;
	}
	thresholds[0]= -std::numeric_limits<double>::infinity();
	thresholds[501] = std::numeric_limits<double>::infinity();
}

void ROC::Calc()
{
	for (int i = 0; i < thresholds.size(); i++)
	{
		
		int TruePos = 0;
		int FaslePos = 0;
		int TrueNeg = 0;
		int FasleNeg = 0;
		for (int j = 0; j < nind; j++)
		{
			if (Predictor[j] < thresholds[i])
			{
				if (abs(Response[j]) < 1e-7)
					TrueNeg++;
				else
					FasleNeg++;
			}
			else
			{
				if (Response[j] > 1e-7)
					TruePos++;
				else
					FaslePos++;
			}
		}
		Sensitivity[i] = (float)TruePos / (float)(TruePos+FasleNeg);
		Specificity[i] = (float)TrueNeg / (float)(TrueNeg+FaslePos);
	}
}

void ROC::AUC()
{
	double auc = 0;
	for (size_t i = 1; i < Sensitivity.size(); i++)
	{
		double height = (Sensitivity[i] + Sensitivity[i-1])/2;
		auc += height * abs(Specificity[i]- Specificity[i - 1]);
	}
	this->auc = auc;
}

