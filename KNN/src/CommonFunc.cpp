#include "../include/CommonFunc.h"
#include <random>


//@brief:	Inverse a float matrix, and rewrite the inversed matrix into original matrix;
//@param:	Ori_Matrix				The matrix to be inversed;
//@param:	DecompositionMode		The mode to decompose the matrix;Cholesky = 0, LU = 1, QR = 2,SVD = 3;
//@param:	AltDecompositionMode	Another mode to decompose if the first mode is failed; Only QR = 2 and SVD = 3 are allowed;
//@param:	allowPseudoInverse		Allow to use pseudo inverse matrix; If True, AltDecompositionMode is avaiable; otherwise AltDecompositionMode is unavaiable.
//@ret:		int						The info of statue of inverse method; 
//									0 means successed; 1 means calculating inverse matrix is failed, and  using pseudo inverse matrix instead if allowPseudoInverse is true; 
//									2 means calculating inverse matrix is failed, and pseudo inverse matrix is also failed.
int Inverse(Eigen::MatrixXf & Ori_Matrix, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse)
{
	int status = 0;
	bool statusInverse;
	switch (DecompositionMode)
	{
	case Cholesky:
	{
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

//@brief:	Inverse a double matrix, and rewrite the inversed matrix into original matrix;
//@param:	Ori_Matrix				The matrix to be inversed;
//@param:	DecompositionMode		The mode to decompose the matrix;Cholesky = 0, LU = 1, QR = 2,SVD = 3;
//@param:	AltDecompositionMode	Another mode to decompose if the first mode is failed; Only QR = 2 and SVD = 3 are allowed;
//@param:	allowPseudoInverse		Allow to use pseudo inverse matrix; If True, AltDecompositionMode is avaiable; otherwise AltDecompositionMode is unavaiable.
//@ret:		int						The info of statue of inverse method; 
//									0 means successed; 1 means calculating inverse matrix is failed, and  using pseudo inverse matrix instead if allowPseudoInverse is true; 
//									2 means calculating inverse matrix is failed, and pseudo inverse matrix is also failed.
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


float Variance(Eigen::VectorXf & Y)
{
	float meanY = mean(Y);
	float variance = (Y.array() - Y.mean()).square().sum() / (Y.size() - 1);
	return variance;
}

float mean(Eigen::VectorXf & Y)
{
	float sumY = Y.sum();
	float nind = Y.size();
	return sumY/nind;
}


bool isNum(std::string line)
{
	std::stringstream sin(line);
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
	std::vector< std::string> splitstr;
	boost::split(splitstr, pathname, boost::is_any_of("/\\"), boost::token_compress_on);
	return splitstr.at(splitstr.size() - 1);
}

std::string GetParentPath(std::string pathname)
{
	std::vector< std::string> splitstr;
	boost::split(splitstr, pathname, boost::is_any_of("/\\"), boost::token_compress_on);
	splitstr.pop_back();
	std::string ParentPath = boost::join(splitstr, "/");
	return ParentPath==""?".":ParentPath;
}

//@brief:	remove the matrix column whose value is constant, and rewrite the new matrix into original matrix;
//@param:	Geno	The matrix to be impute;
//@ret:		void						
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

//@brief:	Centralize a matrix by column, and rewrite the new matrix into original matrix;
//@param:	Geno	The matrix to be centralized;
//@ret:		void	
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

//@brief:	Compare two bimap, and write the overlap elements into a vector;
//@param:	map1		Bimap 1;
//@param:	map2		Bimap 2;
//@param:	overlap		A vector that stores overlap elements;
//@ret:		void	
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

boost::bimap<int, std::string> set_difference(boost::bimap<int, std::string>& map1, boost::bimap<int, std::string>& map2)
{
	boost::bimap<int, std::string> overlap;
	int id = 0;
	for (auto it = map1.left.begin(); it != map1.left.end(); it++)
	{
		if (map2.right.count(it->second))
		{
			overlap.insert({ id++ ,it->second });
		}
	}
	return overlap;
}

boost::bimap<int, std::string> set_difference(std::vector<boost::bimap<int, std::string>>& mapList)
{
	boost::bimap<int, std::string> overlap;
	if (mapList.size()==1)
	{
		overlap = mapList.back();
		mapList.pop_back();
	}
	else
	{
		boost::bimap<int, std::string> map1 = mapList.back();
		mapList.pop_back();
		boost::bimap<int, std::string> map2 = set_difference(mapList);
		overlap=set_difference(map1, map2);
	}
	return overlap;
}

void set_difference(boost::bimap<int, std::string>& map1, boost::bimap<int, std::string>& map2, boost::bimap<int, std::string>& map3, std::vector<std::string>& overlap)
{
	for (auto it = map1.left.begin(); it != map1.left.end(); it++)
	{
		if (map2.right.count(it->second))
		{
			if (map3.right.count(it->second))
			{
				overlap.push_back(it->second);
			}
		}
	}
}

//@brief:	Get a subset of a float matrix given specific row IDs and column IDs;
//@param:	oMatrix			The original matrix;
//@param:	subMatrix		A subset matrix of the origianl matrix;
//@param:	rowIds			A vector containing row IDs to be kept;
//@param:	colIDs			A vector containing column IDs to be kept;
//@ret:		void	
void GetSubMatrix(std::shared_ptr<Eigen::MatrixXf>  oMatrix, std::shared_ptr<Eigen::MatrixXf> subMatrix, std::vector<int> rowIds, std::vector<int> colIDs)
{
	for (int i=0;i<rowIds.size();i++)
	{
		for (int j=0;j<colIDs.size();j++)
		{
			(*subMatrix)(i, j) = (*oMatrix)(rowIds[i], colIDs[j]);
		}
	}
}

//@brief:	Get a subset of a float matrix given specific row IDs;
//@param:	oMatrix			The original matrix;
//@param:	subMatrix		A subset matrix of the origianl matrix;
//@param:	rowIds			A vector containing row IDs to be kept;
//@ret:		void	
void GetSubMatrix(std::shared_ptr<Eigen::MatrixXf> oMatrix, std::shared_ptr<Eigen::MatrixXf> subMatrix, std::vector<int> rowIds)
{
#pragma omp parallel for  collapse(2)
	for (int i = 0; i < rowIds.size(); i++)
	{
		for (int j = 0; j < oMatrix->cols(); j++)
		{
			(*subMatrix)(i, j) = (*oMatrix)(rowIds[i], j);
		}
	}
}

//@brief:	Get a subset of a float matrix given specific row IDs;
//@param:	oMatrix			The original matrix;
//@param:	subMatrix		A subset matrix of the origianl matrix;
//@param:	rowIds			A vector containing row IDs to be kept;
//@ret:		void	
void GetSubMatrix(Eigen::MatrixXf &oMatrix, Eigen::MatrixXf &subMatrix, std::vector<int> rowIds)
{
#pragma omp parallel for  collapse(2)
	for (int i = 0; i < rowIds.size(); i++)
	{
		for (int j = 0; j < oMatrix.cols(); j++)
		{
			subMatrix(i, j) = oMatrix(rowIds[i], j);
		}
	}
}

//@brief:	Get a subset of a float vector given specific IDs;
//@param:	oVector			The original vector;
//@param:	subVector		A subset vector of the origianl matrix;
//@param:	IDs				A vector containing IDs to be kept;
//@ret:		void	
void GetSubVector(Eigen::VectorXf & oVector, Eigen::VectorXf & subVector, std::vector<int> IDs)
{
	for (int i=0;i<IDs.size();i++)
	{
		subVector(i) = oVector(IDs[i]);
	}
}



std::vector<std::string> UniqueCount(std::vector<std::string> vec)
{
	sort(vec.begin(), vec.end());
	vec.erase(unique(vec.begin(), vec.end()), vec.end());
	return vec;
}
double normalCDF(double x, bool lowerTail)
{
	double cdf = std::erfc(-x / std::sqrt(2)) / 2;
	if (!lowerTail)
	{

		cdf = 1 - cdf;
	}
	return cdf;
}

/*
//@brief:	Initialize class ROC;
//@param:	Response		A vector of responses;
//@param:	Predictor		A vector of predicts;
//@ret:		void;
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

//@brief:	Initialize;
//@ret:		void;
void ROC::init()
{
	thresholds.resize(502);
	Specificity.resize(502);
	Sensitivity.resize(502);
	FPR.resize(502);
	thresholds.setZero();
	Specificity.setZero();
	Sensitivity.setZero();
	FPR.setZero();
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

//@brief:	Calculating Sensitivity and Specificity given thresholds;
//@ret:		void;
void ROC::Calc()
{
	for (int i = 0; i < thresholds.size(); i++)
	{
		
		long long TruePos = 0;
		long long  FaslePos = 0;
		long long  TrueNeg = 0;
		long long  FasleNeg = 0;
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
				if (abs(Response[j] -1)< 1e-7)
					TruePos++;
				else
					FaslePos++;
			}
		}
		Sensitivity[i] = (float)TruePos / (float)(TruePos+FasleNeg);
		Specificity[i] = (float)TrueNeg / (float)(TrueNeg+FaslePos);
		FPR[i]= (float)FaslePos / (float)(TrueNeg + FaslePos);
	}
}

//@brief:	Calculating AUC;
//@ret:		void;
void ROC::AUC()
{
	float q1, q2, p1, p2;
	q1 = Sensitivity[0];
	q2 = FPR[0];
	float area = 0.0;
	for (int i = 1; i < Sensitivity.size(); ++i) {
		p1 = FPR[i];
		p2 = Sensitivity[i];
		area += sqrt(pow(((1 - q1) + (1 - p1)) / 2 * (q2 - p2), 2));
		q1 = p1;
		q2 = p2;
	}
	/*for (size_t i = 1; i < Sensitivity.size(); i++)
	{
		double height = (Sensitivity[i] + Sensitivity[i-1])/2;
		auc += height * abs(Specificity[i]- Specificity[i - 1]);
	}

	
	
	this->auc = area;
}
*/

Evaluate::Evaluate()
{
	isnull = 1;
	cor.resize(1);
	cor[0] = -9;
}

Evaluate::Evaluate(Eigen::VectorXf Response, Eigen::VectorXf Predictor, int dataType)
{
	this->dataType = dataType;
	Eigen::MatrixXf y = Eigen::Map<Eigen::MatrixXf>(Response.data(), Response.rows(), 1);
	Eigen::MatrixXf y_hat = Eigen::Map<Eigen::MatrixXf>(Predictor.data(), Predictor.rows(), 1);
	if (dataType == 0)
	{
		mse = calc_mse(y, y_hat);
		auc = calc_cor(Response, Predictor);
	}
	else
	{
		Eigen::VectorXf ref = Eigen::Map<Eigen::VectorXf>(y.data(), y.rows(), 1);
		Eigen::MatrixXf pred_matrix(y_hat.size(), 2);
		pred_matrix.setOnes();
		pred_matrix.col(0) = pred_matrix.col(1) - y_hat.col(0);
		pred_matrix.col(1) = y_hat.col(0);
		auc = multiclass_auc(pred_matrix, ref.cast<int>());
		if (dataType == 1)
		{
			
	
			if (auc < 0.5)
			{
				auc = 1 - auc;
				y_hat = Eigen::VectorXf::Ones(y_hat.rows()); - y_hat;
			}
		/*
			ROC roc(Response, Predictor);
			auc = roc.GetAUC();
			*/
			y_hat = y_hat.array() + 0.5;
			y_hat = y_hat.cast<int>().cast<float>();
			mse = calc_mse(y, y_hat);
		}
		else
		{
			mse = misclass(y, y_hat);
		}
		
	}
	
}

Evaluate::Evaluate(torch::Tensor Response, torch::Tensor Predictor, int dataType)
{
	this->dataType = dataType;
	Eigen::MatrixXf y = dtt::libtorch2eigen<double>(Response).cast<float>();
	Eigen::MatrixXf y_hat = dtt::libtorch2eigen<double>(Predictor).cast<float>();
	if (dataType==0)
	{
	//	Eigen::VectorXf ref = Eigen::Map<Eigen::VectorXf>(y.data(), y.rows(), 1);
	//	Eigen::VectorXf pred = Eigen::Map<Eigen::VectorXf>(y_hat.data(), y_hat.rows(), 1);
		mse = calc_mse(y, y_hat);
		cor = calc_cor(y, y_hat);
	}
	else
	{
		Eigen::VectorXf ref = Eigen::Map<Eigen::VectorXf>(y.data(), y.rows(), 1);
		if (dataType ==1 )
		{
				
			Eigen::MatrixXf pred_matrix(y_hat.size(), 2);
			pred_matrix.setOnes();
			pred_matrix.col(0) = pred_matrix.col(1) - y_hat.col(0);
			pred_matrix.col(1) = y_hat.col(0);
		//	std::cout << pred_matrix << std::endl;
			auc = multiclass_auc(pred_matrix, ref.cast<int>());
			if (auc < 0.5)
			{
				auc = 1 - auc;
				//y_hat = Eigen::VectorXf::Ones(y_hat.rows()) -y_hat;
			}
		
		//	Eigen::VectorXf y_predict = Eigen::Map<Eigen::VectorXf>(y_hat.data(), y_hat.rows(), 1);
		//	ROC roc(ref, y_predict);
		//	auc = roc.GetAUC();
		//	auc = AUROC<float, float>(ref.data(), y_hat.data(), y.size());
			y_hat = y_hat.array() + 0.5;
			y_hat = y_hat.cast<int>().cast<float>();
			mse = calc_mse(y, y_hat);
		}
		else
		{
			auc = multiclass_auc(y_hat, ref.cast<int>());
			Predictor = torch::argmax(Predictor,1,true);
			std::cout << Predictor << std::endl;
			y_hat = dtt::libtorch2eigen<int64_t>(Predictor).cast<float>();
			mse = misclass(y, y_hat);
		
		}
	}
	
}

void Evaluate::test()
{
	
	Eigen::VectorXf Y_binary(10);
	Eigen::VectorXf Y_multi(10);
	Eigen::VectorXf Y_continue = Eigen::VectorXf::Random(10);
	Eigen::MatrixXf pred(10,3);
	Eigen::VectorXf pred_b(10);
	Y_binary << 0,
		0,
		0,
		1,
		0,
		1,
		1,
		0,
		0,
		0;
	Y_multi << 0,
		2,
		1,
		0,
		2,
		0,
		0,
		0,
		2,
		0;
	pred << 0.2059746, 0.93470523, 0.4820801,
		0.1765568, 0.21214252, 0.5995658,
		0.3870228, 0.65167377, 0.4935413,
		0.8841037, 0.12555510, 0.1862176,
		0.3698414, 0.26722067, 0.8273733,
		0.7976992, 0.38611409, 0.6684667,
		0.9176185, 0.01339033, 0.7942399,
		0.4919061, 0.38238796, 0.1079436,
		0.3800352, 0.86969085, 0.7237109,
		0.1774452, 0.34034900, 0.4112744;
	pred_b << pred.col(0);
	torch::Tensor y_b = dtt::eigen2libtorch(Y_binary);
	torch::Tensor y_m = dtt::eigen2libtorch(Y_multi);
	torch::Tensor y_c = dtt::eigen2libtorch(Y_continue);
	torch::Tensor p_m = dtt::eigen2libtorch(pred);
	torch::Tensor p_b = dtt::eigen2libtorch(pred_b);
	Evaluate eva(y_b, p_b, 1);
	std::cout << "misclassification rate: " << eva.getMSE() << " AUC: " << eva.getAUC() << std::endl;
	Evaluate eva2(y_m, p_m, 2);
	std::cout << "misclassification rate: " << eva2.getMSE() << " AUC: " << eva2.getAUC() << std::endl;

}

float Evaluate::calc_mse(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Real_Y, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Predict_Y)
{
	assert(Real_Y.size() == Predict_Y.size());
	int64_t nind = Real_Y.size();
	Eigen::MatrixXd residual = (Real_Y - Predict_Y).cast<double>();
	double mse_tmp = std::pow(residual.norm(),2);
	mse_tmp /= (double)nind;

	return (float)mse_tmp;;
}

Eigen::VectorXf Evaluate::calc_cor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Real_Y, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Predict_Y)
{
	Eigen::VectorXf corr(Real_Y.cols());
	for (size_t i = 0; i < Real_Y.cols(); i++)
	{
		Eigen::VectorXf Y_col = Real_Y.col(i);
		Eigen::VectorXf Y_hat_col= Predict_Y.col(i);
		corr(i) = calc_cor(Y_col, Y_hat_col);
	}
	return corr;
}

//@brief:	Calucating correlation coefficient between vector Y1 and vector Y2;
//@param:	Real_Y		A vector;
//@param:	Predict_Y		Another vector;
//@ret:		float	correlation coefficient
float Evaluate::calc_cor(Eigen::VectorXf& Real_Y, Eigen::VectorXf& Predict_Y)
{
	assert(Real_Y.size() == Predict_Y.size());
	Eigen::VectorXd Y1_double = Real_Y.cast<double>();
	Eigen::VectorXd Y2_double = Predict_Y.cast<double>();
	Y1_double = Y1_double - Y1_double.mean()*Eigen::VectorXd::Ones(Real_Y.size());
	Y2_double = Y2_double - Y2_double.mean() * Eigen::VectorXd::Ones(Real_Y.size());
	double cosin = Y1_double.dot(Y2_double);
	cosin /= Y1_double.lpNorm<2>() * Y2_double.lpNorm<2>();
		/*
		double nind = Real_Y.size();
		double upper = 0;
		double lower = 0;
		Eigen::VectorXd Y1_double = Real_Y.cast<double>();
		Eigen::VectorXd Y2_double = Predict_Y.cast<double>();
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
		*/
	return cosin;
}

Eigen::MatrixXf Evaluate::get_Y(Eigen::MatrixXf pred_y)
{
	Eigen::MatrixXf   maxIndex(pred_y.rows(),1);
	//#pragma omp parallel for
	for (int64_t i; i < pred_y.rows(); i++)
	{
		int index;
		pred_y.row(i).maxCoeff(&index);
		maxIndex(i) = index;
	}
	return maxIndex;
}

float Evaluate::misclass(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Real_Y, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Predict_Y)
{
	Eigen::MatrixXd res = (Real_Y - Predict_Y).cast<double>();
	res = res.cwiseAbs();
	res = res.cast<bool>().cast<double>();
	double miss = res.size()-res.sum();
	miss /= (double)res.size();
	return (float) miss;
}


//computes A(i | j), the probability that a randomly
//chosen member of class j has a lower estimated probability(or score)
// of belonging to class i than a randomly chosen member of class i
float Evaluate::compute_A_conditional(Eigen::MatrixXf pred_matrix, int i, int j, Eigen::VectorXi ref)
{
	// select predictions of class members
	std::vector<float> pred_i, pred_j;
	for (int k = 0; k < ref.size(); k++)
	{
		if (ref[k]==i)
		{
			pred_i.push_back(pred_matrix(k, i));
		}
		if (ref[k] == j)
		{
			pred_j.push_back(pred_matrix(k, i));
		}
	}
	double ni = pred_i.size();
	double nj = pred_j.size();
	std::vector<int> classes(pred_i.size(), i);
	classes.insert(classes.end(), pred_j.size(), j);
	pred_i.insert(pred_i.end(), pred_j.begin(), pred_j.end());
	auto rank = sort_indexes(pred_i);
	std::vector<int> classes_rank(classes.size());
	for (size_t k = 0; k < rank.size(); k++)
	{
		classes_rank[k]=classes[rank[k]];
	}
	// Si: sum of ranks from class i observations
	double Si=0;
	for (size_t k = 1; k <= classes_rank.size(); k++)
	{
		if (classes_rank[k-1]==i)
		{
			Si += k;
		}
	}
	//calculate A(i | j)
	double A = ((Si - ((ni * (ni + 1)) / 2)) / (ni * nj));
	return (float)A;
}

float Evaluate::multiclass_auc(Eigen::MatrixXf pred_matrix, Eigen::VectorXi ref)
{
	int levels = pred_matrix.cols();
	double auc=0;
	for (size_t i = 0; i < levels-1; i++)
	{
		for (size_t j = i+1; j < levels; j++)
		{
//			std::cout << i << "\t" << j << std::endl;
			double A_ij = compute_A_conditional(pred_matrix, i, j, ref);
			double A_ji = compute_A_conditional(pred_matrix, j, i, ref);
			auc += (A_ij + A_ji) / 2.0;
		}
	}
	auc = auc * 2 / (double)(levels * (levels - 1));
	return (float) auc;
}
std::shared_ptr<Dataset> Dataset::wide2long()
{
	std::shared_ptr<Dataset> datasetL = std::make_shared<Dataset>();
	std::vector<float> y_tmp;
	boost::bimap<int, std::string> fid_iid_long;
//	std::vector<float> loc_tmp;
	int id = 0;
	for (int64_t i = 0; i < this->phe.vPhenotype.size(); i++)
	{
	
		for (int64_t j = 0; j < this->phe.vPhenotype[i].size(); j++)
		{
			Eigen::VectorXf newphe(1);
			Eigen::VectorXf newloc(1);
			newphe[0] = this->phe.vPhenotype[i][j];
			newloc(0) = this->phe.vloc[i][j];
			y_tmp.push_back(this->phe.vPhenotype[i][j]);
		//	loc_tmp.push_back(this->phe.vloc[i][j]);
			datasetL->phe.vloc.push_back(newloc);
			datasetL->phe.vPhenotype.push_back(newphe);
			std::stringstream ss;
			ss << id << "_" << id;
			fid_iid_long.insert({ id, ss.str() });
			id++;
		}
	
	}
	datasetL->phe.Phenotype = Eigen::Map<Eigen::MatrixXf>(y_tmp.data(), y_tmp.size(), 1);
//	datasetL->phe.loc = Eigen::Map<Eigen::VectorXf>(loc_tmp.data(), loc_tmp.size(), 1);
	datasetL->phe.fid_iid = fid_iid_long;
	datasetL->phe.isBalance = true;
	datasetL->phe.mean = this->phe.mean;
	datasetL->phe.std = this->phe.std;
	datasetL->phe.nind = y_tmp.size();
	datasetL->phe.dataType = this->phe.dataType;
	datasetL->cov.Covariates.resize(y_tmp.size(), this->cov.npar);
	datasetL->geno.Geno.resize(y_tmp.size(), this->geno.Geno.cols());
	id = 0;
	for (int64_t i = 0; i < this->phe.vPhenotype.size(); i++)
	{

		for (int64_t j = 0; j < this->phe.vPhenotype[i].size(); j++)
		{
			if (this->cov.npar)
			{
				datasetL->cov.Covariates.row(id) << this->cov.Covariates.row(i);
			}
			datasetL->geno.Geno.row(id) << this->geno.Geno.row(i);
			id++;
		}

	}
	datasetL->geno.fid_iid = fid_iid_long;
	datasetL->geno.pos = this->geno.pos;
	datasetL->cov.fid_iid = fid_iid_long;
	datasetL->cov.names = this->cov.names;
	datasetL->cov.nind = y_tmp.size();
	datasetL->cov.npar = this->cov.npar;
	return datasetL;

}
std::tuple<std::shared_ptr<Dataset>, std::shared_ptr<Dataset>> Dataset::split(float seed, float ratio)
{
	
	std::default_random_engine e(seed);
	int64_t num = phe.fid_iid.size();
	int64_t train_num = (float)num * ratio;
	std::vector<int64_t> fid_iid_split(num);
	std::iota(std::begin(fid_iid_split), std::end(fid_iid_split), 0); // Fill with 0, 1, ..., 99.
	std::shuffle(fid_iid_split.begin(), fid_iid_split.end(), e);
	std::shared_ptr<Dataset> train = std::make_shared<Dataset>();
	std::shared_ptr<Dataset> test = std::make_shared<Dataset>();
	train->phe = phe;
	train->cov = cov;
	train->geno = geno;
	test->phe = phe;
	test->cov = cov;
	test->geno = geno;
	if (phe.isBalance)
	{
		train->phe.Phenotype.resize(train_num, phe.Phenotype.cols());
		test->phe.Phenotype.resize(num - train_num, phe.Phenotype.cols());
		//Check if the response are 1d, and will be interpolated at a single knot.
		//If the responses are multivariate, and will be interpolated at same multi-knots, keep the knots vectors in train, and test dataset
		//otherwise, the knots will be chosen according to response
		if (phe.Phenotype.cols() == 1 && phe.loc.size() > 0)
		{
			train->phe.loc.resize(train_num);
			test->phe.loc.resize(num - train_num);
		}
	}
//	else
//	{
		train->phe.vPhenotype.clear();
		train->phe.vloc.clear();
		test->phe.vPhenotype.clear();
		test->phe.vloc.clear();
	//}
	train->phe.fid_iid.clear();
	train->cov.fid_iid.clear();
	train->geno.fid_iid.clear();
	train->cov.Covariates.resize(train_num, cov.npar);
	train->geno.Geno.resize(train_num, geno.pos.size());
	////////////////////////////////////////
	test->phe.fid_iid.clear();
	test->cov.fid_iid.clear();
	test->geno.fid_iid.clear();
	test->cov.Covariates.resize(num - train_num, cov.npar);
	test->geno.Geno.resize(num - train_num, geno.pos.size());
	boost::bimap<int, std::string> fid_iid_train;
	boost::bimap<int, std::string> fid_iid_test;
	int  train_id = 0;
	int test_id = 0;
	for (int64_t i = 0; i < num; i++)
	{
		//std::string rowID = it_row->second;
		int row_index = fid_iid_split[i];
		auto fid_iid = phe.fid_iid.left.find(row_index);
	//	std::cout << fid_iid->first << "\t" << fid_iid->second << std::endl;
		if (i < train_num)
		{
			if (cov.nind)
			{
				train->cov.Covariates.row(train_id) << cov.Covariates.row(row_index);
			}
			if (geno.fid_iid.size() != 0)
			{
				train->geno.Geno.row(train_id) << geno.Geno.row(row_index);
			}

			if (phe.isBalance)
			{
				train->phe.Phenotype.row(train_id) = phe.Phenotype.row(row_index);
				if (phe.Phenotype.cols() == 1 && phe.loc.size() > 0)
				{
					train->phe.loc(train_id) = phe.loc(row_index);
				}
			
			}
	//		else
	//		{
				train->phe.vPhenotype.push_back(phe.vPhenotype[row_index]);
				if (phe.vloc.size())
				{
					train->phe.vloc.push_back(phe.vloc[row_index]);
				}
			
	//		}
			fid_iid_train.insert({ train_id++,fid_iid->second });
		}
		else
		{
			if (cov.nind)
			{
				test->cov.Covariates.row(test_id) << cov.Covariates.row(row_index);
			}
			if (geno.fid_iid.size() != 0)
			{
				test->geno.Geno.row(test_id) << geno.Geno.row(row_index);
			}
			if (phe.isBalance)
			{
				test->phe.Phenotype.row(test_id) = phe.Phenotype.row(row_index);
				if (phe.Phenotype.cols() == 1 && phe.loc.size() > 0)
				{
					test->phe.loc(test_id) = phe.loc(row_index);
				}

			}
	//		else
	//		{
				test->phe.vPhenotype.push_back(phe.vPhenotype[row_index]);
				if (phe.vloc.size())
				{
					test->phe.vloc.push_back(phe.vloc[row_index]);
				}
			//	test->phe.vloc.push_back(phe.vloc[row_index]);
	//		}
			fid_iid_test.insert({ test_id++ ,fid_iid->second });
		}
	}
	train->phe.fid_iid = fid_iid_train;
	train->cov.fid_iid = fid_iid_train;
	train->geno.fid_iid = fid_iid_train;
	train->phe.nind = fid_iid_train.size();
	test->phe.fid_iid = fid_iid_test;
	test->cov.fid_iid = fid_iid_test;
	test->geno.fid_iid = fid_iid_test;
	test->phe.nind = fid_iid_test.size();
	return std::make_tuple(train, test);
}