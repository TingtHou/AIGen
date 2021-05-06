#pragma once
#define EIGEN_USE_MKL_ALL
#include <map>
#include <torch/torch.h>
#include <string>
#include <boost/bimap.hpp>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "ToolKit.h"
#include "easylogging++.h"
#include <thread>
#include <iomanip>
#include <numeric>
#include <vector>



namespace dtt {

	// same as MatrixXf, but with row-major memory layout
	//typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

	// MatrixXrm<double> x; instead of MatrixXf_rm x;
	template <typename V>
	using MatrixXrm = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	// MatrixX<double> x; instead of Eigen::MatrixXf x;
	template <typename V>
	using MatrixX = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>;

	template <typename V>
	using VectorX = typename Eigen::Matrix<V, Eigen::Dynamic, 1>;


	template <typename V>
	torch::Tensor eigen2libtorch(MatrixX<V>& M) {
		//auto options = torch::TensorOptions().dtype(torch::kFloat64);
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M.template cast<double>());
		std::vector<int64_t> dims = { E.rows(), E.cols() };
		auto T = torch::from_blob(E.data(), dims).clone();//.to(torch::kCPU);
		return T;
	}

	template <typename V>
	torch::Tensor eigen2libtorch(VectorX<V>& M) {
		auto options = torch::TensorOptions().dtype(torch::kFloat64);
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M.template cast<double>());
		std::vector<int64_t> dims = { E.rows(), E.cols() };
		auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
		return T;
	}


	template <typename V>
	torch::Tensor eigen2libtorch(MatrixXrm<V>& E, bool copydata = true) {
		//	auto options = torch::TensorOptions().dtype(torch::kFloat64);
		std::vector<int64_t> dims = { E.rows(), E.cols() };
		auto T = torch::from_blob(E.data(), dims);
		if (copydata)
			return T.clone();
		else
			return T;
	}


	template<typename V>
	Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> libtorch2eigen(torch::Tensor& Tin) {
		/*
		 LibTorch is Row-major order and Eigen is Column-major order.
		 MatrixXrm uses Eigen::RowMajor for compatibility.
		 */
		auto T = Tin.to(torch::kCPU);
		Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
		return E;
	}
}


enum MatrixDecompostionOptions :int
{
	Cholesky = 0,
	LU = 1,
	QR = 2,
	SVD = 3
};


enum KernelNames :int
{
	CAR = 0,
	Identity = 1,
	Product = 2,
	Polymonial = 3,
	Gaussian = 4,
	IBS = 5
};

struct MinqueOptions
{
	int iterate = 200;
	float tolerance = 1e-4;
	int MatrixDecomposition = 0;
	int altMatrixDecomposition = 3;
	bool allowPseudoInverse = true;
};

struct KernelData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::MatrixXf kernelMatrix;
	Eigen::MatrixXf VariantCountMatrix;
};

struct PhenoData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::MatrixXf Phenotype;
	std::vector<Eigen::VectorXf> vPhenotype;
	int missing = 0;
//	bool isbinary = true;
	int dataType = 0;                             //0 continue data, 1 binary data, 2 categorical data
//	Eigen::MatrixXf prevalence;
	std::vector<Eigen::VectorXf> vloc;
	Eigen::VectorXf loc;
	bool isBalance;
	bool isUnivariate;
	double mean;
	double std;
	int nind;
};

struct GenoData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::MatrixXf Geno;               //individual mode, row: individual; col: SNP;
	Eigen::VectorXi pos;
};


struct CovData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::MatrixXf Covariates;
	std::vector<std::string> names;
	int nind=0;
	int npar=0;
};


struct Dataset
{
	PhenoData phe;
	GenoData geno;
	CovData cov;
	std::shared_ptr<Dataset> wide2long();
	std::tuple<std::shared_ptr<Dataset>, std::shared_ptr<Dataset>> split(float seed, float ratio);
};


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	std::stable_sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}

int Inverse(Eigen::MatrixXf & Ori_Matrix,int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
int Inverse(Eigen::MatrixXd & Ori_Matrix, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
float Variance(Eigen::VectorXf &Y);
float mean(Eigen::VectorXf &Y);
bool isNum(std::string line);
std::string GetBaseName(std::string pathname);
std::string GetParentPath(std::string pathname);
void stripSameCol(Eigen::MatrixXf &Geno);
void stdSNPmv(Eigen::MatrixXf &Geno);
void set_difference(boost::bimap<int, std::string> &map1, boost::bimap<int, std::string> &map2, std::vector<std::string> &overlap);
boost::bimap<int, std::string> set_difference(boost::bimap<int, std::string>& map1, boost::bimap<int, std::string>& map2);
boost::bimap<int, std::string> set_difference(std::vector<boost::bimap<int, std::string>>& mapList);
void set_difference(boost::bimap<int, std::string>& map1, boost::bimap<int, std::string>& map2, boost::bimap<int, std::string>& map3, std::vector<std::string>& overlap);
void GetSubMatrix(Eigen::MatrixXf* oMatrix, Eigen::MatrixXf* subMatrix, std::vector<int> rowIds, std::vector<int> colIDs);
void GetSubMatrix(Eigen::MatrixXf* oMatrix, Eigen::MatrixXf* subMatrix, std::vector<int> rowIds);
void GetSubVector(Eigen::VectorXf &oVector, Eigen::VectorXf &subVector, std::vector<int> IDs);
std::vector<std::string> UniqueCount(std::vector<std::string> vec);

//ROC curve analysis
class ROC
{
public:
	ROC(Eigen::VectorXf& Response, Eigen::VectorXf& Predictor);
	float GetAUC() { return auc; };
	Eigen::VectorXf getSensitivity() { return Sensitivity; };
	Eigen::VectorXf getSpecificity() { return Specificity; };
private:
	Eigen::VectorXf Response;    //a  vector of responses, typically encoded with 0 (controls) and 1 (cases)
	Eigen::VectorXf Predictor;   //a numeric vector of the same length than response, containing the predicted value of each observation.
	Eigen::VectorXf thresholds; //	the thresholds at which the sensitivitiesand specificities were computed.
	Eigen::VectorXf Sensitivity;
	Eigen::VectorXf Specificity;
	int nind = 0;
	float step = 0;
	float auc = 0;
	void init();
	void Calc();
	void AUC();
};


class Evaluate
{
public:
	Evaluate();
	Evaluate(Eigen::VectorXf Response, Eigen::VectorXf Predictor, int dataType);
	Evaluate(torch::Tensor Response, torch::Tensor Predictor, int dataType);
	float getMSE() { return mse; };
	Eigen::VectorXf getCor() { return cor; };
	float getAUC() { return auc; };
	void test();
	void setMSE(double mse) { this->mse = mse; };
	int dataType;
private:
	float mse = 0;
	Eigen::VectorXf cor;
	float auc = 0;
	std::shared_ptr<ROC> ROC_ptr=nullptr;
	float calc_mse(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Real_Y, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Predict_Y);
	Eigen::VectorXf calc_cor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Real_Y, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Predict_Y);
	float calc_cor(Eigen::VectorXf& Real_Y, Eigen::VectorXf& Predict_Y);
	Eigen::MatrixXf get_Y(Eigen::MatrixXf pred_y);
	float misclass(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Real_Y, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& Predict_Y);
	float compute_A_conditional(Eigen::MatrixXf pred_matrix, int i, int j, Eigen::VectorXi ref);
	float multiclass_auc(Eigen::MatrixXf pred_matrix, Eigen::VectorXi ref);
};