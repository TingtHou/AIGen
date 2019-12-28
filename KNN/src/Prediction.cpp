#include "../include/Prediction.h"

Prediction::Prediction(Eigen::VectorXf& Real_Y, std::vector<Eigen::MatrixXf>&  Kernels, Eigen::VectorXf& vcs, Eigen::MatrixXf& X, Eigen::VectorXf& fixed)
{
	this->Real_Y = Real_Y;
	this->Kernels = Kernels;
	this->vcs = vcs;
	this->X = X;
	this->fixed = fixed;
	nind = Real_Y.size();
	ncvs = vcs.size();
	Predict_Y.resize(nind);
	Predict_Y.setZero();
	Eigen::MatrixXf Identity(nind, nind);
	Identity.setIdentity();
	this->Kernels.push_back(Identity);
	calc();
}

float Prediction::getMSE()
{
	return mse;
}

float Prediction::getCor()
{
	return cor;
}



void Prediction::GetpredictY()
{
	Predict_Y= X * fixed;
	std::ofstream test("predictY.txt");
	test << Predict_Y << std::endl;
	test.close();
/*
	Eigen::VectorXf mu(nind);
	mu.setZero();
	mu = X * fixed;
	for (int i = 0; i < ncvs; i++)
	{
		Eigen::MatrixXf Sigma(nind, nind);
		Sigma = vcs[i] * Kernels[i];
		mvnorm mvn(1, mu, Sigma);
		Predict_Y += mvn.rmvnorm().col(0);
	}
	std::ofstream test("predictY.txt");
	test << Predict_Y << std::endl;
	test.close();
	*/
}

void Prediction::calc_mse()
{
	Eigen::VectorXf residual = Real_Y - Predict_Y;
	double mse_tmp = 0;
	for (int i = 0; i < nind; i++)
	{
		mse_tmp += ((double)residual[i])* ((double)residual[i]);
	}
	mse_tmp /= nind;
	mse = (float)mse_tmp;
}

void Prediction::calc_cor()
{
	double meanRealY, meanPredY;
	meanPredY = meanRealY = 0;
	double upper = 0;
	double lower = 0;
	Eigen::VectorXd PredY_double = Predict_Y.cast<double>();
	Eigen::VectorXd RealY_double = Real_Y.cast<double>();
	Eigen::VectorXd RealY_PredY = RealY_double.cwiseProduct(PredY_double);
	double PredY_sum = PredY_double.sum();
	double RealY_sum = RealY_double.sum();
	double RealY_squre_sum = RealY_double.cwiseProduct(RealY_double).sum();
	double PredY_squre_sum = PredY_double.cwiseProduct(PredY_double).sum();
	double RealY_PredY_sum = RealY_PredY.sum();
	upper = nind * RealY_PredY_sum - RealY_sum * PredY_sum;
	lower = std::sqrt(nind * RealY_squre_sum - RealY_sum * RealY_sum) * std::sqrt(nind * PredY_squre_sum - PredY_sum * PredY_sum);
	cor = (float)upper / lower;
	
}

void Prediction::calc()
{
	GetpredictY();
	calc_mse();
	calc_cor();

}
