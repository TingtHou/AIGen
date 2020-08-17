#include "../include/Prediction.h"

Prediction::Prediction(Eigen::VectorXf& Real_Y, std::vector<Eigen::MatrixXf *>&  Kernels, Eigen::VectorXf& vcs, Eigen::MatrixXf& X, Eigen::VectorXf& fixed, bool isbinary, int mode)
{
	this->Real_Y = Real_Y;
	this->Kernels = Kernels;
	this->vcs = vcs;
	this->X = X;
	this->fixed = fixed;
	this->isbinary = isbinary;
	this->mode = mode;
	nind = Real_Y.size();
	ncvs = vcs.size();
	Predict_Y.resize(nind);
	Predict_Y.setZero();
	Eigen::MatrixXf Identity(nind, nind);
	Identity.setIdentity();
	this->Kernels.push_back(&Identity);
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

float Prediction::getAUC()
{
	return auc;
}



void Prediction::GetpredictY()
{
	Eigen::VectorXf Fix(nind);
	Eigen::VectorXf Random(nind);
	Fix.setZero();
	Random.setZero();
	Fix = X * fixed;
	Eigen::MatrixXf Sigma(nind, nind);
	Eigen::MatrixXf G(nind, nind);
	Sigma.setZero();
	G.setZero();
	int i = 0;
	for (; i < ncvs-1; i++)
	{
		
		Sigma += vcs[i] * (*(Kernels[i]));
		G += vcs[i] * (*(Kernels[i]));
	}
	Sigma += vcs[i] * (*(Kernels[i]));
	ToolKit::comput_inverse_logdet_LU_mkl(Sigma);
	Random = Sigma * G  * (Real_Y - Fix);
	Predict_Y = Fix + Random;
//	std::ofstream test("predictY.txt");
//	test << Predict_Y << std::endl;
//	test.close();
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



void Prediction::calc()
{
	if (mode)
	{
		GetPredictYLOO();
	}
	else
	{
		GetpredictY();
	}
	calc_mse();
	if (isbinary)
	{
		ROC Roc(Real_Y, Predict_Y);
		Specificity = Roc.getSpecificity();
		Sensitivity = Roc.getSensitivity();
		auc = Roc.GetAUC();
	}
	else
	{
		cor = Cor(Real_Y, Predict_Y);
	}
}

void Prediction::GetPredictYLOO()
{
	Eigen::VectorXf Fix(nind);
	Eigen::VectorXf Random(nind);
	Fix.setZero();
	Random.setZero();
	Fix = X * fixed;
	Eigen::MatrixXf Sigma(nind, nind);
	Sigma.setZero();
	for (int i = 0; i < ncvs; i++)
	{

		Sigma += vcs[i] * (*(Kernels[i]));
	}
	ToolKit::comput_inverse_logdet_LU_mkl(Sigma);
	Eigen::VectorXf Diag = Sigma.diagonal();
	Random = (Sigma * (Real_Y - Fix));
	Random = Random.cwiseQuotient(Diag);
	Random = (Real_Y - Fix) - Random;
	Predict_Y = Fix + Random;
}
