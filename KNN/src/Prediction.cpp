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
	if (mode)
	{
		PredictYLOO();
	}
	else
	{
		predictY();
	}
}



void Prediction::predictY()
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




void Prediction::PredictYLOO()
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
