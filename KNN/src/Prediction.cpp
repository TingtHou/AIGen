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
	for (; i < ncvs - 1; i++)
	{

		Sigma += vcs[i] * (*(Kernels[i]));
		G += vcs[i] * (*(Kernels[i]));
	}
	Sigma += vcs[i] * (*(Kernels[i]));
	ToolKit::comput_inverse_logdet_LU_mkl(Sigma);
	Random = Sigma * G * (Real_Y - Fix);
	Predict_Y = Fix + Random;
	/*
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
	float* pr_Sigma = Sigma.data();
	float* pr_G = G.data();
	Eigen::MatrixXf Identity(nind, nind);
	Identity.setIdentity();
	float* pr_Identity = Identity.data();
	for (; i < ncvs - 1; i++)
	{
		//		std::stringstream debug_ss1;
		//		debug_ss1 << "V"<<i<<" in thread " << omp_get_thread_num() << std::endl;;
		//		debug_ss1 << " First 10x10: \n" << Vi[i]->block(0, 0, 10, 10) << std::endl;
		//		debug_ss1 << "Last 10x10: \n" << Vi[i]->block(Vi[i]->rows() - 10, Vi[i]->cols() - 10, 10, 10);
		//		LOG(INFO) << debug_ss1.str();
		float* pr_Vi = (*Kernels[i]).data();
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, vcs[i], pr_Vi, nind, pr_Identity, nind, 1, pr_Sigma, nind);
	//	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, vcs[i], pr_Vi, nind, pr_Identity, nind, 1, pr_G, nind);
	}
	float* pr_Vi = (*Kernels[i]).data();
	memcpy(pr_G, pr_Sigma, nind * nind * sizeof(float));
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, vcs[i], pr_Vi, nind, pr_Identity, nind, 1, pr_Sigma, nind);
//	Identity.resize(0, 0);
	/*
	int i = 0;
	for (; i < ncvs-1; i++)
	{
		
		Sigma += vcs[i] * (*(Kernels[i]));
		G += vcs[i] * (*(Kernels[i]));
	}
	Sigma += vcs[i] * (*(Kernels[i]));*/
	/*
	ToolKit::comput_inverse_logdet_LU_mkl(Sigma);
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, 1, pr_Sigma, nind, pr_G , nind, 0, pr_Identity, nind);
	Eigen::VectorXf res = (Real_Y - Fix);
	cblas_sgemv(CblasColMajor, CblasNoTrans, nind, nind, 1, pr_Identity, nind, res.data(), 1, 0, Random.data(),1);
	//Random = Identity * (Real_Y - Fix);
	Predict_Y = Fix + Random;
//	std::ofstream test("predictY.txt");
//	test << Predict_Y << std::endl;
//	test.close();
	*/
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
