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
	
	LOG(INFO) << "-------------------------old--------------------" << std::endl;
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
	std::stringstream debug_ss1;
	debug_ss1 << "G First 10x10: \n" << G.block(0, 0, 10, 10) << std::endl;
	debug_ss1 << "G 10x10: \n" << G.block(G.rows() - 10, G.cols() - 10, 10, 10);
	LOG(INFO) << debug_ss1.str();

	Sigma += vcs[i] * (*(Kernels[i]));
	
	std::stringstream debug_ss;
	debug_ss << "Sigma First 10x10: \n" << Sigma.block(0, 0, 10, 10) << std::endl;
	debug_ss << "Sigma 10x10: \n" << Sigma.block(Sigma.rows() - 10, Sigma.cols() - 10, 10, 10);
	LOG(INFO) << debug_ss.str() ;
	
	
	ToolKit::comput_inverse_logdet_LU_mkl(Sigma);
	
	std::stringstream debug_ss2;
	debug_ss2 << "Sigma inverse First 10x10: \n" << Sigma.block(0, 0, 10, 10) << std::endl;
	debug_ss2 << "Sigma inverse 10x10: \n" << Sigma.block(Sigma.rows() - 10, Sigma.cols() - 10, 10, 10);
	LOG(INFO) << debug_ss2.str();

	G = Sigma * G;
	std::stringstream debug_ss0;
	debug_ss0 << "Sigma^-1 G First 10x10: \n" << G.block(0, 0, 10, 10) << std::endl;
	debug_ss0 << "Sigma^-1 G lsat 10x10: \n" << G.block(G.rows() - 10, G.cols() - 10, 10, 10);
	LOG(INFO) << debug_ss0.str();

	Random = G * (Real_Y - Fix);
	LOG(INFO) << "Random First 10: \n" << Random.block(0,0,10,1) << std::endl;
	LOG(INFO) << "last First 10: \n" << Random.block(Random.rows() - 10, 0, 10, 1) << std::endl;
	Predict_Y = Fix + Random;
	
	/*
	LOG(INFO) << "-------------------------New--------------------"<< std::endl;
	
	Eigen::VectorXf Fix(nind);
	Eigen::VectorXf Random(nind);
	Fix.setZero();
	Random.setZero();
	LOG(INFO) << "Calculate fix effect in prediction";
	Fix = X * fixed;
	Eigen::MatrixXf Sigma(nind, nind);
	//Eigen::MatrixXf G(nind, nind);
	Sigma.setZero();
	//G.setZero();
	int i = 0;
	//i = 0;
	float* pr_Sigma = Sigma.data();
	//float* pr_G = G.data();
//	Eigen::MatrixXf Identity(nind, nind);
//	Identity.setIdentity();
//	float* pr_Identity = Identity.data();
	LOG(INFO) << "Calculate G matrix in prediction";
	size_t vector_size = nind;
	vector_size = vector_size* vector_size;
	LOG(INFO) << "size of elements: "<< vector_size;
	for (; i < ncvs - 1; i++)
	{
		std::stringstream debug_ss1;
		debug_ss1 << "V"<<i<<":"<< std::endl;;
		debug_ss1 << " First 10x10: \n" << Kernels[i]->block(0, 0, 10, 10) << std::endl;
		debug_ss1 << "Last 10x10: \n" << Kernels[i]->block(Kernels[i]->rows() - 10, Kernels[i]->cols() - 10, 10, 10);
		LOG(INFO) << debug_ss1.str();
		//float* pr_Vi = Kernels[i]->data();
		Sigma += vcs[i] * (*(Kernels[i]));
		//cblas_saxpy(vector_size, vcs[i], pr_Vi, 1, pr_Sigma, 1);
	//	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, vcs[i], pr_Vi, nind, pr_Identity, nind, 1, pr_Sigma, nind);
	//	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, vcs[i], pr_Vi, nind, pr_Identity, nind, 1, pr_G, nind);
	}
	float* pr_Vi = Kernels[i]->data();
	float* pr_G = Kernels[0]->data();
	LOG(INFO) << "Copy G matrix in prediction";
	//*Kernels[0] = Sigma;
	memcpy(pr_G, pr_Sigma, vector_size * sizeof(float));
//	cblas_scopy(vector_size, pr_Sigma, 1, pr_G, 1);
	std::stringstream debug_ss3;
	debug_ss3 << "G First 10x10: \n" << (*Kernels[0]).block(0, 0, 10, 10) << std::endl;
	debug_ss3 << "G Last 10x10: \n" << (*Kernels[0]).block((*Kernels[0]).rows() - 10, (*Kernels[0]).cols() - 10, 10, 10);
	LOG(INFO) << debug_ss3.str();
	//memcpy(pr_G, pr_Sigma, nind * nind * sizeof(float));
	//cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, vcs[i], pr_Vi, nind, pr_Identity, nind, 1, pr_Sigma, nind);
	LOG(INFO) << "Calculate Sigma matrix in prediction";
	//cblas_saxpy(vector_size, vcs[i], pr_Vi, 1, pr_Sigma, 1);
	Sigma += vcs[i] * (*(Kernels[i]));
	std::stringstream debug_ss4;
	debug_ss4 << "Sigma First 10x10: \n" << Sigma.block(0, 0, 10, 10) << std::endl;
	debug_ss4 << "Sigma Last 10x10: \n" << Sigma.block(Sigma.rows() - 10, Sigma.cols() - 10, 10, 10);
	LOG(INFO) << debug_ss4.str();
	//Identity.resize(0, 0);

	LOG(INFO) << "Inverse Sigma matrix in prediction";
	ToolKit::comput_inverse_logdet_LU_mkl(Sigma); 
	
	std::stringstream debug_ss5;
	debug_ss5 << "Sigma inverse First 10x10: \n" << Sigma.block(0, 0, 10, 10) << std::endl;
	debug_ss5 << "Sigma inverse 10x10: \n" << Sigma.block(Sigma.rows() - 10, Sigma.cols() - 10, 10, 10);
	LOG(INFO) << debug_ss5.str();

	float* pr_Sigma_inv_G = Kernels[1]->data();
	LOG(INFO) << "Calculate Sigma^-1 G in prediction";
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, 1, pr_Sigma, nind, pr_G , nind, 0, pr_Sigma_inv_G, nind);
	std::stringstream debug_ss6;
	debug_ss6 << "Sigma^-1 G First 10x10: \n" << Kernels[1]->block(0, 0, 10, 10) << std::endl;
	debug_ss6 << "Sigma^-1 G lsat 10x10: \n" << Kernels[1]->block(Kernels[1]->rows() - 10, Kernels[1]->cols() - 10, 10, 10);
	LOG(INFO) << debug_ss6.str();

	Eigen::VectorXf res = (Real_Y - Fix);
	LOG(INFO) << "Calculate random effect in prediction";
	//Random = (*Kernels[1]) * res;
	cblas_sgemv(CblasColMajor, CblasNoTrans, nind, nind, 1, pr_Sigma_inv_G, nind, res.data(), 1, 0, Random.data(),1);
	LOG(INFO) << "Random First 10: \n" << Random.block(0, 0, 10, 1) << std::endl;
	LOG(INFO) << "last First 10: \n" << Random.block(Random.rows() - 10, 0, 10, 1) << std::endl;
	LOG(INFO) << "Predict y";
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
