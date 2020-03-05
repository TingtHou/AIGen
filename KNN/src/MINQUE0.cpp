#include "../include/MINQUE0.h"

MINQUE0::MINQUE0(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomposition = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomposition = altDecompositionMode;
}

void MINQUE0::estimateVCs()
{

	vcs.resize(nVi);
	VW.setIdentity();
	//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	std::vector<Eigen::MatrixXf> RV(nVi);
	Eigen::VectorXf Ry(nind);
	std::vector<float*> pr_rv_list(nVi);
	std::vector<float*> pr_vi_list(nVi);
	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		RV[i].resize(nind, nind);
		RV[i].setZero();
		pr_rv_list[i] = RV[i].data();
		pr_vi_list[i] = Vi[i]->data();

	}
	if (ncov == 0)
	{
		Ry =  Y;
	}
	else
	{

		Eigen::MatrixXf B(nind, X.cols());
		Eigen::MatrixXf Xt_X(X.cols(), X.cols());
		Eigen::MatrixXf X_inv_XtX(X.cols(), nind);
		float* pr_X = X.data();
		float* pr_Y = Y.data();
		float* pr_RY = Ry.data();
		float* pr_Xt_X = Xt_X.data();
		float* pr_X_inv_XtX = X_inv_XtX.data();
		//Xt_X=Xt*X
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_X, nind, 0, pr_Xt_X, X.cols());
		//inv(XtX)
		int status = Inverse(Xt_X, Decomposition, altDecomposition, allowPseudoInverse);
		CheckInverseStatus(status);
		//X_inv_XtX=X*inv(XtX)
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), X.cols(), 1, pr_X, nind, pr_Xt_X, X.cols(), 0, pr_X_inv_XtX, nind);
		int nthread = omp_get_max_threads();
	//	printf("Thread %d, Max threads for mkl %d\n", omp_get_max_threads(), mkl_get_max_threads());
		int threadInNest = 1;
		if (nthread != 1)
		{
			int tmp_thread = nthread > nVi ? nVi : nthread;
			omp_set_num_threads(tmp_thread);
			threadInNest =(nthread - tmp_thread )/ nVi;
		}
		#pragma omp parallel for shared(pr_X_inv_XtX,threadInNest)
		for (int i = 0; i < nVi; i++)
		{
			omp_set_num_threads(threadInNest);
	//		printf("Thread %d, Max threads for mkl %d\n", omp_get_thread_num(), mkl_get_max_threads());
			float* tmp;
			tmp = (float*)malloc(sizeof(float) * nind * X.cols());
			memset(tmp, 0, sizeof(float) * nind * X.cols());
			//pr_VW = Vi * X*inv(XtX)
			cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, nind, X.cols(), 1, pr_vi_list[i] , nind, pr_X_inv_XtX , nind, 0, tmp, nind);
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nind, nind, X.cols() , 1, tmp, nind, pr_X, nind, 0, pr_rv_list[i], nind);
			RV[i] = (*Vi[i]) - RV[i];
			free(tmp);
		}
		omp_set_num_threads(nthread);
//		printf("Thread %d, Max threads for mkl %d\n", omp_get_max_threads(), mkl_get_max_threads());
		//MKL_free(pr_VWp);
		Ry = VW * Y;
	}
	
	Eigen::VectorXf u(nVi);
	u.setZero();
	#pragma omp parallel for shared(Ry)
	for (int i = 0; i < nVi; i++)
	{
		Eigen::VectorXd Ry_Vi_Ry = (Ry.cwiseProduct((*Vi[i]) * Ry)).cast<double>();
		u[i] = (float)Ry_Vi_Ry.sum();
	}
	Eigen::MatrixXf F(nVi, nVi);
	//Eigen::MatrixXf F_(nVi, nVi);
	#pragma omp parallel for shared(RV,F)
	for (int k = 0; k < (nVi+1) * nVi / 2; k++) 
	{
		int i = k / nVi, j = k % nVi;
		if (j < i) i = nVi - i , j = nVi - j - 1;
		Eigen::MatrixXf RVij = (RV[i].transpose().cwiseProduct(RV[j]));
		double sum = RVij.cast<double>().sum();
		F(i, j) = (float)sum;
		F(j, i) = F(i, j);
	}
	/*
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < nVi; i++)
	{
		for (int j = 0; j < nVi; j++)
		{
			Eigen::MatrixXf RVij = (RV[i].transpose().cwiseProduct(RV[j]));
			double sum = RVij.cast<double>().sum();
			F(i, j) = (float)sum;
			F(j, i) = F(i, j);
		}
	}
	*/
	int status = Inverse(F, Decomposition, altDecomposition, allowPseudoInverse);
	CheckInverseStatus(status);
	vcs = F * u;
}

MINQUE0::~MINQUE0()
{
}
