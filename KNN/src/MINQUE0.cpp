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
	float* pr_VW = VW.data();
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
		Eigen::MatrixXf inv_XtX_Xt(X.cols(), nind);
		float* pr_X = X.data();
		float* pr_Y = Y.data();
		float* pr_RY = Ry.data();
		float* pr_Xt_X = Xt_X.data();
		float* pr_inv_XtX_Xt = inv_XtX_Xt.data();
		//Xt_X=Xt*X
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_X, nind, 0, pr_Xt_X, X.cols());
		//inv(XtX)
		int status = Inverse(Xt_X, Decomposition, altDecomposition, allowPseudoInverse);
		CheckInverseStatus(status);
		//inv_XtX_Xt=inv(XtX)*Xt
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, X.cols(), nind, X.cols(), 1, pr_Xt_X, X.cols(), pr_X, nind, 0, pr_inv_XtX_Xt, X.cols());
		//inv_VM=I-X*inv(XtX)*Xt
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, X.cols(), -1, pr_X, nind, pr_inv_XtX_Xt, X.cols(), 1, pr_VW, nind);
		//MKL_free(pr_VWp);
		Ry = VW * Y;
	}
	#pragma omp parallel for shared(pr_VW)
	for (int i = 0; i < nVi; i++)
	{
		cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, nind, nind, 1, pr_VW, nind, pr_vi_list[i], nind, 0, pr_rv_list[i], nind);
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
	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		for (int j = i; j < nVi; j++)
		{
			Eigen::MatrixXf RVij = (RV[i].transpose().cwiseProduct(RV[j]));
			double sum = 0;
			for (int m = 0; m < nind; m++)
			{
				for (int n = m; n < nind; n++)
				{
					sum += RVij(m, n);
					if (n != m)
					{
						sum += RVij(m, n); //lower triangle
					}
				}
			}
			F(i, j) = (float)sum;
			F(j, i) = F(i, j);
		}
	}

	int status = Inverse(F, Decomposition, altDecomposition, allowPseudoInverse);
	CheckInverseStatus(status);
	vcs = F * u;
}

MINQUE0::~MINQUE0()
{
}
