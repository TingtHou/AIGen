#include "../include/MINQUE1.h"


minque1::minque1(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomposition = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomposition = altDecompositionMode;
}

void minque1::estimateVCs()
{
	VW = Eigen::MatrixXf(nind, nind);
	VW.setZero();
	Eigen::MatrixXf Identity(nind, nind);
	Identity.setIdentity();
	vcs.resize(nVi);
	if (W.size()==0)
	{
		W = Eigen::VectorXf(nVi);
		W.setOnes();
	}
	std::vector<float*> pr_vi_list(nVi);
	float* pr_VW = VW.data();
	float* pr_Identity = Identity.data();// Vi.at(nVi - 1)->data();
	LOG(INFO) << "Calcuate VW";
	//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < nVi; i++)
	{
		float* pr_Vi = (*Vi[i]).data();
		pr_vi_list[i] = Vi[i]->data();
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, W[i], pr_Vi, nind, pr_Identity, nind, 1, pr_VW, nind);
	}
	Identity.resize(0, 0);
	LOG(INFO) << "Inverse VW";
	int status=Inverse(VW, Decomposition, altDecomposition, allowPseudoInverse);
	CheckInverseStatus("V matrix",status, allowPseudoInverse);
//	std::vector<Eigen::MatrixXf> RV(nVi);
	Eigen::VectorXf Ry(nind);
//	std::vector<float*> pr_rv_list(nVi);

//	#pragma omp parallel for
	//for (int i = 0; i < nVi; i++)
//	{
	//	RV[i].resize(nind, nind);
	//	RV[i].setZero();
	//	pr_rv_list[i] = RV[i].data();
//		pr_vi_list[i] = Vi[i]->data();
//	}
	if (ncov == 0)
	{
		float* pr_Y = Y.data();
		float* pr_RY = Ry.data();
		cblas_sgemv(CblasColMajor, CblasNoTrans, nind, nind, 1, pr_VW, nind, pr_Y, 1, 0, pr_RY, 1);
	//	Ry = VW * Y;
	}
	else
	{
		Eigen::MatrixXf B(nind, X.cols());
		Eigen::MatrixXf Xt_B(X.cols(), X.cols());
		Eigen::MatrixXf inv_XtB_Bt;
		Eigen::MatrixXf B_inv_XtB_Bt(nind, nind);
		inv_XtB_Bt.resize(X.cols(), nind);
		float* pr_X = X.data();
		float* pr_Y = Y.data();
		float* pr_RY = Ry.data();
		float* pr_B = B.data();
		float* pr_Xt_B = Xt_B.data();
		float* pr_inv_XtB_Bt = inv_XtB_Bt.data();
		LOG(INFO) << "calc B=inv(VW)*X";
		//B=inv(VW)*X
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), nind, 1, pr_VW, nind, pr_X, nind, 0, pr_B, nind);
		//XtB=Xt*B
		LOG(INFO) << "calc XtB=Xt*B";
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_B, nind, 0, pr_Xt_B, X.cols());
		//inv(XtB)
		LOG(INFO) << "inverse XtB";
		int status = Inverse(Xt_B, Cholesky , SVD, false);
		LOG(WARNING) << "Check inverse status";
		CheckInverseStatus("P matrix",status, false);
		//inv_XtB_Bt=inv(XtB)*Bt
		LOG(INFO) << "calc inv_XtB_Bt=inv(XtB)*Bt";
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, X.cols(), nind, X.cols(), 1, pr_Xt_B, X.cols(), pr_B, nind, 0, pr_inv_XtB_Bt, X.cols());
		//inv_VM=inv_VM-B*inv(XtB)*Bt
		LOG(INFO) << "calc inv_VM=inv_VM-B*inv(XtB)*Bt";
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, X.cols(), -1, pr_B, nind, pr_inv_XtB_Bt, X.cols(), 1, pr_VW, nind);
		//MKL_free(pr_VWp);
		LOG(INFO) << "calc Ry = VW *Y; ";
		cblas_sgemv(CblasColMajor, CblasNoTrans, nind, nind, 1, pr_VW, nind, pr_Y, 1, 0, pr_RY, 1);
		//Ry = VW *Y; 
	}
	LOG(INFO) << "calc u";
	Eigen::VectorXf u(nVi);
	u.setZero();
	#pragma omp parallel for shared(Ry)
	for (int i = 0; i < nVi; i++)
	{
		Eigen::VectorXd Ry_Vi_Ry = (Ry.cwiseProduct((*Vi[i]) * Ry)).cast<double>();
		u[i] = (float)Ry_Vi_Ry.sum();
	}
//	int nthread = omp_get_max_threads();
	//	printf("Thread %d, Max threads for mkl %d\n", omp_get_max_threads(), mkl_get_max_threads());
//	int threadInNest = 1;
//	if (nthread != 1)
//	{
//		int tmp_thread = nthread > nVi ? nVi : nthread;
//		omp_set_num_threads(tmp_thread);
//		threadInNest = nthread/ nVi;
//	}
//	#pragma omp parallel for shared(pr_VW)
	LOG(INFO) << "calc RV";
	Eigen::MatrixXf RV(nind, nind);
	for (int i = 0; i < nVi; i++)
	{
	//	RV[i].resize(nind, nind);
		RV.setZero();
//		omp_set_num_threads(threadInNest);
		cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, nind, nind, 1, pr_VW, nind, pr_vi_list[i], nind, 0, RV.data(), nind);
		(*Vi[i]) = RV;
	}
	RV.resize(0, 0);
//	omp_set_num_threads(nthread);
	LOG(INFO) << "calc F";
	Eigen::MatrixXf F(nVi, nVi);
	//Eigen::MatrixXf F_(nVi, nVi);
	#pragma omp parallel for shared(RV,F)
	for (int k = 0; k < (nVi + 1) * nVi / 2; k++)
	{
		int i = k / nVi, j = k % nVi;
		if (j < i) i = nVi - i, j = nVi - j - 1;
		Eigen::MatrixXf RVij = (*Vi[i]).transpose().cwiseProduct((*Vi[j]));
		double sum = RVij.cast<double>().sum();
		F(i, j) = (float)sum;
		F(j, i) = F(i, j);
	}
	LOG(INFO) << "inverse F";
	status = Inverse(F, Cholesky,  SVD, true);
	CheckInverseStatus("S matrix",status,true);
	vcs = F * u;
}

minque1::~minque1()
{
}