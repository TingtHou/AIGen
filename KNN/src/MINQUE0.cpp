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
	//VW.setIdentity();
	//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	std::vector<Eigen::MatrixXf> RV(nVi);
	Eigen::VectorXf Ry(nind);
	std::vector<float*> pr_rv_list(nVi);
	std::vector<float*> pr_vi_list(nVi);
	Eigen::VectorXd u(nVi);
	Eigen::MatrixXd F(nVi, nVi);

	u.setZero();
	LOG(INFO) << "Getting pointer of Vi";
//	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
//		RV[i].resize(nind, nind);
//		RV[i].setZero();
//		pr_rv_list[i] = RV[i].data();
		pr_vi_list[i] = Vi[i]->data();

	}
	if (ncov == 0)
	{
		Ry =  Y;
		for (int i = 0; i < nVi; i++)
		{
			Eigen::VectorXd Ry_Vi_Ry = (Ry.cwiseProduct((*Vi[i]) * Ry)).cast<double>();
			u[i] = (float)Ry_Vi_Ry.sum();
		}
	}
	else
	{
		LOG(INFO) << "Initial Xt_X";
		Eigen::MatrixXf Xt_X(X.cols(), X.cols());

		LOG(INFO) << "Initial X_inv_XtX";
		Eigen::MatrixXf X_inv_XtX(nind, X.cols());

		LOG(INFO) << "Initial X_inv_XtX_Xt";
		Eigen::MatrixXf X_inv_XtX_Xt(nind, nind);
		float* pr_X = X.data();
		float* pr_Y = Y.data();
		float* pr_RY = Ry.data();
		float* pr_Xt_X = Xt_X.data();
		float* pr_X_inv_XtX = X_inv_XtX.data();
		float* pr_X_inv_XtX_Xt = X_inv_XtX_Xt.data();

		LOG(INFO) << "Calculate Xt_X=Xt*X";
		//Xt_X=Xt*X
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_X, nind, 0, pr_Xt_X, X.cols());
		LOG(INFO) << "Inverse Xt_X";
		//inv(XtX)
		int status = Inverse(Xt_X, Cholesky, SVD, false);
		CheckInverseStatus("P matrix",status, false);

		LOG(INFO) << "Calculate X_inv_XtX=X*inv(XtX)";
		//X_inv_XtX=X*inv(XtX)
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), X.cols(), 1, pr_X, nind, pr_Xt_X, X.cols(), 0, pr_X_inv_XtX, nind);
		LOG(INFO) << "Calculate X_inv_XtX_Xt=X_inv_XtX * Xt";
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nind, nind, X.cols(), 1, pr_X_inv_XtX, nind, pr_X, nind, 0, pr_X_inv_XtX_Xt, nind);
		LOG(INFO) << "Calculate Ry";
		Ry = Y - X_inv_XtX_Xt * Y;
	//	using namespace boost::multiprecision;
		std::vector<long double> u_dd;
	//	mpfr_float::default_precision(10);

		for (int i = 0; i < nVi; i++)
		{
			LOG(INFO) << "Calculate u "<<i ;
			Eigen::VectorXd Ry_Vi_Ry = (Ry.cwiseProduct((*Vi[i]) * Ry)).cast<double>();
	//		size_t n = Ry_Vi_Ry.size();
		//	long double sum=0.0L;

		//	#pragma omp parallel for reduction( + : sum) shared(Ry_Vi_Ry)
		//	for (size_t i = 0; i < n; i++) 
		//	{
		//		sum += Ry_Vi_Ry[i];
		//	}
			u[i] = Ry_Vi_Ry.sum();
		//	u_dd.push_back(sum);
		}
		LOG(INFO) << "u:\n" << u << std::endl;
	/*	LOG(INFO) << "u_dd:\n";
		std::stringstream ss;
		ss << "\n";
		for (int i = 0; i < nVi; i++)
		{
			ss << u_dd[i] << "\n";
		}
		LOG(INFO) << ss.str();
		/*
		int nthread = omp_get_max_threads();
	//	printf("Thread %d, Max threads for mkl %d\n", omp_get_max_threads(), mkl_get_max_threads());
		int threadInNest = 1;
		if (nthread != 1)
		{
			int tmp_thread = nthread > nVi ? nVi : nthread;
			omp_set_num_threads(tmp_thread);
			threadInNest =nthread/ nVi;
		}
		#pragma omp parallel for shared(pr_X_inv_XtX,threadInNest)
		*/
		
		for (int i = 0; i < nVi; i++)
		{
	///		omp_set_num_threads(threadInNest);
	//		printf("Thread %d, Max threads for mkl %d\n", omp_get_thread_num(), mkl_get_max_threads());
			LOG(INFO) << "initial tmp " << i;
			float* tmp;
			tmp = (float*)malloc(sizeof(float) * nind * X.cols());
			memset(tmp, 0, sizeof(float) * nind * X.cols());
			//pr_VW = Vi * X*inv(XtX)
			LOG(INFO) << "Calculate tmp = Vi * X*inv(XtX)";
			cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, nind, X.cols(), 1, pr_vi_list[i] , nind, pr_X_inv_XtX , nind, 0, tmp, nind);
			LOG(INFO) << "Calculate Vi = Vi- Vi * X*inv(XtX)";
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nind, nind, X.cols() , -1, tmp, nind, pr_X, nind, 1, pr_vi_list[i], nind);			

		//	RV[i] = (*Vi[i]) - RV[i];
			LOG(INFO) << "free tmp";
			free(tmp);
		}
	//	omp_set_num_threads(nthread);
//		printf("Thread %d, Max threads for mkl %d\n", omp_get_max_threads(), mkl_get_max_threads());
		//MKL_free(pr_VWp);
	
	}
	
	/*
//	#pragma omp parallel for shared(Ry)
	for (int i = 0; i < nVi; i++)
	{
		Eigen::VectorXd Ry_Vi_Ry = (Ry.cwiseProduct((*Vi[i]) * Ry)).cast<double>();
		u[i] = (float)Ry_Vi_Ry.sum();
	}
	*/
	//using namespace boost::multiprecision;
//	std::vector<long double> f_dd;
	//Eigen::MatrixXf F_(nVi, nVi);
//	#pragma omp parallel for shared(RV,F)
	for (int k = 0; k < (nVi+1) * nVi / 2; k++) 
	{
		int i = k / nVi, j = k % nVi;
		if (j < i) i = nVi - i , j = nVi - j - 1;
		LOG(INFO) << "Calculate RVij "<< i<<" "<<j;
		Eigen::MatrixXf RVij = (*Vi[i]).transpose().cwiseProduct(*Vi[j]);
		LOG(INFO) << "sum element in RVij";
		double sum = RVij.cast<double>().sum();
//		size_t n = RVij.cols()* RVij.rows();
//		long double sum_dd=0.0L;
//		#pragma omp parallel for reduction(+:sum_dd)
//		for (size_t id = 0; id < n; id++) 
//		{
//			size_t i = id / nind, j = id % nind;
//			sum_dd += RVij(i,j);
//		}
		LOG(INFO) << " calculate F " << i << " " << j;
		F(i, j) = sum;
		F(j, i) = F(i, j);
//		f_dd.push_back(sum_dd);
	}
	/*
	LOG(INFO) << "f_dd:\n";
	std::stringstream ss;
	ss << "\n";
	for (int i = 0; i < f_dd.size(); i++)
	{
		ss << f_dd[i] << "\n";
	}
	LOG(INFO) << ss.str();
	*/
	LOG(INFO) << "Inverse F";
	LOG(INFO) << "F:\n" << F << std::endl;
	int status = Inverse(F, Cholesky, SVD, true);
	CheckInverseStatus("S matrix", status,true);
	Eigen::VectorXd vcs_d = F * u;
	#pragma omp parallel for 
	for (int i = 0; i < nVi; i++)
	{
		vcs[i] = (float)vcs_d[i];
	}
}

MINQUE0::~MINQUE0()
{
}
