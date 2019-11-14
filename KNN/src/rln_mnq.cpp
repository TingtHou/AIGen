#include "../include/rln_mnq.h"


rln_mnq::rln_mnq(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomp = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomp = altDecompositionMode;
}

void rln_mnq::estimate()
{
	Eigen::MatrixXd Identity(nind, nind);
	Identity.setIdentity();
	vcs.resize(nVi);
	if (W.size()==0)
	{
		W = Eigen::VectorXd(nVi);
		W.setOnes();
	}
	double* pr_VW = VW.data();
	double* pr_Identity = Identity.data();
	for (int i = 0; i < nVi; i++)
	{
		double* pr_Vi = Vi[i].data();
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, W[i], pr_Vi, nind, pr_Identity, nind, 1, pr_VW, nind);
	}
	int status=Inverse(VW, Decomp, altDecomp, allowPseudoInverse);
	CheckInverseStatus(status);
	std::vector<Eigen::MatrixXd> RV(nVi);
	Eigen::VectorXd Ry(nind);
	Eigen::MatrixXd C;
	if (ncov == 0)
	{
		for (int i = 0; i < nVi; i++)
		{
			RV[i] = Eigen::MatrixXd(nind, nind);
			double* pr_Vi = Vi[i].data();
			double* pr_rvi = RV[i].data();
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, 1, pr_VW, nind, pr_Vi, nind, 0, pr_rvi, nind);

		}
		double* pr_Y = Y.data();
		double* pr_RY = Ry.data();
		cblas_dgemv(CblasColMajor, CblasNoTrans, nind, nind, 1, pr_VW, nind, pr_Y, 1, 0, pr_RY, 1);
	//	Ry = VW * Y;
	}
	else
	{
		Eigen::MatrixXd B(nind, X.cols());
		Eigen::MatrixXd Xt_B(X.cols(), X.cols());
		C.resize(X.cols(), nind);
		double* pr_X = X.data();
		double* pr_Y = Y.data();
		double* pr_RY = Ry.data();
		double* pr_B = B.data();
		double* pr_Xt_B = Xt_B.data();
		double* pr_C = C.data();
		double* p_mkl= (double*)mkl_calloc(nind * nind, sizeof(double), 64);
		//B=(VM_inv)*X
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), nind, 1, pr_VW, nind, pr_X, nind, 0, pr_B, nind);  
		//XtB=Xt*B
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,  X.cols(), X.cols(),nind, 1, pr_X, nind, pr_B, nind, 0, pr_Xt_B, X.cols());
		//inv(XtB)
		int status = Inverse(Xt_B, Decomp, altDecomp, allowPseudoInverse);
		CheckInverseStatus(status);
		//C=XtB*Bt
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, X.cols(), nind, X.cols(), 1, pr_Xt_B, X.cols(), pr_B, nind, 0, pr_C, X.cols());
		//P=X*C
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, X.cols(), 1, pr_X, nind, pr_C, X.cols(), 0, p_mkl, nind);
		double *inv_vw_mkl = VW.data();
		int* dim_array = new int[nVi];
		double* alpha1_array = new double[nVi];
		double* alpha2_array = new double[nVi];
		double* beta_array= new double[nVi];
		double** a_array;
		double** b_array;
		double** c_array;
		double** d_array;
		double** inv_vw_mkl_array;
		a_array = (double**)malloc(sizeof(double*) * nVi);
		b_array = (double**)malloc(sizeof(double*) * nVi);
		d_array = (double**)malloc(sizeof(double*) * nVi);
		c_array = (double**)malloc(sizeof(double*) * nVi);
		inv_vw_mkl_array = (double**)malloc(sizeof(double*) * nVi);
		CBLAS_TRANSPOSE* transa_array = new CBLAS_TRANSPOSE[nVi];
		CBLAS_TRANSPOSE* transb_array = new CBLAS_TRANSPOSE[nVi];
		int* group_size = new int[nVi];
		for (int i = 0; i < nVi; i++)
		{
			dim_array[i] = nind;
			alpha1_array[i] = 1;
			alpha2_array[i] = -1;
			beta_array[i] = 0;
			a_array[i] = p_mkl;
			inv_vw_mkl_array[i] = inv_vw_mkl;
			b_array[i]= (Vi[i]).data();
			RV[i].resize(nind, nind);
			RV[i].setZero();
			c_array[i] = RV[i].data();
			transa_array[i] = transb_array[i] = CblasNoTrans;
			group_size[i] = 1;
			d_array[i]= (double*)mkl_calloc(nind * nind, sizeof(double), 64);
			mkl_domatcopy('c', 'n', nind, nind, 1, b_array[i], nind, d_array[i], nind);
		}
		
		cblas_dgemm_batch(CblasColMajor, transa_array, transb_array, dim_array, dim_array, dim_array, 
			alpha2_array, (const double**)a_array, dim_array, (const double**)b_array, dim_array, alpha1_array, d_array, dim_array, nVi, group_size);
		cblas_dgemm_batch(CblasColMajor, transa_array, transb_array, dim_array, dim_array, dim_array,
			alpha1_array, (const double**)inv_vw_mkl_array, dim_array, (const double**) d_array, dim_array, beta_array, c_array, dim_array, nVi, group_size);
		delete[] dim_array;
		delete[] alpha1_array;
		delete[] alpha2_array;
		delete[] beta_array;
		delete[] transa_array;
		delete[] transb_array;
		free(a_array);
		free(b_array);
		free(c_array);
		free(d_array);
		free(inv_vw_mkl_array);
		double* pr_Ycopy = (double*)mkl_calloc(nind, sizeof(double), 64);
		cblas_dcopy(nind, pr_Y, 1, pr_Ycopy, 1);
		cblas_dgemv(CblasColMajor, CblasNoTrans, nind, nind, -1, p_mkl, nind, pr_Y, 1, 1, pr_Ycopy, 1);
		cblas_dgemv(CblasColMajor, CblasNoTrans, nind, nind, 1, pr_VW, nind, pr_Ycopy, 1, 0, pr_RY, 1);
		mkl_free(p_mkl);
		mkl_free(pr_Ycopy);
	}
	Eigen::VectorXd u(nVi);
	u.setZero();
//	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		Eigen::VectorXd Vi_Ry = Vi[i]*Ry;
		u[i] = Ry.cwiseProduct(Vi_Ry).sum();
	}
	Eigen::MatrixXd F(nVi, nVi);
//	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		for (int j=i;j<nVi;j++)
		{
			F(i, j) = RV[i].transpose().cwiseProduct(RV[j]).sum();
			F(j, i) = F(i, j);
		}
	}
	status = Inverse(F, Decomp, altDecomp, allowPseudoInverse);
	CheckInverseStatus(status);
	vcs = F * u;
	if (ncov)
	{
		fix = C * Y;
	}
//	std::cout << "other  elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
}

rln_mnq::~rln_mnq()
{
}

void rln_mnq::CheckInverseStatus(int status)
{
	switch (status)
	{
	case 0:
		break;
	case 1:
		if (allowPseudoInverse)
		{
			stringstream ss;
			ss << YELLOW << "[Warning]: " << WHITE << "Thread ID: " << ThreadId
				<< "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead\n";
			printf("%s", ss.str().c_str());
			//			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
			LOG(WARNING)<< "Thread ID: " << ThreadId << "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead";
		}
		else
		{
			stringstream ss;
			ss << RED << "[Error]: " << WHITE << "Thread ID: " << ThreadId
				<< "\tcalculating inverse matrix is failed, and pseudo inverse matrix is not allowed\n";
			throw std::exception(logic_error(ss.str().c_str()));
		}
		break;
	case 2:
	{
		stringstream ss;
		ss << RED << "[Error]: " << WHITE << "Thread ID: " << ThreadId
			<< "\tcalculating inverse matrix is failed, and pseudo inverse matrix is also failed\n";
		throw std::exception(logic_error(ss.str().c_str()));
	}
	break;
	default:
		stringstream ss;
		ss << RED << "[Error]: " << WHITE << "Thread ID: " << ThreadId
			<< "\tunknown code [" << std::to_string(status) << "] from calculating inverse matrix.\n";
		throw std::exception(logic_error(ss.str().c_str()));
	}
}

void rln_mnq::Calc(double* vi, double* rvi, double* inv_vw_mkl, double* p_mkl, int nind)
{
	double* c = (double*)mkl_calloc(nind * nind, sizeof(double), 64);
	mkl_domatcopy('c', 'n', nind, nind, 1, vi, nind, c, nind);
	//		clock_t t2 = clock();
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		nind, nind, nind, -1, p_mkl, nind, vi, nind, 1, c, nind);
	//		std::cout << "first sum elapse Time : " << (clock() - t2) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		//	t2 = clock();
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		nind, nind, nind, 1, inv_vw_mkl, nind, c, nind, 0, rvi, nind);
	//		std::cout << "first sum elapse Time : " << (clock() - t2) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	mkl_free(c);
}



