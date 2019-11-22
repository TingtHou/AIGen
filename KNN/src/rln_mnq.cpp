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
	Eigen::MatrixXd inv_XtB_Bt;
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
		Eigen::MatrixXd B_inv_XtB_Bt(nind, nind);
		inv_XtB_Bt.resize(X.cols(), nind);
		double* pr_X = X.data();
		double* pr_Y = Y.data();
		double* pr_RY = Ry.data();
		double* pr_B = B.data();
		double* pr_Xt_B = Xt_B.data();
		double* pr_inv_XtB_Bt = inv_XtB_Bt.data();
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), nind, 1, pr_VW, nind, pr_X, nind, 0, pr_B, nind);
		//XtB=Xt*B
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_B, nind, 0, pr_Xt_B, X.cols());
		//inv(XtB)
		int status = Inverse(Xt_B, Decomp, altDecomp, allowPseudoInverse);
		CheckInverseStatus(status);
		//inv_XtB_Bt=inv(XtB)*Bt
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, X.cols(), nind, X.cols(), 1, pr_Xt_B, X.cols(), pr_B, nind, 0, pr_inv_XtB_Bt, X.cols());
		//inv_VM=inv_VM-B*inv(XtB)*Bt
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, X.cols(), -1, pr_B, nind, pr_inv_XtB_Bt, X.cols(), 1, pr_VW, nind);
		std::vector<double*> pr_rv_list;
		std::vector<double*> pr_vi_list;
		for (int i = 0; i < nVi; i++)
		{
			RV[i].resize(nind, nind);
			pr_rv_list.push_back(RV[i].data());
			pr_vi_list.push_back(Vi[i].data());
		}
		#pragma omp parallel for
		for (int i = 0; i < nVi; i++)
		{
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, 1, pr_VW, nind, pr_vi_list[i], nind, 0, pr_rv_list[i], nind);
		}
		Ry = VW *Y;
	}
	Eigen::VectorXd u(nVi);
	u.setZero();
	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		Eigen::VectorXd Vi_Ry = Vi[i]*Ry;
		u[i] = Ry.cwiseProduct(Vi_Ry).sum();
	}
	Eigen::MatrixXd F(nVi, nVi);
	#pragma omp parallel for
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
		fix = inv_XtB_Bt * Y;
	}

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
			ss << "[Warning]: Thread ID: " << ThreadId
				<< "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead\n";
			printf("%s", ss.str().c_str());
			//			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
			LOG(WARNING)<< "Thread ID: " << ThreadId << "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead";
		}
		else
		{
			stringstream ss;
			ss <<  "[Error]: Thread ID: " << ThreadId
				<< "\tcalculating inverse matrix is failed, and pseudo inverse matrix is not allowed\n";
			throw std::exception(logic_error(ss.str().c_str()));
		}
		break;
	case 2:
	{
		stringstream ss;
		ss << "[Error]: Thread ID: " << ThreadId
			<< "\tcalculating inverse matrix is failed, and pseudo inverse matrix is also failed\n";
		throw std::exception(logic_error(ss.str().c_str()));
	}
	break;
	default:
		stringstream ss;
		ss << "[Error]: Thread ID: " << ThreadId
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



