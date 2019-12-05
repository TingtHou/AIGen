#include "../include/rln_mnq.h"


rln_mnq::rln_mnq(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomp = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomp = altDecompositionMode;
}

void rln_mnq::estimate()
{
	Eigen::MatrixXf Identity(nind, nind);
	Identity.setIdentity();
	vcs.resize(nVi);
	if (W.size()==0)
	{
		W = Eigen::VectorXf(nVi);
		W.setOnes();
	}
	float* pr_VW = VW.data();
	float* pr_Identity = Identity.data();
	for (int i = 0; i < nVi; i++)
	{
		float* pr_Vi = Vi[i].data();
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, W[i], pr_Vi, nind, pr_Identity, nind, 1, pr_VW, nind);
	}
	int status=Inverse(VW, Decomp, altDecomp, allowPseudoInverse);
	CheckInverseStatus(status);
	std::vector<Eigen::MatrixXf> RV(nVi);
	Eigen::VectorXf Ry(nind);
	Eigen::MatrixXf inv_XtB_Bt;
	if (ncov == 0)
	{
		for (int i = 0; i < nVi; i++)
		{
			RV[i] = Eigen::MatrixXf(nind, nind);
			float* pr_Vi = Vi[i].data();
			float* pr_rvi = RV[i].data();
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, 1, pr_VW, nind, pr_Vi, nind, 0, pr_rvi, nind);

		}
		float* pr_Y = Y.data();
		float* pr_RY = Ry.data();
		cblas_sgemv(CblasColMajor, CblasNoTrans, nind, nind, 1, pr_VW, nind, pr_Y, 1, 0, pr_RY, 1);
	//	Ry = VW * Y;
	}
	else
	{

		Eigen::MatrixXf B(nind, X.cols());
		Eigen::MatrixXf Xt_B(X.cols(), X.cols());
		Eigen::MatrixXf B_inv_XtB_Bt(nind, nind);
		inv_XtB_Bt.resize(X.cols(), nind);
		float* pr_X = X.data();
		float* pr_Y = Y.data();
		float* pr_RY = Ry.data();
		float* pr_B = B.data();
		float* pr_Xt_B = Xt_B.data();
		float* pr_inv_XtB_Bt = inv_XtB_Bt.data();
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, X.cols(), nind, 1, pr_VW, nind, pr_X, nind, 0, pr_B, nind);
		//XtB=Xt*B
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, X.cols(), X.cols(), nind, 1, pr_X, nind, pr_B, nind, 0, pr_Xt_B, X.cols());
		//inv(XtB)
		int status = Inverse(Xt_B, Decomp, altDecomp, allowPseudoInverse);
		CheckInverseStatus(status);
		//inv_XtB_Bt=inv(XtB)*Bt
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, X.cols(), nind, X.cols(), 1, pr_Xt_B, X.cols(), pr_B, nind, 0, pr_inv_XtB_Bt, X.cols());
		//inv_VM=inv_VM-B*inv(XtB)*Bt
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, X.cols(), -1, pr_B, nind, pr_inv_XtB_Bt, X.cols(), 1, pr_VW, nind);
		std::vector<float*> pr_rv_list;
		std::vector<float*> pr_vi_list;
		for (int i = 0; i < nVi; i++)
		{
			RV[i].resize(nind, nind);
			pr_rv_list.push_back(RV[i].data());
			pr_vi_list.push_back(Vi[i].data());
		}
		#pragma omp parallel for
		for (int i = 0; i < nVi; i++)
		{
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, 1, pr_VW, nind, pr_vi_list[i], nind, 0, pr_rv_list[i], nind);
		}
		Ry = VW *Y;
	}
	Eigen::VectorXf u(nVi);
	u.setZero();
	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		Eigen::VectorXd Ry_Vi_Ry = (Ry.cwiseProduct(Vi[i]*Ry)).cast<double>();
		u[i] = (float)Ry_Vi_Ry.sum();
	}
	Eigen::MatrixXf F(nVi, nVi);
	//Eigen::MatrixXf F_(nVi, nVi);
	#pragma omp parallel for
	for (int i = 0; i < nVi; i++)
	{
		for (int j=i;j<nVi;j++)
		{
			Eigen::MatrixXd RVij = (RV[i].transpose().cwiseProduct(RV[j])).cast<double>();
			F(i, j) = (float)RVij.sum();
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

