#include "../include/MNQTest.h"

MINQUETest::MINQUETest(std::shared_ptr<MinqueBase> MNQobj)
{
	Vi = MNQobj->getKernels();
	vcs = MNQobj->getvcs();
	F = MNQobj->getMatrixFInverse();
}

float MINQUETest::MINQUE_Normal(Eigen::VectorXf weights, std::vector<int> TestingIndex)
{
	Eigen::MatrixXf CovarianeM = 2*F(TestingIndex,TestingIndex);
	ToolKit::comput_inverse_logdet_SVD_mkl(CovarianeM);
	Eigen::MatrixXf CovarianeM_sqrt = CovarianeM;
	ToolKit::comput_msqrt_SVD_mkl(CovarianeM_sqrt);
	ToolKit::comput_inverse_logdet_LU_mkl(CovarianeM_sqrt);
	Eigen::VectorXf vcs_standard = CovarianeM_sqrt * vcs(TestingIndex);
	float overall_effect = vcs_standard.transpose() * vcs_standard;
	boost::math::chi_squared  chisqare(vcs_standard.size());
	boost::math::normal normaldist(0, 1);
	float pvalue_overall = vcs_standard.sum() > 0 ? (1 - boost::math::pdf(chisqare, overall_effect)) / 2 : 1;
	pvalue.resize(vcs_standard.size());
	for (size_t i = 0; i < vcs_standard.size(); i++)
	{
		pvalue[i] = 1 - boost::math::pdf(normaldist, vcs_standard(i));
	}

//	boost::math::chi_squared mydist(5)
//	
//		chiP < -ifelse(sum(est.vcs.standard[c(2, 3)]) > 0, pchisq(chi, 2, lower.tail = F) / 2, 1)
	return 1.0;
}



float MINQUETest::MINQUE0_overall(Eigen::VectorXf weights, std::vector<int> TestingIndex)
{
	int nKernels = Vi.size();
	int nind = Vi[0]->cols();
	const int strideX = 1;
	double ratio = 0;
	Eigen::MatrixXf SigmaTotal(nind, nind);
	SigmaTotal.setZero();
	std::shared_ptr<Eigen::MatrixXf> tmp = std::make_shared<Eigen::MatrixXf>(nind, nind);
	for (size_t i = 0; i < TestingIndex.size(); i++)
	{

		tmp->setZero();
		for (size_t j = 0; j < F.cols(); j++)
		{
			cblas_saxpby(nind*nind, F(TestingIndex[i], j), Vi[j]->data(),1,1, tmp->data(),1);
		}
		cblas_saxpby(nind * nind, 1/std::sqrt(2*F(TestingIndex[i], TestingIndex[i])), tmp->data(), 1, 1, SigmaTotal.data(), 1);
	//	std::cout << SigmaTotal.block(0, 0, 10, 10) << std::endl;
		ratio = ratio + vcs[TestingIndex[i]] * weights[TestingIndex[i]];
	}
	tmp.reset();
    Eigen::VectorXd eigenvalue = SigmaTotal.eigenvalues().real().cast<double>();
	SigmaTotal.resize(0,0);
	//std::cout << "The eigenvalues of the 3x3 matrix of ones are:" << std::endl << eigenvalue << std::endl;
	//////////////////////////////////////////////////////////////////////////////
	/// parameters for davies methods
	/// using deault setting as in R package "CompQuadForm"
	Eigen::VectorXd noncenter(eigenvalue.size()); 
	noncenter.setZero();
	Eigen::VectorXi df(eigenvalue.size());
	df.setOnes();
	int r = eigenvalue.size();
	double sigma = 0;
	int lim = 10000;
	double acc = 0.0001;
	Eigen::VectorXd trace(7);
	trace.setZero();
	int ifault = 0;
	double res = 0;

	double pvalue = 0;
	
	Davies* mixtureChi = new Davies();
	mixtureChi->qfc(eigenvalue.data(), noncenter.data(), df.data(), &r, &sigma, &ratio, &lim, &acc, trace.data(), &ifault, &res);
	pvalue = 1 - res;
	return (float)pvalue;
}


float MINQUETest::MINQUE0_singleparameter(Eigen::VectorXf est_null, int TestingIndex)
{
	int nKernels = Vi.size();
	int nind = Vi[0]->cols();
	const int strideX = 1;
	double ratio = 0;
	Eigen::MatrixXf Sigma_i(nind, nind);
	Sigma_i.setZero();
	Eigen::MatrixXf Sigma_H0(nind, nind);
	Sigma_H0.setZero();
	int temp_k = 0;
	for (size_t j = 0; j < F.cols(); j++)
	{
		cblas_saxpby(nind * nind, F(TestingIndex, j), Vi[j]->data(), 1, 1, Sigma_i.data(), 1);
		if (j != TestingIndex)
		{
			cblas_saxpby(nind * nind, est_null[temp_k++], Vi[j]->data(), 1, 1, Sigma_H0.data(), 1);
		}
	}

	ToolKit::comput_msqrt_SVD_mkl(Sigma_H0);
	Eigen::MatrixXf tmp(nind, nind);
	tmp.setZero();
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nind, nind, nind, 1, Sigma_H0.data(), nind, Sigma_i.data(), nind, 0, tmp.data(), nind);
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nind, nind, nind, 1, tmp.data(), nind, Sigma_H0.data(), nind, 0, Sigma_i.data(), nind);
	tmp.resize(0, 0);
	Sigma_H0.resize(0, 0);
	Eigen::VectorXd eigenvalue = Sigma_i.eigenvalues().real().cast<double>();
	Sigma_i.resize(0, 0);
	ratio = (double)vcs[TestingIndex];
	//std::cout << "The eigenvalues of the 3x3 matrix of ones are:" << std::endl << eigenvalue << std::endl;
	//////////////////////////////////////////////////////////////////////////////
	/// parameters for davies methods
	/// using deault setting as in R package "CompQuadForm"
	Eigen::VectorXd noncenter(eigenvalue.size());
	noncenter.setZero();
	Eigen::VectorXi df(eigenvalue.size());
	df.setOnes();
	int r = eigenvalue.size();
	double sigma = 0;
	int lim = 10000;
	double acc = 0.0001;
	Eigen::VectorXd trace(7);
	trace.setZero();
	int ifault = 0;
	double res = 0;

	double pvalue = 0;

	Davies* mixtureChi = new Davies();
	mixtureChi->qfc(eigenvalue.data(), noncenter.data(), df.data(), &r, &sigma, &ratio, & lim, & acc, trace.data(), & ifault, & res);
	pvalue = 1 - res;
	return (float)pvalue;
}
