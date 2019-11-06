#include "pch.h"
#include "rln_mnq.h"


rln_mnq::rln_mnq(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse)
{
	this->Decomp = DecompositionMode;
	this->allowPseudoInverse = allowPseudoInverse;
	this->altDecomp = altDecompositionMode;
}

void rln_mnq::estimate()
{
	vcs.resize(nVi);
	if (W.size()==0)
	{
		W = Eigen::VectorXd(nVi);
		W.setOnes();
		//#pragma omp parallel for
// 		for (int i=0;i<nVi;i++)
// 		{
// 			W[i] = 1 / (double)nVi;
// 		}
	}
	for (int i = 0; i < nVi; i++)
	{
		VW += W[i] * Vi[i];
	}
// 	Eigen::MatrixXd Inv_VW(nind,nind);
// 	Inv_VW.setZero();
	int status=Inverse(VW, Decomp, altDecomp, allowPseudoInverse);
	
	//ToolKit::Inv_SVD(VW, Inv_VW,true);
	switch (status)
	{
	case 0:
		break;
	case 1:
		if (allowPseudoInverse)
		{
			stringstream ss;
			ss<< YELLOW << "[Warning]: " << WHITE << "Thread ID: " << ThreadId 
				<< "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead\n";
			printf("%s",ss.str().c_str());
//			logfile->write("Calculating inverse matrix is failed, using pseudo inverse matrix instead", false);
			logger::record(logger::Level::Warning) << "Thread ID: "<<ThreadId<<"\tCalculating inverse matrix is failed, using pseudo inverse matrix instead";
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
	std::vector<Eigen::MatrixXd> RV(nVi);
	Eigen::VectorXd Ry;
	Eigen::MatrixXd C;
	if (ncov == 0)
	{
		for (int i = 0; i < nVi; i++)
		{
			Eigen::MatrixXd Inv_VW_Vi = VW * Vi[i];
			RV.push_back(Inv_VW_Vi);
		}
		Ry = VW * Y;
	}
	else
	{
//		clock_t t1 = clock();
		Eigen::MatrixXd B = VW * X;
		Eigen::MatrixXd Xt_B = X.transpose()*B;
//		Eigen::MatrixXd INV_Xt_B(Xt_B.rows(), Xt_B.cols());
		int status = Inverse(Xt_B, Decomp, altDecomp, allowPseudoInverse);
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
				logger::record(logger::Level::Warning) << "Thread ID: " << ThreadId << "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead";
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
	//	ToolKit::Inv_SVD(Xt_B, INV_Xt_B,true);
		C = Xt_B * B.transpose();
		Eigen::MatrixXd P = X * C;
//		std::cout<<"1st Stage: "<< (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	//	t1 = clock();
		int j = nVi;
//		
		double *p_mkl = P.data();
		double *inv_vw_mkl = VW.data();
		int m = P.rows();
		int k = P.cols();
		int n = m;
	//	#pragma omp parallel for private(j)
		for (int i=0;i<nVi;i++)
		{
			double *vi = (Vi[i]).data();
			double *c = (double *)mkl_calloc(n*n, sizeof(double), 64);
			double *rvi = (double *)mkl_calloc(n*n, sizeof(double), 64);
			mkl_domatcopy('c', 'n', n, n, 1, vi, n, c, n);
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				m, n, k, -1, p_mkl, m, vi, k, 1, c, m);
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				n, n, n,  1, inv_vw_mkl, n, c,	n,	0, rvi,	n);
			RV[i] = Eigen::Map<Eigen::MatrixXd>(rvi,n,n);
			mkl_free(c);
			mkl_free(rvi);
		}
//		std::cout << "2nd Stage: " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
		Ry = VW * (Y - P * Y);
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
//	Eigen::MatrixXd INV_F(nVi, nVi);
	status = Inverse(F, Decomp, altDecomp, allowPseudoInverse);
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
			logger::record(logger::Level::Warning) << "Thread ID: " << ThreadId << "\tCalculating inverse matrix is failed, using pseudo inverse matrix instead";
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

//	ToolKit::Inv_SVD(F, INV_F,true);
	
	vcs = F * u;
	if (ncov)
	{
		fix = C * Y;
	}
}

rln_mnq::~rln_mnq()
{
}



