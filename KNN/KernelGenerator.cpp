#include "pch.h"
#include "KernelGenerator.h"
#include <cmath>
#include <fstream>

KernelGenerator::KernelGenerator(GenoData & gd, int KernelName, double weight, double constant, double deg, double sigmma)
{
	int nrow = gd.Geno.rows();
	int ncol = gd.Geno.cols();
	Eigen::VectorXd weights(ncol);
	weights.setOnes();
	weights *= weight;
	kernels.kernelMatrix.resize(nrow, nrow);
	switch (KernelName)
	{
	case CAR:
		//getCAR(gd.Geno, weights, kernels.kernelMatrix);
		throw("Error: this kernel generator does not work now.");
		break;
	case Identity:
		getIdentity(gd.Geno, kernels.kernelMatrix);
		break;
	case Product:
		getProduct(gd.Geno, weights, kernels.kernelMatrix);
		break;
	case Ploymonial:
	//	getPolynomial(gd.Geno, weights, constant, deg, kernels.kernelMatrix);
		throw("Error: this kernel generator does not work now.");
		break;
	case Gaussian:
	//	getGaussian(gd.Geno, weights, sigmma, kernels.kernelMatrix);
		throw("Error: this kernel generator does not work now.");
		break;
	case IBS:
	//	getIBS(gd.Geno, weights, kernels.kernelMatrix);
		throw("Error: this kernel generator does not work now.");
		break;
	default:
		throw ("Invalided kernel name!");
		break;
	}
	for (auto it=gd.fid_iid.begin();it!=gd.fid_iid.end();it++)
	{
		kernels.fid_iid.insert({ it->first, it->second });
	}
	//kernels.fid_iid = gd.fid_iid;
	int totalsize = gd.Geno.rows()*gd.Geno.cols();
	kernels.VariantCountMatrix.resize(nrow, nrow);
	kernels.VariantCountMatrix.setOnes();
	kernels.VariantCountMatrix *= totalsize;
}

KernelGenerator::KernelGenerator()
{

}

void KernelGenerator::BuildBin(std::string prefix)
{
	KernelWriter kwriter(kernels);
	kwriter.write(prefix);
}

KernelGenerator::~KernelGenerator()
{
}

void KernelGenerator::test()
{
	PlinkReader pk;
	pk.read("../m20.ped", "../m20.map");
	GenoData gd = pk.GetGeno();
	Eigen::MatrixXd geno = gd.Geno;
	ofstream genot("../genoC.txt", ios::out);
	for (int i=0;i<geno.rows();i++)
	{
		for (int j=0;j<geno.cols();j++)
		{
			genot << geno(i, j) << "\t";
		}
		genot << std::endl;
	}
	genot.close();
	stripSameCol(geno);
	int nrow = geno.rows();
	int ncol = geno.cols();
// 	stdSNPmv(geno);
// 	ofstream ofo("../stdC.txt", ios::out);
// 	for (int i = 0; i < nrow; i++)
// 	{
// 		for (int j = 0; j < ncol; j++)
// 		{
// 			ofo << std::fixed << std::setprecision(4) << geno(i, j) << "\t";
// 		}
// 		ofo << std::endl;
// 	}
// 
// 	ofo.close();

	Eigen::MatrixXd kernel(nrow, nrow);
	Eigen::VectorXd weight(ncol);
	weight.setOnes();
	getProduct(geno, weight, kernel);
	ofstream of("../proC.txt", ios::out);
	for (int i=0;i<nrow;i++)
	{
		for (int j=0;j<nrow;j++)
		{
			of <<std::fixed<<std::setprecision(4)<< kernel(i,j) << "\t";
		}
		of << std::endl;
	}
	
	of.close();
	kernels.kernelMatrix = kernel;
//	kernels.fid_iid = gd.fid_iid;
	for (auto it = gd.fid_iid.begin(); it != gd.fid_iid.end(); it++)
	{
		kernels.fid_iid.insert({ it->first, it->second });
	}
	int totalsize = gd.Geno.rows()*gd.Geno.cols();
	kernels.VariantCountMatrix.resize(nrow, nrow);
	kernels.VariantCountMatrix.setOnes();
	kernels.VariantCountMatrix *= totalsize;
	BuildBin("../KC");
//	std::cout << kernel << std::endl;
}

void KernelGenerator::getCAR(Eigen::MatrixXd & Geno, Eigen::VectorXd &weights, Eigen::MatrixXd & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	Eigen::MatrixXd IBS(nrow, nrow);
	getIBS(Geno, weights, IBS);
	Eigen::MatrixXd S = IBS / (2 * weights.sum());
	S.diagonal().setZero();
	Eigen::MatrixXd D(nrow, nrow);
	D.setZero();
	for (int i=0;i<nrow;i++)
	{
		D(i, i) = S.row(i).sum();
	}
	//////////////////
	//this is Covariance 
	//Eigen::MatrixXd centered = Geno.rowwise() - Geno.colwise().mean();
	//Eigen::MatrixXd cov = (centered.adjoint() * centered) ;
	///////////////////////
	Eigen::MatrixXd centered = Geno.rowwise() - Geno.colwise().mean();
	Eigen::MatrixXd cor = (centered.transpose() * centered)/ double(nrow - 1);
	for (int i=0;i<ncol;i++)
	{
		for (int j=0;j<=i;j++)
		{
			Eigen::VectorXd Coli(nrow);
			Eigen::VectorXd Colj(nrow);
			Coli << Geno.col(i);
			Colj << Geno.col(j);
			cor(i, j) = cor(j, i) = cor(j, i)/(std::sqrt(Variance(Coli)*Variance(Colj)));
		}
	}
	double gamma = cor.mean();
	Eigen::MatrixXd Va = D - gamma * S;
	std::cout << Va << std::endl;
	Inverse(Va, kernel, 0, 3, true);
}

void KernelGenerator::getIdentity(Eigen::MatrixXd & Geno, Eigen::MatrixXd & kernel)
{
	int nrow = Geno.rows();
	kernel.resize(nrow, nrow);
	kernel.setIdentity();
}

void KernelGenerator::getProduct(Eigen::MatrixXd & Geno, Eigen::VectorXd &weights, Eigen::MatrixXd & kernel)
{
	int ncol = Geno.cols();
	int nrow = Geno.rows();
	kernel.resize(nrow, nrow);
// 	Eigen::MatrixXd sqrtw=weights.asDiagonal();
// 	sqrtw.cwiseSqrt();
// 	Eigen::MatrixXd Geno_sqetw = Geno * sqrtw;
// 	kernel = (1 / weights.sum())*(Geno_sqetw*Geno_sqetw.transpose());
	kernel = (1 / double(ncol))*(Geno*Geno.transpose());
}

void KernelGenerator::getPolynomial(Eigen::MatrixXd & Geno, Eigen::VectorXd & weights, double constant, double deg, Eigen::MatrixXd & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	kernel.resize(nrow, nrow);
	Eigen::MatrixXd sqrtw = weights.asDiagonal();
	sqrtw.cwiseSqrt();
	Eigen::MatrixXd Geno_sqetw = Geno * sqrtw;
	kernel = (constant*Eigen::MatrixXd::Ones(nrow,nrow) + (1 / weights.sum())*(Geno_sqetw*Geno_sqetw.transpose())).array().pow(deg);
}



void KernelGenerator::getGaussian(Eigen::MatrixXd & Geno, Eigen::VectorXd & weights, double sigmma, Eigen::MatrixXd & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	kernel.resize(nrow, nrow);
	Eigen::MatrixXd wtGeno = (Geno * weights.cwiseSqrt().asDiagonal());
	Eigen::MatrixXd DistMat(nrow, nrow);
	//compute the euclidean distances between the rows of a data matrix.
	for (int i=0;i<nrow;i++)
	{
		for (int j=0;j<=i;j++)
		{
			Eigen::VectorXd rowi_rowj(ncol);
			rowi_rowj = wtGeno.row(i) - wtGeno.row(j);
			rowi_rowj=rowi_rowj.array().pow(2);
			DistMat(j, i) = DistMat(i, j) = std::sqrt(rowi_rowj.sum());
		}
	}
	kernel = -(1 / (2 * sigmma*weights.sum()))*(DistMat.array().pow(2));
	kernel=kernel.array().exp();

}


void KernelGenerator::getIBS(Eigen::MatrixXd & Geno, Eigen::VectorXd & weights, Eigen::MatrixXd & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	Eigen::MatrixXd gtemp1 = Geno, gtemp2 = Geno;
	for (int i=0;i<nrow;i++)
	{
		for (int j=0;j<ncol;j++)
		{
			if (Geno(i,j)==2)
			{
				gtemp2(i, j) = gtemp1(i, j) = 1;
			}
			if (Geno(i,j)==1)
			{
				gtemp2(i, j) = 0;
			}
		}
	}
	Eigen::MatrixXd gtemp(nrow, 2 * ncol);
	gtemp << gtemp1, gtemp2;
	Eigen::VectorXd Weight(2 * ncol);
	Weight<<weights,
			weights;
	Eigen::MatrixXd Inner = gtemp * Weight.asDiagonal()*gtemp.transpose();
	Eigen::VectorXd InnerDiad = Inner.diagonal();
	Eigen::MatrixXd X2(nrow, nrow), Y2(nrow, nrow);
	for (int i=0;i<nrow;i++)
	{
		X2.col(i) = InnerDiad;
		Y2.row(i) = InnerDiad;
	}
	Eigen::MatrixXd VWeights(1, ncol);
	VWeights.setOnes();
	VWeights *= weights*2;
	double Bound = VWeights.sum();
	Eigen::MatrixXd Dis = X2 + Y2 - 2 * Inner;
	kernel = (-Dis).array() + Bound;
}

