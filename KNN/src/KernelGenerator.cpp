#include "../include/KernelGenerator.h"


KernelGenerator::KernelGenerator(GenoData & gd, int KernelName, Eigen::VectorXf &weights,  float scale, float constant, float deg, float sigmma)
{
	int nrow = gd.Geno.rows();
	int ncol = gd.Geno.cols();
	kernels.kernelMatrix.resize(nrow, nrow);
	this->scale = scale;
	switch (KernelName)
	{
	case CAR:
		getCAR(gd.Geno, weights, kernels.kernelMatrix);
	//	throw std::exception("Error: this kernel generator does not work now.");
		break;
	case Identity:
		getIdentity(gd.Geno, kernels.kernelMatrix);
		break;
	case Product:
		getProduct(gd.Geno, weights, kernels.kernelMatrix);
		break;
	case Polymonial:
		getPolynomial(gd.Geno, weights, constant, deg, kernels.kernelMatrix);
	//	throw std::exception("Error: this kernel generator does not work now.");
		break;
	case Gaussian:
		getGaussian(gd.Geno, weights, sigmma, kernels.kernelMatrix);
	//	throw std::exception("Error: this kernel generator does not work now.");
		break;
	case IBS:
		getIBS(gd.Geno, weights, kernels.kernelMatrix);
	//	throw std::exception("Error: this kernel generator does not work now.");
		break;
	default:
		throw std::string("Invalided kernel name!");
		break;
	}
	kernels.fid_iid = gd.fid_iid;
//	for (auto it=gd.fid_iid.begin();it!=gd.fid_iid.end();it++)
//	{
//		kernels.fid_iid.insert({ it->first, it->second });
//	}
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
	Eigen::MatrixXf geno = gd.Geno;
	std::ofstream genot("../genoC.txt", std::ios::out);
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

	Eigen::MatrixXf kernel(nrow, nrow);
	Eigen::VectorXf weight(ncol);
	weight.setOnes();
	getProduct(geno, weight, kernel);
	std::ofstream of("../proC.txt", std::ios::out);
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
	kernels.fid_iid = gd.fid_iid;
//	for (auto it = gd.fid_iid.begin(); it != gd.fid_iid.end(); it++)
//	{
//		kernels.fid_iid.insert({ it->first, it->second });
//	}
	int totalsize = gd.Geno.rows()*gd.Geno.cols();
	kernels.VariantCountMatrix.resize(nrow, nrow);
	kernels.VariantCountMatrix.setOnes();
	kernels.VariantCountMatrix *= totalsize;
	BuildBin("../KC");
//	std::cout << kernel << std::endl;
}

void KernelGenerator::getCAR(Eigen::MatrixXf & Geno, Eigen::VectorXf &weights, Eigen::MatrixXf & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	Eigen::MatrixXf IBS(nrow, nrow);
	getIBS(Geno, weights, IBS);
	Eigen::MatrixXf S = IBS / (2 * weights.sum());
	S.diagonal().setZero();
	Eigen::MatrixXf D(nrow, nrow);
	D.setZero();
    #pragma omp parallel for
	for (int i=0;i<nrow;i++)
	{
		D(i, i) = S.row(i).sum();
	}
	//////////////////
	//this is Covariance 
	//Eigen::MatrixXf centered = Geno.rowwise() - Geno.colwise().mean();
	//Eigen::MatrixXf cov = (centered.adjoint() * centered) ;
	///////////////////////
	Eigen::MatrixXf centered = Geno.rowwise() - Geno.colwise().mean();
	Eigen::MatrixXf cor = (centered.transpose() * centered)/ float(nrow - 1);
	#pragma omp parallel for
	for (int i=0;i<ncol;i++)
	{
		for (int j=0;j<=i;j++)
		{
			Eigen::VectorXf Coli(nrow);
			Eigen::VectorXf Colj(nrow);
			Coli << Geno.col(i);
			Colj << Geno.col(j);
			cor(i, j) = cor(j, i) = cor(j, i)/(std::sqrt(Variance(Coli)*Variance(Colj)));
		}
	}
	float gamma = cor.mean();
	Eigen::MatrixXf Va = D - gamma * S;
//	std::cout << Va << std::endl;
	Inverse(Va, 0, 3, true);
	kernel = Va;
}

void KernelGenerator::getIdentity(Eigen::MatrixXf & Geno, Eigen::MatrixXf & kernel)
{
	int nrow = Geno.rows();
	kernel.resize(nrow, nrow);
	kernel.setIdentity();
}

void KernelGenerator::getProduct(Eigen::MatrixXf & Geno, Eigen::VectorXf &weights, Eigen::MatrixXf & kernel)
{
	int ncol = Geno.cols();
	int nrow = Geno.rows();
	kernel.resize(nrow, nrow);
 	Eigen::MatrixXf sqrtw=weights.asDiagonal();
	sqrtw = sqrtw.cwiseSqrt();
 	Eigen::MatrixXf Geno_sqetw = Geno * sqrtw;
	kernel = (1 / float(ncol))*(Geno_sqetw*Geno_sqetw.transpose());
	if (scale)
	{
		kernel = (1 / weights.sum())*kernel;
	}
// 	kernel = (1 / weights.sum())*(Geno_sqetw*Geno_sqetw.transpose());
	
}

void KernelGenerator::getPolynomial(Eigen::MatrixXf & Geno, Eigen::VectorXf & weights, float constant, float deg, Eigen::MatrixXf & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	kernel.resize(nrow, nrow);
	Eigen::MatrixXf sqrtw = weights.asDiagonal();
	sqrtw = sqrtw.cwiseSqrt();
	Eigen::MatrixXf Geno_sqetw = Geno * sqrtw;
	kernel = (1 / float(ncol))*(Geno_sqetw*Geno_sqetw.transpose());
	if (scale)
	{
		kernel = (1 / weights.sum())*kernel;
	}
	kernel = (constant*Eigen::MatrixXf::Ones(nrow,nrow) + kernel).array().pow(deg);
}



void KernelGenerator::getGaussian(Eigen::MatrixXf & Geno, Eigen::VectorXf & weights, float sigmma, Eigen::MatrixXf & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	kernel.resize(nrow, nrow);
	Eigen::MatrixXf wtGeno = (Geno * weights.cwiseSqrt().asDiagonal());
	Eigen::MatrixXf DistMat(nrow, nrow);
	//compute the euclidean distances between the rows of a data matrix.
	for (int i=0;i<nrow;i++)
	{
		for (int j=0;j<=i;j++)
		{
			Eigen::VectorXf rowi_rowj(ncol);
			rowi_rowj = wtGeno.row(i) - wtGeno.row(j);
			rowi_rowj=rowi_rowj.array().pow(2);
			DistMat(j, i) = DistMat(i, j) = std::sqrt(rowi_rowj.sum());
		}
	}
	kernel = -(1 / (2 * float(ncol) * sigmma))*(DistMat.array().pow(2));
	if (scale)
	{
		kernel= (1 / weights.sum())*kernel;
	}
	kernel=kernel.array().exp();

}


void KernelGenerator::getIBS(Eigen::MatrixXf & Geno, Eigen::VectorXf & weights, Eigen::MatrixXf & kernel)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	Eigen::MatrixXf gtemp1 = Geno, gtemp2 = Geno;
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
	Eigen::MatrixXf gtemp(nrow, 2 * ncol);
	gtemp << gtemp1, gtemp2;
	Eigen::VectorXf Weight(2 * ncol);
	Weight<<weights,
			weights;
	Eigen::MatrixXf Inner = gtemp * Weight.asDiagonal()*gtemp.transpose();
	Eigen::VectorXf InnerDiad = Inner.diagonal();
	Eigen::MatrixXf X2(nrow, nrow), Y2(nrow, nrow);
	for (int i=0;i<nrow;i++)
	{
		X2.col(i) = InnerDiad;
		Y2.row(i) = InnerDiad;
	}
	Eigen::MatrixXf VWeights(1, ncol);
	VWeights.setOnes();
	VWeights *= weights*2;
	float Bound = VWeights.sum();
	Eigen::MatrixXf Dis = X2 + Y2 - 2 * Inner;
	kernel = (-Dis).array() + Bound;
}

