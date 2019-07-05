#include "pch.h"
#include "KernelCompute.h"


KernelCompute::KernelCompute()
{

}

KernelCompute::~KernelCompute()
{
}

void KernelCompute::test()
{
	PlinkReader pk("../m20.bed", "../m20.bim", "../m20.fam");
	Eigen::MatrixXd geno = pk.GetGeno();
	Eigen::VectorXd a(10);
	a.setOnes();
	Eigen::MatrixXd testgeno(10,geno.cols()+1);
	testgeno << a, geno;
	std::cout << testgeno << std::endl;
	stripSameCol(testgeno);
	std::cout << testgeno << std::endl;
	if (testgeno==geno)
	{
		std::cout << "Strip columns successed" << std::endl;

	}

}

void KernelCompute::getGaussian(Eigen::MatrixXd & Geno, double weights, double sigmma)
{
}

void KernelCompute::getIBS(Eigen::MatrixXd &Geno, double weights)
{
	int nrow = Geno.rows();
	int ncol = Geno.cols();
	Eigen::MatrixXd gtemp1 = Geno;
	Eigen::MatrixXd gtemp2 = Geno;


}

void KernelCompute::stripSameCol(Eigen::MatrixXd & Geno)
{
	std::vector<int> repeatID;
	repeatID.clear();
	for (int i=0;i<Geno.cols();i++)
	{
		double *test = Geno.col(i).data();
		std::vector<double> rowi(test, test+Geno.col(i).size());
		std::sort(rowi.begin(), rowi.end());
		auto it = std::unique(rowi.begin(), rowi.end());
		rowi.erase(it, rowi.end());
		int len = rowi.size();
		if (len==1)
		{
			repeatID.push_back(i);
		}
	}
	if (repeatID.empty())
	{
		return;
	}
	Eigen::MatrixXd tmpGeno = Geno;
	Geno.resize(tmpGeno.rows(), tmpGeno.cols() - repeatID.size());
	int j = 0;
	for (int i=0;i<tmpGeno.cols();i++)
	{
		if (std::find(repeatID.begin(), repeatID.end(), i) != repeatID.end())
		{
			continue;
		}
		Geno.col(j++) = tmpGeno.col(i);
	}

}

void KernelCompute::stdSNPmv(Eigen::MatrixXd & Geno)
{
	int ncol = Geno.cols();
	int nrow = Geno.rows();
	Eigen::MatrixXd tmpGeno = Geno;
	Geno.setZero();
	for (int i=0;i<ncol;i++)
	{
		Eigen::VectorXd Coli(nrow);
		Coli << tmpGeno.col(i);
		double means = mean(Coli);
		double sd = std::sqrt(Variance(Coli));
		Coli -= means * Eigen::VectorXd::Ones(nrow);
		Coli /= sd;
		Geno.col(i) << Coli;
	}
}


