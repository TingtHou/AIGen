#include "../include/KernelExpansion.h"


KernelExpansion::KernelExpansion(std::vector<Eigen::MatrixXd> &MatrixHList, int dimension)
{
	this->OriKernelList = MatrixHList;
	this->dimension = dimension;
	KernelCount = MatrixHList.size();
	ncol = OriKernelList[0].cols();
	nrow = OriKernelList[0].rows();
	Expanse(dimension, ExtendedMatrix);
}

KernelExpansion::KernelExpansion(std::vector<KernelData> &kernels, int dimension)
{
	for (int i=0;i<kernels.size();i++)
	{
		OriKernelList.push_back(kernels[i].kernelMatrix);
	}
	KernelCount = OriKernelList.size();
	this->dimension = dimension;
	ncol = OriKernelList[0].cols();
	nrow = OriKernelList[0].rows();
	Expanse(dimension, ExtendedMatrix);
}

KernelExpansion::~KernelExpansion()
{
}

void KernelExpansion::test()
{
	double a[] = { 1,2,3,4,5,6,7,8,11,12 };
	double b[] = { 5,6,7,8,9,0,1,2, 9,10 };
	Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(a, 3, 3);
	Eigen::MatrixXd B= Eigen::Map<Eigen::MatrixXd>(b, 3, 3);
	std::vector<Eigen::MatrixXd> vlist;
	vlist.push_back(A);
	vlist.push_back(B);
	KernelExpansion elist(vlist, 3);
	std::vector<Eigen::MatrixXd> ex = elist.GetExtendMatrix();
	std::cout << "Matrix a:\n" <<A << std::endl;
	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	std::cout << "Matrix b:\n" << B << std::endl;
	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	std::cout << "Extended Matrixes:\n" << std::endl;
	for (int i=0;i<ex.size();i++)
	{
		std::cout << "Matrix "<<i<<":\n" << ex[i] << std::endl;
		std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	}
}

void KernelExpansion::GetFullIndex(int degree, int M, int *Index, std::vector<std::vector<int>> &Comb)
{
	if (degree>0)
	{
		for (int i=0;i<KernelCount;i++)
		{
			Index[degree-1] = i;
			GetFullIndex(degree - 1, M, Index, Comb);
		}
	}
	else
	{
		Comb.push_back(std::vector<int>(Index, Index + M));
		return;
	}
}
void KernelExpansion::GetUniqueIndex(std::vector<std::vector<int>>& Comb, std::vector<std::vector<int>>& UniqueComb)
{
	for (int i = 0; i < Comb.size(); i++)
	{
		std::vector<int> CombEach = Comb[i];
		int isInUnique = 0;
		if (std::find(UniqueComb.begin(), UniqueComb.end(), CombEach) != UniqueComb.end())
		{
			isInUnique = 1;
		}
		else
		{
			std::vector<int> sortedCombEach = CombEach;
			std::sort(sortedCombEach.begin(), sortedCombEach.end());
			if (std::find(UniqueComb.begin(), UniqueComb.end(), sortedCombEach) != UniqueComb.end())
			{
				isInUnique = 1;
				break;
			}
			while (next_permutation(sortedCombEach.begin(), sortedCombEach.end()))
			{
				if (std::find(UniqueComb.begin(), UniqueComb.end(), sortedCombEach) != UniqueComb.end())
				{
					isInUnique = 1;
					break;
				}
			}
		}
		if (!isInUnique)
		{
			UniqueComb.push_back(CombEach);
		}
	}
}
void KernelExpansion::Expanse(int degree,  std::vector<Eigen::MatrixXd> &EMatrix)
{
	std::vector<std::vector<int>> ExpansedIndex;
	for (int i=1;i<=degree;i++)
	{
		int * Index = new int[degree];
		std::vector<std::vector<int>> FullComb;
		std::vector<std::vector<int>> UniqueComb;
		GetFullIndex(i, i, Index, FullComb);
		GetUniqueIndex(FullComb, UniqueComb);
		ExpansedIndex.insert(ExpansedIndex.end(),UniqueComb.begin(),UniqueComb.end());
		delete Index;
	}
	for (int i = 0; i < ExpansedIndex.size(); i++)
	{
		Eigen::MatrixXd exMatrix(nrow,ncol);
		exMatrix.setOnes();
		for (int j=0;j<ExpansedIndex[i].size();j++)
		{
			exMatrix = exMatrix.cwiseProduct(OriKernelList[ExpansedIndex[i][j]]);
		}
		EMatrix.push_back(exMatrix);
	}
}
