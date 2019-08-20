#include "pch.h"
#include "DataManager.h"
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include "KernelManage.h"
#include "PlinkReader.h"
#include "CommonFunc.h"
DataManager::DataManager()
{
}

void DataManager::match()
{
	for (int i=0;i<KernelList.size();i++)
	{
		match(phe, KernelList[i]);
	}
}

DataManager::~DataManager()
{
}

// void DataManager::read()
// {
// 	if (programOptions.count("phe"))
// 	{
// 		std::string reponsefile = programOptions["phe"].as < std::string >();
// 
// 	}
// 	if (programOptions.count("kernel"))
// 	{
// 		std::string kernelfiles = programOptions["kernel"].as<std::string >();
// 		KernelReader kreader(kernelfiles);
// 		kreader.read();
// 		KernelData tmp = kreader.getKernel();
// 		if (phe.fid_iid.size() > 0)
// 		{
// 			match(phe, tmp, kernelfiles);
// 		}
// 
// 		KernelList.push_back(tmp);
// 	}
// 	if (programOptions.count("mkernel"))
// 	{
// 		std::string mkernelfile = programOptions["mkernel"].as<std::string >();
// 		readmkernel(mkernelfile);
// 	}
// }

void DataManager::readPhe(std::string phefilename)
{
	readResponse(phefilename, phe);
}

void DataManager::readKernel(std::string prefix)
{
	clock_t t1 = clock();
	KernelReader kreader(prefix);
	kreader.read();
	KernelData tmp = kreader.getKernel();
	KernelList.push_back(tmp);

}

void DataManager::readmKernel(std::string mkernelfilename)
{
	readmkernel(mkernelfilename);
}

void DataManager::readGeno(std::vector<std::string> filelist, bool isImpute)
{
	PlinkReader preader;
	preader.setImpute(isImpute);
	switch (filelist.size())
	{
	case 2:
		preader.read(filelist[0], filelist[1]); //read ped and map
		break;
	case 3:
		preader.read(filelist[0], filelist[1], filelist[2]); //read bed, bim, map
		break;
	default:
		std::stringstream ss;
		ss << "Error: Cannot open genotype files, ";
		for (int i=0; i<filelist.size();i++)
		{
			ss << "[" << filelist[i] << "], ";
		}
		ss << "to read." << std::endl;
		throw (ss.str());
		break;
	}
	geno = preader.GetGeno();
}

void DataManager::readResponse(std::string resopnsefile, PhenoData & phe)
{
	std::ifstream infile;
	
	std::vector<double> yvector;
	yvector.clear();
	//	std::cout << "Reading Phenotype from " << resopnsefile << std::endl;
	infile.open(resopnsefile);
	if (!infile.is_open())
	{
		throw  ("Error: Cannot open [" + resopnsefile + "] to read.");
	}
	int id = 0;
	while (!infile.eof())
	{
		std::string str;
		getline(infile, str);
		if (infile.fail())
		{
			continue;
		}
		boost::algorithm::trim(str);
		if (str.empty())
		{
			continue;
		}
		std::vector<std::string> strVec;
		boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"));
		phe.fid_iid.insert({ id++,strVec[0] + "_" + strVec[1] });
		yvector.push_back(stod(strVec[2]));
	}
	infile.close();
	int nind = yvector.size();
	std::cout << nind << " Total" << std::endl;
	phe.Phenotype = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(yvector.data(), yvector.size());
	//	std::cout << "Reading Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
}

void DataManager::readmkernel(std::string mkernel)
{
	std::ifstream klistifstream;
	klistifstream.open(mkernel, std::ios::in);
	if (!klistifstream.is_open())
	{
		throw ("Error: cannot open the file [" + mkernel + "] to read.");
	}
	while (!klistifstream.eof())
	{
		std::string prefix;
		getline(klistifstream, prefix);
		if (klistifstream.fail())
		{
			continue;
		}
		KernelReader kreader(prefix);
		kreader.read();
		KernelData tmpData = kreader.getKernel();
		KernelList.push_back(tmpData);
	}
	klistifstream.close();
}




void DataManager::match(PhenoData &phenotype, KernelData &kernel)
{
// 	if (phenotype.fid_iid.size() != kernel.fid_iid.size())
// 	{
// 		throw ("Error: the number of individuals in phenotype file cannot match the kernel file.\n");
// 	}
	if (phenotype.fid_iid == kernel.fid_iid)
	{
		return;
	}
	PhenoData tmpPhe = phenotype;
	KernelData tmpKernel = kernel;
	std::vector<std::string> overlapID;
	set_difference(phenotype.fid_iid, kernel.fid_iid, overlapID);
	int nind = overlapID.size(); //overlap FID_IID
	kernel.kernelMatrix.resize(nind, nind);
	kernel.VariantCountMatrix.resize(nind, nind);
	kernel.fid_iid.clear();
	phenotype.Phenotype.resize(nind);
	phenotype.fid_iid.clear();
	for (int i = 0; i < nind; i++)
	{
		std::string rowID = overlapID[i];
		auto itrow = tmpKernel.fid_iid.right.find(rowID);
		int OriKernelRowID = itrow->second;
		for (int j = 0; j <= i; j++)
		{
			
			std::string colID = overlapID[j];
			auto itcol = tmpKernel.fid_iid.right.find(colID);
			int OriKernelColID = itcol->second;
// 			if (!rKernelID.count(rowID))
// 			{
// 				throw ("Error: cannot find the individual [" + rowID + "] in the kernel file.\n");
// 			}
			kernel.kernelMatrix(i, j) = kernel.kernelMatrix(j, i) = tmpKernel.kernelMatrix(OriKernelRowID, OriKernelColID);
			kernel.VariantCountMatrix(i, j) = kernel.VariantCountMatrix(j, i) = tmpKernel.VariantCountMatrix(OriKernelRowID, OriKernelColID);
		}
		auto it = tmpPhe.fid_iid.right.find(rowID);
		int OriPheID = it->second;
		phenotype.Phenotype[i] = tmpPhe.Phenotype[OriPheID];
		phenotype.fid_iid.insert({ i, rowID });
		kernel.fid_iid.insert({ i, rowID });
	}
}
