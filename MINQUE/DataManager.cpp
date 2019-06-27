#include "pch.h"
#include "DataManager.h"
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include "KernelManage.h"

DataManager::DataManager(boost::program_options::variables_map programOptions, LOG * log)
{
	this->programOptions = programOptions;
	this->log = log;
}

DataManager::~DataManager()
{
}

void DataManager::read()
{
	if (programOptions.count("phe"))
	{
		std::string reponsefile = programOptions["phe"].as < std::string >();
		readResponse(reponsefile, phe);
	}
	if (programOptions.count("kernel"))
	{
		std::string kernelfiles = programOptions["kernel"].as<std::string >();
		KernelReader kreader(kernelfiles);
		kreader.read();
		KernelData tmp = kreader.getKernel();
		match(phe, tmp, kernelfiles);
		KernelList.push_back(tmp);
	}
	if (programOptions.count("mkernel"))
	{
		std::string mkernelfile = programOptions["mkernel"].as<std::string >();
		readmkernel(mkernelfile);
	}
}

void DataManager::readResponse(std::string resopnsefile, PhenoData & phe)
{
	std::ifstream infile;
	
	std::vector<double> yvector;
	yvector.clear();
	clock_t t1 = clock();
	//	std::cout << "Reading Phenotype from " << resopnsefile << std::endl;
	std::stringstream ss;
	ss << "Reading Phenotype from " << resopnsefile;
	log->write(ss.str(), true);
	infile.open(resopnsefile);
	if (!infile.is_open())
	{
		std::stringstream ssin;
		ssin << "Reading " + resopnsefile + " is failed.";
		log->write(ssin.str(), true);
		exit(0);
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
		std::pair<int, std::string> fid_iid(id++, strVec[0] + "_" + strVec[1]);
		phe.fid_iid.insert(fid_iid);
		yvector.push_back(stod(strVec[2]));
	}
	infile.close();
	int nind = yvector.size();
	std::cout << nind << " Total" << std::endl;
	phe.Phenotype = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(yvector.data(), yvector.size());
	//	std::cout << "Reading Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	ss.str("");
	ss << "Read Phenotype Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";
	log->write(ss.str(), true);
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
		match(phe, tmpData, prefix);
		KernelList.push_back(tmpData);
	}
	klistifstream.close();
}

void DataManager::match(PhenoData &phenotype, KernelData &kernel, std::string prefix)
{
	if (phenotype.fid_iid.size()!=kernel.fid_iid.size())
	{
		throw ("Error: the number of individuals in phenotype file cannot match the kernel file [" + prefix + "].\n");
	}
	if (phenotype.fid_iid==kernel.fid_iid)
	{
		return;
	}
	Eigen::MatrixXd tmpKMatrix = kernel.kernelMatrix;
	Eigen::MatrixXd tmpCMatrix = kernel.VariantCountMatrix;
	std::map<std::string,int> rKernelID = kernel.rfid_iid;
	kernel.kernelMatrix.setZero();
	kernel.VariantCountMatrix.setZero();
	kernel.fid_iid.clear();
	kernel.rfid_iid.clear();
	int nind = phenotype.fid_iid.size();
	for (int i=0;i<nind;i++)
	{
		for (int j=0;j<=i;j++)
		{
			std::string rowID = phenotype.fid_iid[i];
			std::string colID = phenotype.fid_iid[j];
			if (!rKernelID.count(rowID))
			{
				throw ("Error: cannot find the individual [" + rowID + "] in the kernel file [" + prefix + "].\n");
			}
			if (!rKernelID.count(colID))
			{
				throw ("Error: cannot find the individual [" + colID + "] in the kernel file [" + prefix + "].\n");
			}
			kernel.kernelMatrix(i, j) = kernel.kernelMatrix(j, i) = tmpKMatrix(rKernelID[rowID], rKernelID[colID]);
			kernel.VariantCountMatrix(i, j) = kernel.VariantCountMatrix(j, i) = tmpCMatrix(rKernelID[rowID], rKernelID[colID]);
		}
		kernel.rfid_iid[phenotype.fid_iid[i]] = i;
	}
	kernel.fid_iid = phenotype.fid_iid;

}
