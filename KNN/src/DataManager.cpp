#include "../include/DataManager.h"

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


void DataManager::readCovariates(std::string covfilename)
{
	std::vector<std::vector<float>> covs;
	std::ifstream infile;
	infile.open(covfilename);
	if (!infile.is_open())
	{
		throw  ("Error: Cannot open [" + covfilename + "] to read.");
	}
	while (!infile.eof())
	{
		std::string str;
		getline(infile, str);
		if (!str.empty() && str.back() == 0x0D)
			str.pop_back();
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
		boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"), boost::token_compress_on);
		std::vector<float> floatVector;
		floatVector.reserve(strVec.size());
		transform(strVec.begin(), strVec.end(), back_inserter(floatVector),
			[](string const& val) {return stof(val); });
		covs.push_back(floatVector);
	}
	infile.close();
	Covariates.resize(covs.size(), 1+covs[0].size());
	#pragma omp parallel for
	for (int i = 0; i < Covariates.rows(); i++)
	{
		Covariates(i, 0) = 1;
		for (int j = 1; j < Covariates.cols(); j++)
		{
			Covariates(i, j) = covs[i][j];
		}
	}
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
		ss  <<"[Error]: Cannot open genotype files, ";
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

void DataManager::readWeight(std::string filename)
{

	std::ifstream infile;
	std::vector<float> WVector;
	WVector.clear();
	infile.open(filename);
	if (!infile.is_open())
	{
		throw  ("Error: Cannot open [" + filename + "] to read.");
	}
	std::string str;
	getline(infile, str);
	if (!str.empty() && str.back() == 0x0D)
		str.pop_back();
	boost::algorithm::trim(str);
	std::vector<std::string> strVec;
	boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"), boost::token_compress_on);
	for (size_t i = 0; i < strVec.size(); i++)
	{
		if (!isNum(strVec[i]))
		{
			std::stringstream ss;
			ss << "Error: The " << i << "th element in " << filename << " is not a number.\nPlease check it again.";
			LOG(ERROR) << ss.str();
			throw(ss.str());
		}
		WVector.push_back(stof(strVec[i]));
	}
	infile.close();
	std::stringstream ss;
	ss << "Reading " << strVec.size() << " weights for Iterative MINQUE" << std::endl;
	std::cout << ss.str();
	LOG(INFO) << ss.str();
	Weights = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(WVector.data(), WVector.size());
}

void DataManager::readResponse(std::string resopnsefile, PhenoData & phe)
{
	std::ifstream infile;
	std::vector<float> yvector;
	yvector.clear();
	infile.open(resopnsefile);
	if (!infile.is_open())
	{
		throw  ("Error: Cannot open [" + resopnsefile + "] to read.");
	}
	int id = 0;
	int missing = 0;
	while (!infile.eof())
	{
		std::string str;
		getline(infile, str);
		if (!str.empty() && str.back() == 0x0D)
			str.pop_back();
		if (infile.fail())
		{
			continue;
		}
		boost::algorithm::trim(str);
		if (str.empty())
		{
			missing++;
			continue;
		}
		std::vector<std::string> strVec;
		boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"), boost::token_compress_on);
		if (strVec[2]=="-9")
		{
			continue;
		}
		std::string fid_iid = strVec[0] + "_" + strVec[1];
		phe.fid_iid.insert({ id++, fid_iid });
		if (abs(stof(strVec[2])-0)>1e-7 && abs(stof(strVec[2]) - 1) > 1e-7)
		{
			phe.isbinary = false;
		}
		yvector.push_back(stof(strVec[2]));
	}
	infile.close();
	int nind = yvector.size();
	std::stringstream ss;
	if (phe.isbinary)
	{
		ss << "The Phenotype is considered as binary traits." << std::endl;
	}
	ss << "Reading "<< nind << " individuals, and missing "<< missing << std::endl;
	std::cout << ss.str();
	LOG(INFO) << ss.str();
	phe.Phenotype = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(yvector.data(), yvector.size());
	phe.missing = missing;
	Covariates.resize(phe.fid_iid.size(),1);
	Covariates.setOnes();
}

void DataManager::readmkernel(std::string mkernel)
{
	std::ifstream klistifstream;
	klistifstream.open(mkernel, std::ios::in);
	if (!klistifstream.is_open())
	{
		throw ("Error: cannot open the file [" + mkernel + "] to read.");
	}
	std::vector<std::string> filenames;
	while (!klistifstream.eof())
	{
		std::string prefix;
		getline(klistifstream, prefix);
		if (!prefix.empty() && prefix.back() == 0x0D)
			prefix.pop_back();
		if (klistifstream.fail())
		{
			continue;
		}
		filenames.push_back(prefix);
	}
	klistifstream.close();
	KernelList.resize(filenames.size());
#pragma omp parallel for
	for (int i = 0; i < filenames.size(); i++)
	{
		KernelReader kreader(filenames[i]);
		kreader.read();
		KernelData tmpData = kreader.getKernel();
		KernelList[i]=tmpData;
	}
}




void DataManager::match(PhenoData &phenotype, KernelData &kernel)
{

	if (phenotype.fid_iid.size()!=Covariates.rows())
	{
		throw ("Error: the number of individuals in phenotype file cannot match the covariates file.\n");
	}
	if (phenotype.fid_iid == kernel.fid_iid)
	{
		return;
	}
	std::cout << "Match genotypes and phenotypes" << std::endl;
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
