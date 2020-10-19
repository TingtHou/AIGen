#include "../include/DataManager.h"

DataManager::DataManager()
{
}

void DataManager::match()
{
	std::vector< boost::bimap<int, std::string>> IDLists;
	if (fid_iid_keeping.size())
	{
		IDLists.push_back(fid_iid_keeping);
	}
	for (int i = 0; i < KernelList.size(); i++)
	{
		IDLists.push_back(KernelList[i].fid_iid);
	}
	IDLists.push_back(phe.fid_iid);
	IDLists.push_back(Covs.fid_iid);
	bool isSame = 1;
	for (int i = 1; i < IDLists.size(); i++)
	{
		if (IDLists[i-1]!= IDLists[i])
		{
			isSame = 0;
			break;
		}
	}
	if (isSame)
	{
		return;
	}
	boost::bimap<int, std::string> overlapped = set_difference(IDLists);
	if (!overlapped.size())
	{
		throw std::string("There is a file whose individual is not existed in other files. Please check the input.\n");
	}
	std::cout << overlapped.size()<<" individuals are in common in these files."<< std::endl;
	/////////////
	std::cout << "Matching files" << std::endl;
	PhenoData tmpPhe = phe;
	CovData tmpCovs = Covs;
	int nind = overlapped.size(); //overlap FID_IID

	phe.Phenotype.resize(nind);
	phe.fid_iid.clear();
	Covs.fid_iid.clear();
	Covs.Covariates.resize(nind, tmpCovs.npar);
	int i = 0;
	for (auto it_row = overlapped.left.begin(); it_row != overlapped.left.end(); it_row++)
	{
		std::string rowID = it_row->second;
		auto itcov = tmpCovs.fid_iid.right.find(rowID);
		int cov_ID = itcov->second;
		Covs.Covariates.row(i) << tmpCovs.Covariates.row(cov_ID);

		auto it = tmpPhe.fid_iid.right.find(rowID);
		int OriPheID = it->second;
		phe.Phenotype[i++] = tmpPhe.Phenotype[OriPheID];
	}
	phe.fid_iid = overlapped;
	Covs.fid_iid = overlapped;
	i = 0;
	for (; i < KernelList.size(); i++)
	{
		match_Kernels(KernelList[i], overlapped);
	}
}

void DataManager::match_Kernels(KernelData & kernel, boost::bimap<int, std::string> &overlapped)
{
	KernelData tmpKernel = kernel;
	int nind = overlapped.size(); //overlap FID_IID
	kernel.kernelMatrix.resize(nind, nind);
//	kernel.VariantCountMatrix.resize(nind, nind);
	kernel.fid_iid.clear();
	int i = 0;
	for (auto it_row = overlapped.left.begin(); it_row != overlapped.left.end(); it_row++)
	{
		std::string rowID = it_row->second;
		auto itrow = tmpKernel.fid_iid.right.find(rowID);
		int OriKernelRowID = itrow->second;
		int j = 0;
		for (auto it_col = overlapped.left.begin(); it_col != overlapped.left.end(); it_col++)
		{

			std::string colID = it_col->second;
			auto itcol = tmpKernel.fid_iid.right.find(colID);
			int OriKernelColID = itcol->second;
			kernel.kernelMatrix(i, j)=tmpKernel.kernelMatrix(OriKernelRowID, OriKernelColID);
//			kernel.VariantCountMatrix(i, j) = tmpKernel.VariantCountMatrix(OriKernelRowID, OriKernelColID);
			j++;
		}
		i++;

	}
	kernel.fid_iid = overlapped;
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


void DataManager::readCovariates_quantitative(std::string covfilename, CovData & Quantitative)
{
	std::vector<std::vector<float>> covs;
	std::ifstream infile;
	infile.open(covfilename);
	if (!infile.is_open())
	{
		throw  std::string("Error: Cannot open [" + covfilename + "] to read.");
	}
	std::string str;
	getline(infile, str);
	std::vector<std::string> strVec;
	boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"), boost::token_compress_on);
	for (int i = 2; i < strVec.size(); i++)
	{
		Quantitative.names.push_back(strVec[i]);
	}
	int id = 0;
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
		std::string fid_iid = strVec[0] + "_" + strVec[1];
		Quantitative.fid_iid.insert({ id++, fid_iid });
		std::vector<float> floatVector;
		floatVector.reserve(strVec.size()-2);
		transform(strVec.begin()+2, strVec.end(), back_inserter(floatVector),
			[](string const& val) {return stof(val); });
		covs.push_back(floatVector);
	}
	infile.close();
	Quantitative.Covariates.resize(covs.size(), covs[0].size());
	#pragma omp parallel for
	for (int i = 0; i < Quantitative.Covariates.rows(); i++)
	{
		for (int j = 0; j < Quantitative.Covariates.cols(); j++)
		{
			Quantitative.Covariates(i, j) = covs[i][j];
		}
	}
	Quantitative.nind = Quantitative.fid_iid.size();
	Quantitative.npar = Quantitative.Covariates.cols();
}

void DataManager::readCovariates_Discrete(std::string covfilename, CovData & Discrete)
{
	
	std::ifstream infile;
	infile.open(covfilename);
	if (!infile.is_open())
	{
		throw  std::string("Error: Cannot open [" + covfilename + "] to read.");
	}
	std::string str;
	getline(infile, str);
	std::vector<std::string> strVec;
	std::vector<std::string> ori_names;
	boost::algorithm::split(strVec, str, boost::algorithm::is_any_of(" \t"), boost::token_compress_on);
	for (int i = 2; i < strVec.size(); i++)
	{
		ori_names.push_back(strVec[i]);
	}
	std::vector<std::string> ref(strVec.size() - 2);
	std::vector<std::vector<std::string>> covs(strVec.size() - 2);
	int id = 0;
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
		std::string fid_iid = strVec[0] + "_" + strVec[1];
		Discrete.fid_iid.insert({ id, fid_iid });
		for (int i = 0; i < covs.size(); i++)
		{
			covs[i].push_back(strVec[i+2]);
			if (!id)
			{
				ref[i]=strVec[i + 2];
			}
		}
		id++;
	}
	infile.close();
	std::vector<std::vector<std::string>> categorties;
	int ncols = 0;
	for (int i = 0; i < covs.size(); i++)
	{
		std::vector<std::string> categories_all = UniqueCount(covs[i]);
		categories_all.erase(std::remove(categories_all.begin(), categories_all.end(), ref[i]), categories_all.end());
		ncols += categories_all.size();
		categorties.push_back(categories_all);
	}
	for (int i = 0; i < ori_names.size(); i++)
	{
		for (int j = 0; j < categorties[i].size(); j++)
		{
			Discrete.names.push_back(ori_names[i] + ":" + categorties[i][j]);
		}
	}
	Discrete.Covariates.resize(Discrete.fid_iid.size(), ncols);
	for (int i = 0; i < Discrete.Covariates.rows(); i++)
	{
		int col_ID = 0;
		for (int j = 0; j < covs.size(); j++)
		{
			for (int k = 0; k < categorties[j].size(); k++)
			{
				if (covs[j][i]== categorties[j][k])
				{
					Discrete.Covariates(i,col_ID++) = 1;
				}
				else
				{
					Discrete.Covariates(i, col_ID++) = 0;
				}
			}
		}
	}
	Discrete.nind = Discrete.fid_iid.size();
	Discrete.npar = Discrete.Covariates.cols();
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
		throw std::string(ss.str());
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
		throw  std::string("Error: Cannot open [" + filename + "] to read.");
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
			throw std::string(ss.str());
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

void DataManager::readCovariates(std::string qfilename, std::string dfilename)
{
	CovData qCov;
	CovData dCov;
	if (qfilename.empty() && dfilename.empty())
	{
		return;
	}
	if (!qfilename.empty())
	{
		std::cout << "Reading the covariate file [" << qfilename << "]" << std::endl;
		LOG(WARNING) << "Reading the covariate file [" << qfilename << "]";
		readCovariates_quantitative(qfilename, qCov);
		if (!qCov.nind)
		{
			std::cout << "The covariate file [" << qfilename << "] is empty." << std::endl;
			LOG(WARNING) << "The covariate file [" << qfilename << "] is empty.";
		}
		else
		{
			std::cout << "Read "<<qCov.nind<<" individuals with "<< qCov.npar <<" covariates from ["<< qfilename <<"]."<< std::endl;
			LOG(WARNING) << "Read " << qCov.nind << " individuals with " << qCov.npar << " covariates from [" << qfilename << "].";
		}
	}
	if (!dfilename.empty())
	{
		std::cout << "Reading the covariate file [" << dfilename << "]" << std::endl;
		LOG(WARNING) << "Reading the covariate file [" << dfilename << "]";
		readCovariates_Discrete(dfilename, dCov);
		if (!dCov.nind)
		{
			std::cout << "The covariate file [" << dfilename << "] is empty." << std::endl;
			LOG(WARNING) << "The covariate file [" << dfilename << "] is empty.";
		}
		else
		{
			std::cout << "Read " << dCov.nind << " individuals with " << dCov.npar << " dummy covariates from [" << dfilename << "]." << std::endl;
			LOG(WARNING) << "Read " << dCov.nind << " individuals with " << dCov.npar << " dummy covariates from [" << dfilename << "].";
		}
	}
	
	if (qCov.nind && dCov.nind)
	{
		if (qCov.fid_iid == dCov.fid_iid)
		{
			Covs.Covariates.resize(dCov.nind, dCov.npar + qCov.npar + 1);
			Covs.names.resize(dCov.npar + qCov.npar + 1);
			Covs.names[0] = "intercept";
			std::copy(qCov.names.begin(), qCov.names.end(), Covs.names.begin() + 1);
			std::copy(dCov.names.begin(), dCov.names.end(), Covs.names.begin() + qCov.npar + 1);
			Covs.fid_iid = dCov.fid_iid;
			Eigen::MatrixXf intercept(dCov.nind, 1);
			intercept.setOnes();
			Covs.Covariates << intercept, qCov.Covariates, dCov.Covariates;
		}
		else
		{
			std::vector<std::string> overlapID;
			set_difference(qCov.fid_iid, dCov.fid_iid, overlapID);
			int nind = overlapID.size(); //overlap FID_IID
			Covs.fid_iid.clear();
			Covs.Covariates.resize(nind, dCov.npar + qCov.npar + 1);
			Covs.names.resize(dCov.npar + qCov.npar + 1);
			Covs.names[0] = "intercept";
			std::copy(qCov.names.begin(), qCov.names.end(), Covs.names.begin() + 1);
			std::copy(dCov.names.begin(), dCov.names.end(), Covs.names.begin() + qCov.npar + 1);
			for (int i = 0; i < nind; i++)
			{
				std::string ID = overlapID[i];
				auto qit = qCov.fid_iid.right.find(ID);
				auto dit = dCov.fid_iid.right.find(ID);
				int q_ID = qit->second;
				int d_ID = dit->second;
				Covs.Covariates.row(i) << 1, qCov.Covariates.row(q_ID), dCov.Covariates.row(d_ID);
				Covs.fid_iid.insert({ i, ID });
			}
		}
	}
	else
	{
		if (!qCov.nind || !dCov.nind)
		{
			CovData tmpCovs = qCov.nind ? qCov : dCov;
			Covs.Covariates.resize(tmpCovs.nind, tmpCovs.npar + 1);
			Covs.names.resize(tmpCovs.npar + 1);
			Covs.names[0] = "intercept";
			std::copy(tmpCovs.names.begin(), tmpCovs.names.end(), Covs.names.begin() + 1);
			Covs.fid_iid = tmpCovs.fid_iid;
			Eigen::MatrixXf intercept(tmpCovs.nind, 1);
			intercept.setOnes();
			Covs.Covariates << intercept, tmpCovs.Covariates;
		}
	}
	Covs.nind = Covs.fid_iid.size();
	Covs.npar = Covs.Covariates.cols();

}

void DataManager::readkeepFile(std::string filename)
{
	std::ifstream infile;
	infile.open(filename);
	if (!infile.is_open())
	{
		throw  std::string("Error: Cannot open [" + filename + "] to read.");
	}
	int id = 0;
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
		if (strVec.size()<2)
		{
			throw std::string("The format of keep file is not correct.");
		}
		std::string fid_iid = strVec[0] + "_" + strVec[1];
		fid_iid_keeping.insert({ id++, fid_iid });
		
	}
	infile.close();
	std::cout << fid_iid_keeping.size() << " individuals is listed in [" << filename << "]." << std::endl;
	LOG(INFO)<< fid_iid_keeping.size() << " individuals is listed in [" << filename << "]." << std::endl;
}

void DataManager::readResponse(std::string resopnsefile, PhenoData & phe)
{
	std::ifstream infile;
	std::vector<float> yvector;
	yvector.clear();
	infile.open(resopnsefile);
	if (!infile.is_open())
	{
		throw  std::string("Error: Cannot open [" + resopnsefile + "] to read.");
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
	
		float sum = std::accumulate(yvector.begin(), yvector.end(), 0.0);
		phe.prevalence = sum / yvector.size();
		ss << "The Phenotype is considered as binary traits, whose prevalence is "<< phe.prevalence <<"."<< std::endl;
	}
	ss << "Reading "<< nind << " individuals, and missing "<< missing << std::endl;
	std::cout << ss.str();
	LOG(INFO) << ss.str();
	phe.Phenotype = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(yvector.data(), yvector.size());
	phe.missing = missing;
	Covs.Covariates.resize(phe.fid_iid.size(),1);
	Covs.Covariates.setOnes();
	Covs.fid_iid = phe.fid_iid;
	Covs.nind = Covs.fid_iid.size();
	Covs.npar = Covs.Covariates.cols();
}

void DataManager::readmkernel(std::string mkernel)
{
	std::ifstream klistifstream;
	klistifstream.open(mkernel, std::ios::in);
	if (!klistifstream.is_open())
	{
		std::stringstream ss;
		ss << "Error: cannot open the file [" + mkernel + "] to read.";
		throw ss.str().c_str();
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

	int nthread = omp_get_max_threads();
	int threadInNest = 1;
	if (nthread != 1)
	{
		int tmp_thread = nthread > filenames.size() ? filenames.size() : nthread;
		omp_set_num_threads(tmp_thread);
		threadInNest = nthread / filenames.size();
	}
	#pragma omp parallel for
	for (int i = 0; i < filenames.size(); i++)
	{
		omp_set_num_threads(threadInNest);
		KernelReader kreader(filenames[i]);
		kreader.read();
		KernelData tmpData = kreader.getKernel();
		KernelList[i]=tmpData;
	}
	omp_set_num_threads(nthread);
}
/*
void DataManager::match(PhenoData &phenotype, KernelData &kernel)
{

	if (phenotype.fid_iid == kernel.fid_iid)
	{
		return;
	}
	std::cout << "Match genotypes and phenotypes" << std::endl;
	PhenoData tmpPhe = phenotype;
	KernelData tmpKernel = kernel;
	CovData tmpCovs = Covs;
	std::vector<std::string> overlapID;
	set_difference(phenotype.fid_iid, kernel.fid_iid, Covs.fid_iid, overlapID);
	int nind = overlapID.size(); //overlap FID_IID
	kernel.kernelMatrix.resize(nind, nind);
	kernel.VariantCountMatrix.resize(nind, nind);
	kernel.fid_iid.clear();
	phenotype.Phenotype.resize(nind);
	phenotype.fid_iid.clear();
	Covs.fid_iid.clear();
	Covs.Covariates.resize(nind, tmpCovs.npar);
	for (int i = 0; i < nind; i++)
	{
		std::string rowID = overlapID[i];
		auto itrow = tmpKernel.fid_iid.right.find(rowID);
		auto itcov = tmpCovs.fid_iid.right.find(rowID);
		int cov_ID = itcov->second;
		Covs.Covariates.row(i) <<tmpCovs.Covariates.row(cov_ID);
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
		Covs.fid_iid.insert({ i, rowID });
	}
}
*/