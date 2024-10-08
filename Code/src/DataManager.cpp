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
		IDLists.push_back(KernelList[i]->fid_iid);
	}
	if (phe.fid_iid.size())
	{
		IDLists.push_back(phe.fid_iid);
	}
	
	if (Covs.fid_iid.size()>0)
	{
		IDLists.push_back(Covs.fid_iid);
	}
	
	if (geno.fid_iid.size()>0)
	{
		IDLists.push_back(geno.fid_iid);
	}
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
	GenoData tmpGen = geno;
	int nind = overlapped.size(); //overlap FID_IID

	if (tmpPhe.isBalance)
	{
		phe.Phenotype.resize(nind, tmpPhe.Phenotype.cols());
	}
	else
	{
		phe.vPhenotype.clear();
		phe.vloc.clear();
	}
	phe.fid_iid.clear();
	Covs.fid_iid.clear();
	geno.fid_iid.clear();
	Covs.Covariates.resize(nind, tmpCovs.npar);
	geno.Geno.resize(nind, tmpGen.pos.size());
	int i = 0;
	for (auto it_row = overlapped.left.begin(); it_row != overlapped.left.end(); it_row++)
	{
		std::string rowID = it_row->second;
		auto itcov = tmpCovs.fid_iid.right.find(rowID);
		auto itgen = tmpGen.fid_iid.right.find(rowID);
		int gen_ID = itgen->second;
		int cov_ID = itcov->second;
		if (itcov!= tmpCovs.fid_iid.right.end())
		{
			Covs.Covariates.row(i) << tmpCovs.Covariates.row(itcov->second);
		}
		if (itgen != tmpGen.fid_iid.right.end())
		{
			geno.Geno.row(i) << tmpGen.Geno.row(itgen->second);
		}
	
		auto it = tmpPhe.fid_iid.right.find(rowID);
		int OriPheID = it->second;
		if (tmpPhe.isBalance)
		{
			phe.Phenotype.row(i) = tmpPhe.Phenotype.row(OriPheID);
		}
		else
		{
			phe.vPhenotype.push_back(tmpPhe.vPhenotype[OriPheID]);
			phe.vloc.push_back(tmpPhe.vloc[OriPheID]);
		}
		i++;
	
	}
	phe.fid_iid = overlapped;
	Covs.fid_iid = overlapped;
	geno.fid_iid = overlapped;
	phe.nind = overlapped.size();
	/*
	for (size_t i = 0; i < KernelList.size(); i++)
	{
		std::stringstream ss;
		ss << "kernel: " << i << "before matching" << std::endl;
		ss << "First 10x10: \n" << KernelList[i]->kernelMatrix.block(0, 0, 10, 10) << std::endl;
		ss << "Last 10x10: \n" << KernelList[i]->kernelMatrix.block(KernelList[i]->kernelMatrix.rows() - 10, KernelList[i]->kernelMatrix.cols() - 10, 10, 10);
		LOG(INFO) << ss.str() << std::endl;
	}
	*/
	i = 0;
	for (; i < KernelList.size(); i++)
	{
		KernelList[i]=match_Kernels(KernelList[i], overlapped);
	}
	/*
	for (size_t i = 0; i < KernelList.size(); i++)
	{
		std::stringstream ss;
		ss << "kernel: " << i << "after matching" << std::endl;
		ss << "First 10x10: \n" << KernelList[i]->kernelMatrix.block(0, 0, 10, 10) << std::endl;
		ss << "Last 10x10: \n" << KernelList[i]->kernelMatrix.block(KernelList[i]->kernelMatrix.rows() - 10, KernelList[i]->kernelMatrix.cols() - 10, 10, 10);
		LOG(INFO) << ss.str() << std::endl;
	}
	*/
	std::cout << "After matching, there are " << overlapped.size() << " individuals included for the analysis." << std::endl;
	LOG(INFO) << "After matching, there are " << overlapped.size() << " individuals included for the analysis.";
}

/*
std::tuple<std::shared_ptr<Dataset>, std::shared_ptr<Dataset>>  DataManager::split(float seed,float ratio)
{
	std::stringstream ss;
	ss << "Split the dataset into two sub-dataset, training, testing";
	std::cout << ss.str() << std::endl;
	LOG(INFO) << ss.str();
	std::default_random_engine e(seed);
	int64_t num = phe.fid_iid.size();
	int64_t train_num = (float)num * ratio;
	std::vector<int64_t> fid_iid_split(num);
	std::iota(std::begin(fid_iid_split), std::end(fid_iid_split), 0); // Fill with 0, 1, ..., 99.
	std::shuffle(fid_iid_split.begin(), fid_iid_split.end(), e);
	std::shared_ptr<Dataset> train=std::make_shared<Dataset>();
	std::shared_ptr<Dataset> test = std::make_shared<Dataset>();
	train->phe = phe;
	train->cov = Covs;
	train->geno = geno;
	test->phe = phe;
	test->cov = Covs;
	test->geno = geno;
	if (phe.isBalance)
	{
		train->phe.Phenotype.resize(train_num, phe.Phenotype.cols());
		test->phe.Phenotype.resize(num-train_num, phe.Phenotype.cols());
		//Check if the response are 1d, and will be interpolated at a single knot.
		//If the responses are multivariate, and will be interpolated at same multi-knots, keep the knots vectors in train, and test dataset
		//otherwise, the knots will be chosen according to response
		if (phe.Phenotype.cols()==1 && phe.loc.size()>0)
		{
			train->phe.loc.resize(train_num);
			test->phe.loc.resize(num - train_num);
		}
	}
	else
	{
		train->phe.vPhenotype.clear();
		train->phe.vloc.clear();
		test->phe.vPhenotype.clear();
		test->phe.vloc.clear();
	}
	train->phe.fid_iid.clear();
	train->cov.fid_iid.clear();
	train->geno.fid_iid.clear();
	train->cov.Covariates.resize(train_num, Covs.npar);
	train->geno.Geno.resize(train_num, geno.pos.size());
	////////////////////////////////////////
	test->phe.fid_iid.clear();
	test->cov.fid_iid.clear();
	test->geno.fid_iid.clear();
	test->cov.Covariates.resize(num-train_num, Covs.npar);
	test->geno.Geno.resize(num-train_num, geno.pos.size());
	boost::bimap<int, std::string> fid_iid_train;
	boost::bimap<int, std::string> fid_iid_test;
	int64_t train_id = 0;
	int64_t test_id = 0;
	for (int64_t i=0;i<num;i++)
	{
		//std::string rowID = it_row->second;
		int row_index = fid_iid_split[i];
		auto fid_iid = phe.fid_iid.left.find(row_index);
		if (i <train_num)
		{
			if (Covs.nind)
			{
				train->cov.Covariates.row(train_id) << Covs.Covariates.row(row_index);
			}
			if (geno.fid_iid.size()!=0)
			{
				train->geno.Geno.row(train_id) << geno.Geno.row(row_index);
			}
		
			if (phe.isBalance)
			{
				train->phe.Phenotype.row(train_id) = phe.Phenotype.row(row_index);
				if (phe.Phenotype.cols() == 1 && phe.loc.size() > 0)
				{
					train->phe.loc(train_id) = phe.loc(row_index);
				}
				train_id++;
			}
			else
			{
				train->phe.vPhenotype.push_back(phe.vPhenotype[row_index]);
				train->phe.vloc.push_back(phe.vloc[row_index]);
			}
			fid_iid_train.insert({ fid_iid->first,fid_iid->second });
		}
		else
		{
			if (Covs.nind)
			{
				test->cov.Covariates.row(test_id) << Covs.Covariates.row(row_index);
			}
			if (geno.fid_iid.size() != 0)
			{
				test->geno.Geno.row(test_id) << geno.Geno.row(row_index);
			}
			if (phe.isBalance)
			{
				test->phe.Phenotype.row(test_id++) = phe.Phenotype.row(row_index);
			}
			else
			{
				test->phe.vPhenotype.push_back(phe.vPhenotype[row_index]);
				test->phe.vloc.push_back(phe.vloc[row_index]);
			}
			fid_iid_test.insert({ fid_iid->first,fid_iid->second });
		}
	}
	train->phe.fid_iid = fid_iid_train;
	train->cov.fid_iid = fid_iid_train;
	train->geno.fid_iid = fid_iid_train;
	train->phe.nind = fid_iid_train.size();
	test->phe.fid_iid = fid_iid_test;
	test->cov.fid_iid = fid_iid_test;
	test->geno.fid_iid = fid_iid_test;
	test->phe.nind = fid_iid_test.size();
	std::stringstream ss1;
	ss1 << "Spliting Completed. There are "<< train_num <<" individuals in the training dataset, and "<<num-train_num<< " individuals in the testing dataset";
	std::cout << ss1.str() << std::endl;
	LOG(INFO) << ss1.str();
	return std::make_tuple(train, test);
}
*/
std::shared_ptr<KernelData> DataManager::match_Kernels(std::shared_ptr<KernelData> kernel, boost::bimap<int, std::string> &overlapped)
{
	if (overlapped == kernel->fid_iid)
	{
		return kernel;
	}
	std::shared_ptr<KernelData> match_kernel = std::make_shared<KernelData>();
	//KernelData tmpKernel = kernel;
	long long nind = overlapped.size(); //overlap FID_IID
	match_kernel->kernelMatrix=std::make_shared<Eigen::MatrixXf>(nind, nind);
//	kernel.VariantCountMatrix.resize(nind, nind);
	match_kernel->fid_iid.clear();
//	int i = 0;
	long long criteria = (nind + 1) * nind / 2;
#pragma omp parallel for shared(kernel,match_kernel, overlapped)
	for (long long k = 0; k < criteria; k++)
	{
		unsigned long long tmp_K = k;
		unsigned long long i = tmp_K / nind, j = tmp_K % nind;
		if (j < i) i = nind - i, j = nind - j - 1;
		auto row_it = overlapped.left.find(i);
		auto col_it = overlapped.left.find(j);
		std::string rowID = row_it->second;
		std::string colID = col_it->second;
		auto itrow = kernel->fid_iid.right.find(rowID);
		auto itcol = kernel->fid_iid.right.find(colID);
		int OriKernelRowID = itrow->second;
		int OriKernelColID = itcol->second;
		(*match_kernel->kernelMatrix)(j, i)= (*match_kernel->kernelMatrix)(i, j) = (*kernel->kernelMatrix)(OriKernelRowID, OriKernelColID);
	}
	std::stringstream ss;
	ss << "kernel: " << "in matching" << std::endl;
	ss << "First 10x10: \n" << match_kernel->kernelMatrix->block(0, 0, 10, 10) << std::endl;
	ss << "Last 10x10: \n" << match_kernel->kernelMatrix->block(match_kernel->kernelMatrix->rows() - 10, match_kernel->kernelMatrix->cols() - 10, 10, 10);
	LOG(INFO) << ss.str() << std::endl;
	/*
#pragma omp parallel for
	for (boost::bimap< int, std::string >::left_map::const_iterator it_row = overlapped.left.begin(); it_row != overlapped.left.end(); it_row++)
	{
		std::string rowID = it_row->second;
		auto itrow = tmpKernel->fid_iid.right.find(rowID);
		int OriKernelRowID = itrow->second;
		long long i = it_row->first; //new row id
	//	int j = 0; 
		for (boost::bimap< int, std::string >::left_map::const_iterator it_col = overlapped.left.begin(); it_col != overlapped.left.end(); it_col++)
		{

			std::string colID = it_col->second;
			auto itcol = tmpKernel->fid_iid.right.find(colID);
			int OriKernelColID = itcol->second;
			long long j = it_col->first;  //new col id
			kernel->kernelMatrix(i, j)=tmpKernel->kernelMatrix(OriKernelRowID, OriKernelColID);
//			kernel.VariantCountMatrix(i, j) = tmpKernel.VariantCountMatrix(OriKernelRowID, OriKernelColID);
		//	j++;
		}
		//i++;
	}*/
	match_kernel->fid_iid = overlapped;
	return match_kernel;
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
	KernelReader kreader(prefix);
	kreader.read();
	KernelList.push_back(kreader.getKernel());

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
	boost::algorithm::trim(str);
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
			[](std::string const& val) {return  std::stof(val); });
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
	boost::algorithm::trim(str);
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

void DataManager::readCovariates(std::string qfilename, std::string dfilename, bool intercept)
{
	CovData qCov;
	CovData dCov;
	if (qfilename.empty() && dfilename.empty())
	{
		if (intercept)
		{
			Covs.Covariates.resize(phe.fid_iid.size(), 1);
			Covs.Covariates.setOnes();
			Covs.names.resize(intercept);
			Covs.names[0] = "intercept";
			Covs.fid_iid = phe.fid_iid;
			Covs.nind = Covs.fid_iid.size();
			Covs.npar = Covs.Covariates.cols();
		}
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
			Covs.Covariates.resize(dCov.nind, dCov.npar + qCov.npar + intercept);
			Covs.names.resize(dCov.npar + qCov.npar + intercept);
			
			std::copy(qCov.names.begin(), qCov.names.end(), Covs.names.begin() + intercept);
			std::copy(dCov.names.begin(), dCov.names.end(), Covs.names.begin() + qCov.npar + intercept);
			Covs.fid_iid = dCov.fid_iid;
			if (intercept)
			{
				Covs.names[0] = "intercept";
				Eigen::MatrixXf Ones(dCov.nind, 1);
				Ones.setOnes();
				Covs.Covariates << Ones, qCov.Covariates, dCov.Covariates;
			}
			else
			{
				Covs.Covariates << qCov.Covariates, dCov.Covariates;
			}
		
		}
		else
		{
			std::vector<std::string> overlapID;
			set_difference(qCov.fid_iid, dCov.fid_iid, overlapID);
			int nind = overlapID.size(); //overlap FID_IID
			Covs.fid_iid.clear();
			Covs.Covariates.resize(nind, dCov.npar + qCov.npar + intercept);
			Covs.names.resize(dCov.npar + qCov.npar + 1);
			if (intercept)
			{
				Covs.names[0] = "intercept";
			}
			std::copy(qCov.names.begin(), qCov.names.end(), Covs.names.begin() + intercept);
			std::copy(dCov.names.begin(), dCov.names.end(), Covs.names.begin() + qCov.npar + intercept);
			for (int i = 0; i < nind; i++)
			{
				std::string ID = overlapID[i];
				auto qit = qCov.fid_iid.right.find(ID);
				auto dit = dCov.fid_iid.right.find(ID);
				int q_ID = qit->second;
				int d_ID = dit->second;
				if (intercept)
				{
					Covs.Covariates.row(i) << 1, qCov.Covariates.row(q_ID), dCov.Covariates.row(d_ID);
				}
				else
				{
					Covs.Covariates.row(i) << qCov.Covariates.row(q_ID), dCov.Covariates.row(d_ID);
				}
			
				Covs.fid_iid.insert({ i, ID });
			}
		}
	}
	else
	{
		if (!qCov.nind || !dCov.nind)
		{
			CovData tmpCovs = qCov.nind ? qCov : dCov;
			Covs.Covariates.resize(tmpCovs.nind, tmpCovs.npar + intercept);
			Covs.names.resize(tmpCovs.npar + intercept);
		
			std::copy(tmpCovs.names.begin(), tmpCovs.names.end(), Covs.names.begin() + intercept);
			Covs.fid_iid = tmpCovs.fid_iid;
			if (intercept)
			{
				Covs.names[0] = "intercept";
				Eigen::MatrixXf Ones(tmpCovs.nind, 1);
				Ones.setOnes();
				Covs.Covariates << Ones, tmpCovs.Covariates;
			}
			else
			{
				Covs.Covariates << tmpCovs.Covariates;
			}
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

CovData DataManager::GetCovariates()
{
	return Covs;
}

std::shared_ptr<Dataset> DataManager::GetDataset()
{
	auto dataset = std::make_shared<Dataset>();
	dataset->cov = Covs;
	dataset->geno = geno;
	dataset->phe = phe;
	return dataset;
}

void DataManager::SetKernel(std::vector<std::shared_ptr<KernelData>> KernelList)
{
	//this->KernelList.clear();
	this->KernelList.resize(KernelList.size());
	for (int i = 0; i < KernelList.size(); i++)
	{
		this->KernelList[i] =KernelList[i];
	}
}

void DataManager::shuffle(float seed, float ratio)
{

	long long nInd = phe.fid_iid.size();

	auto rng = std::default_random_engine{};
	rng.seed(seed);
	
	PhenoData phe_new;
	phe_new.Phenotype.resize(phe.Phenotype.rows(), phe.Phenotype.cols());
	if (phe.dataType==0)
	{
		std::vector<int> shuffledID(nInd);
		#pragma omp parallel for
		for (int i = 0; i < nInd; i++)
		{
			shuffledID[i] = i;
		}
		std::shuffle(std::begin(shuffledID), std::end(shuffledID), rng);

		for (int i = 0; i < nInd; i++)
		{
			int row_index = shuffledID[i];
			auto fid_iid = phe.fid_iid.left.find(row_index);
			phe_new.Phenotype(i, 0) = phe.Phenotype(row_index, 0);
			phe_new.fid_iid.insert({ i ,fid_iid->second });
		}
	
	}
	else if (phe.dataType==1)
	{
		long long train_size = phe.nind * ratio;
		long long test_size = phe.nind - train_size;
		std::vector<int> fid_iid_case;
		std::vector<int> fid_iid_control;
		for (int i = 0; i < phe.nind; i++)
		{
			if (abs(phe.Phenotype(i,0) - 1) < 1e-6)
			{
				fid_iid_case.push_back(i);
			}
			else
			{
				fid_iid_control.push_back(i);
			}
		}
		std::shuffle(std::begin(fid_iid_case), std::end(fid_iid_case), rng);
		std::shuffle(std::begin(fid_iid_control), std::end(fid_iid_control), rng);
		long long case_training = fid_iid_case.size() * ratio;
		long long case_testing = fid_iid_case.size() - case_training;
		long long control_training = fid_iid_control.size() * ratio;
		long long control_testing = fid_iid_control.size() - control_training;
		std::vector<int> ID_training;
		std::vector<int> ID_testing;
		for (size_t i = 0; i < fid_iid_case.size(); i++)
		{
			if (i<case_training)
			{
				ID_training.push_back(fid_iid_case[i]);
			}
			else
			{
				ID_testing.push_back(fid_iid_case[i]);
			}
			
		}
		for (size_t i = 0; i < fid_iid_control.size(); i++)
		{
			if (i< control_training)
			{
				ID_training.push_back(fid_iid_control[i]);
			}
			else
			{
				ID_testing.push_back(fid_iid_control[i]);
			}
		
		}
		int id = 0;
		for (int i = 0; i < ID_training.size(); i++)
		{
			int row_index = ID_training[i];
			auto fid_iid = phe.fid_iid.left.find(row_index);
			phe_new.Phenotype(id, 0) = phe.Phenotype(row_index, 0);
			phe_new.fid_iid.insert({ id ,fid_iid->second });
			id++;
		}
		for (int i = 0; i < ID_testing.size(); i++)
		{
			int row_index = ID_testing[i];
			auto fid_iid = phe.fid_iid.left.find(row_index);
			phe_new.Phenotype(id, 0) = phe.Phenotype(row_index, 0);
			phe_new.fid_iid.insert({ id ,fid_iid->second });
			id++;
		}
	}
	phe.Phenotype = phe_new.Phenotype;
	phe.fid_iid = phe_new.fid_iid;
	match();
}



void DataManager::readResponse(std::string resopnsefile, PhenoData & phe)
{
	std::ifstream infile;
	std::vector<std::vector<float>> yvector;
	std::vector<std::vector<float>> locvector;
	yvector.clear();
	infile.open(resopnsefile);
	if (!infile.is_open())
	{
		throw  std::string("Error: Cannot open [" + resopnsefile + "] to read.");
	}
	int id = 0;
	int missing = 0;
	std::vector<double> all_y;
	bool singleResponse=true;
	bool isbinary = true;
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
		auto itrow = phe.fid_iid.right.find(fid_iid);
		if (itrow == phe.fid_iid.right.end())
		{
			phe.fid_iid.insert({ id++, fid_iid });
			std::vector<float> newY;
			double v = std::stod(strVec[2]);
			all_y.push_back(v);
			newY.push_back((float) v);
			yvector.push_back(newY);
			if (strVec.size() == 4)
			{
				std::vector<float> newLoc;
				double loci = std::stod(strVec[3]);
				newLoc.push_back((float) loci);
				locvector.push_back(newLoc);
			}
		}
		else
		{
			int64_t index = itrow->second;
			double v = std::stod(strVec[2]);
			all_y.push_back(v);
			yvector[index].push_back((float)v);
			if (strVec.size() == 4)
			{
				double loci = std::stod(strVec[3]);
				locvector[index].push_back((float)loci);
			}
			singleResponse = false;
		}
		if (abs(stod(strVec[2])-0)>1e-7 && abs(stod(strVec[2]) - 1) > 1e-7 && isbinary)
		{
			isbinary = false;
		}
	
	}
	phe.dataType = isbinary;
	infile.close();
	int nind = yvector.size();
	phe.nind = nind;
	bool isbalanced=true;
//	bool same_value = true;
//	bool same_size = true;
	std::vector<float> loc_all;
	for (int64_t i = 0; i < locvector.size(); i++)
	{
		phe.vloc.push_back(Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(locvector[i].data(), locvector[i].size()));
		loc_all.insert(loc_all.end(), locvector[i].begin(), locvector[i].end());
		if (i>0)
		{
			if (locvector[i - 1].size() != locvector[i].size())
			{
				isbalanced = false;
			}
			else
			{
				if (!std::equal(locvector[i - 1].begin(), locvector[i - 1].end(), locvector[i].begin()))
				{
					isbalanced = false;
				}
			}
		}
	}
	phe.isBalance = isbalanced || singleResponse;
	phe.isUnivariate = singleResponse;
	if (isbalanced)
	{
		phe.Phenotype.resize(yvector.size(), yvector[0].size());
		if (phe.vloc.size())
		{
			phe.loc = phe.vloc[0];
		}
		
	}
	else
	{
		if (singleResponse)
		{
			phe.loc = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(loc_all.data(), loc_all.size());
		}
	}
	if (singleResponse)
	{
		phe.Phenotype.resize(yvector.size(), yvector[0].size());
	}

	for (int64_t i = 0; i < yvector.size(); i++)
	{
		phe.vPhenotype.push_back(Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(yvector[i].data(), yvector[i].size()));
		if (phe.isBalance)
		{
			phe.Phenotype.row(i) = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(yvector[i].data(), yvector[i].size());
		}
	}
	double sum = std::accumulate(all_y.begin(), all_y.end(), 0.0);
	double mean = sum / all_y.size();

	std::vector<double> diff(all_y.size());
	std::transform(all_y.begin(), all_y.end(), diff.begin(), [mean](double x) { return x - mean; });
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double stdev = std::sqrt(sq_sum / all_y.size());

	phe.mean = mean;
	phe.std = stdev;
	std::stringstream ss;
//	if (phe.dataType ==1)
//	{

		//float sum = std::accumulate(yvector.begin(), yvector.end(), 0.0);
//		phe.prevalence = phe.Phenotype.colwise().mean();
//		ss << "The Phenotype is considered as binary traits, whose prevalence is " << phe.prevalence.transpose() << "." << std::endl;
//	}
	ss << "Reading " << nind << " individuals, and missing " << missing <<". Total mean is "<< phe.mean <<" and standard deviation is "<<phe.std<<"."<<std::endl;
	std::cout << ss.str();
	LOG(INFO) << ss.str();
	//phe.Phenotype = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(yvector.data(), yvector.size());
	phe.missing = missing;

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
	std::string error="";
	#pragma omp parallel for 
	for (int i = 0; i < filenames.size(); i++)
	{
		omp_set_num_threads(threadInNest);
		try {
			KernelReader kreader(filenames[i]);
			kreader.read();
			KernelList[i] = kreader.getKernel();
			LOG(INFO) << "Reading kernel: [" << filenames[i] << "] is completed.";
		}
		catch (std::string & e)
		{
			#pragma omp critical
			error = e;
			//throw e;
		}
	}
	if (error!="")
	{
		throw error;
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