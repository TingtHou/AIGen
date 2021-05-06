#include "../include/KNN_Imp.h"

void BatchMINQUE1(MinqueOptions& minque, std::vector<Eigen::MatrixXf*>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, int nsplit, int seed, int nthread, bool isecho)
{
	int nkernel = Kernels.size();
	bool nofix = coefs[0] == -999 ? true : false;
	Eigen::VectorXf pheV;
	if (phe.Phenotype.cols() == 1)
	{
		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
	}
	else
	{
		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
	}
	Batch b = Batch(Kernels, pheV, Covs, nsplit, seed, true);
	b.start(phe.dataType);
	std::vector<std::vector<Eigen::MatrixXf>> KernelsBatch;
	std::vector<Eigen::VectorXf> PheBatch;
	std::vector<Eigen::MatrixXf> CovBatch;
	b.GetBatchKernels(KernelsBatch);
	b.GetBatchPhe(PheBatch);
	b.GetBatchCov(CovBatch);
	std::vector<float> time(KernelsBatch.size());
	std::vector<std::vector<float>> varsBatch(KernelsBatch.size());
	std::vector<std::vector<float>> fixsBatch(KernelsBatch.size());
	std::cout << "starting CPU MINQUE" << std::endl;
	omp_set_num_threads(nthread);
#pragma omp parallel for
	for (int i = 0; i < KernelsBatch.size(); i++)
	{

		printf("Starting CPU-based analysis on thread %d\n", i);
		LOG(INFO) << "Starting CPU-based analysis on thread " << i;
		imnq varest;
		varest.isEcho(isecho);
		varest.setThreadId(i);
		varest.setOptions(minque);
		varest.importY(PheBatch[i]);
		varest.pushback_X(CovBatch[i], false);

		Eigen::MatrixXf e(PheBatch[i].size(), PheBatch[i].size());
		e.setIdentity();
		for (int j = 0; j < KernelsBatch[i].size(); j++)
		{
			varest.pushback_Vi(&KernelsBatch[i][j]);
		}
		varest.pushback_Vi(&e);
		if (variances.size() != 0)
		{
			varest.pushback_W(variances);
		}
		try
		{
			varest.estimateVCs();
			varsBatch[i].resize(varest.getvcs().size());
			Eigen::VectorXf::Map(&varsBatch[i][0], varest.getvcs().size()) = varest.getvcs();
			if (!nofix)
			{
				varest.estimateFix();
				fixsBatch[i].resize(varest.getfix().size());
				Eigen::VectorXf::Map(&fixsBatch[i][0], varest.getfix().size()) = varest.getfix();
			}
			printf("The thread %d is finished\n", i);
			LOG(INFO) << "The thread " << i << " is finished";
		}
		catch (const std::exception & err)
		{
			stringstream ss;
			ss << "[Warning]: The thread " << i << " is interrupt, because " << err.what();
			printf("%s\n", ss.str().c_str());
			LOG(WARNING) << ss.str();
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
			if (!nofix)
			{
				fixsBatch[i].resize(1);
				fixsBatch[i][0] = -999;
			}
		}
		catch (string & e)
		{
			std::cout << e;
			LOG(ERROR) << e;
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
			if (!nofix)
			{
				fixsBatch[i].resize(1);
				fixsBatch[i][0] = -999;
			}
		}
	}

	for (int i = 0; i < varsBatch.size(); i++)
	{
		stringstream ss;
		ss << "Thread ID: " << i << "\t";
		for (int j = 0; j < varsBatch[i].size(); j++)
		{
			ss << varsBatch[i][j] << "\t";
		}
		LOG(INFO) << ss.str();
	}
	auto it = varsBatch.begin();
	while (it != varsBatch.end())
	{
		auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
		if (itzero != (*it).end())
		{
			it = varsBatch.erase(it);
		}
		else
		{
			++it;
		}
	}
	variances.resize(nkernel + 1);
	for (int i = 0; i < nkernel + 1; i++)
	{
		float sum = 0;
		for (int j = 0; j < varsBatch.size(); j++)
		{
			sum += varsBatch[j][i];
		}
		variances[i] = sum / float(varsBatch.size());
	}

	if (!nofix)
	{
		auto it = fixsBatch.begin();
		while (it != fixsBatch.end())
		{
			auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
			if (itzero != (*it).end())
			{
				it = fixsBatch.erase(it);
			}
			else
			{
				++it;
			}
		}
		coefs.resize(fixsBatch[0].size());
		for (int i = 0; i < coefs.size(); i++)
		{
			float sum = 0;
			for (int j = 0; j < fixsBatch.size(); j++)
			{
				sum += fixsBatch[j][i];
			}
			coefs[i] = sum / float(fixsBatch.size());
		}
	}

	iterateTimes = accumulate(time.begin(), time.end(), 0.0) / time.size(); ;
}

void cMINQUE1(MinqueOptions& minque, std::vector<Eigen::MatrixXf*>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho)
{
	imnq varest;
	varest.setOptions(minque);
	varest.isEcho(isecho);
	Eigen::VectorXf pheV;
	if (phe.Phenotype.cols() == 1)
	{
		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
	}
	else
	{
		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
	}
	varest.importY(pheV);
	varest.pushback_X(Covs, false);
	for (int i = 0; i < Kernels.size(); i++)
	{
		varest.pushback_Vi(Kernels[i]);
	}
	Eigen::MatrixXf e(phe.fid_iid.size(), phe.fid_iid.size());
	e.setIdentity();
	varest.pushback_Vi(&e);
	if (variances.size() != 0)
	{
		varest.pushback_W(variances);
	}
	std::cout << "starting CPU MINQUE(1) " << std::endl;
	varest.estimateVCs();
	variances = varest.getvcs();
	iterateTimes = varest.getIterateTimes();
	if (coefs[0] != -999)
	{
		varest.estimateFix();
		coefs = varest.getfix();
	}

}



void BatchMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf*>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, int nsplit, int seed, int nthread)
{
	int nkernel = Kernels.size();
	bool nofix = coefs[0] == -999 ? true : false;
	Eigen::VectorXf pheV;
	if (phe.Phenotype.cols() == 1)
	{
		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
	}
	else
	{
		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
	}
	Batch b = Batch(Kernels, pheV, Covs, nsplit, seed, true);
	b.start(phe.dataType);
	std::vector<std::vector<Eigen::MatrixXf>> KernelsBatch;
	std::vector<Eigen::VectorXf> PheBatch;
	std::vector<Eigen::MatrixXf> CovBatch;
	b.GetBatchKernels(KernelsBatch);
	b.GetBatchPhe(PheBatch);
	b.GetBatchCov(CovBatch);
	std::vector<float> time(KernelsBatch.size());
	std::vector<std::vector<float>> varsBatch(KernelsBatch.size());
	std::vector<std::vector<float>> fixsBatch(KernelsBatch.size());
	std::cout << "starting CPU MINQUE" << std::endl;
	omp_set_num_threads(nthread);
#pragma omp parallel for
	for (int i = 0; i < KernelsBatch.size(); i++)
	{
		clock_t t1 = clock();
		printf("Starting CPU-based analysis on thread %d\n", i);
		LOG(INFO) << "Starting CPU-based analysis on thread " << i;
		MINQUE0 varest(minque.MatrixDecomposition, minque.altMatrixDecomposition, minque.allowPseudoInverse);
		varest.setThreadId(i);
		varest.importY(PheBatch[i]);
		varest.pushback_X(CovBatch[i], false);
		Eigen::MatrixXf e(PheBatch[i].size(), PheBatch[i].size());
		e.setIdentity();
		for (int j = 0; j < KernelsBatch[i].size(); j++)
		{
			varest.pushback_Vi(&KernelsBatch[i][j]);
		}
		varest.pushback_Vi(&e);
		try
		{
			varest.estimateVCs();
			varsBatch[i].resize(varest.getvcs().size());
			Eigen::VectorXf::Map(&varsBatch[i][0], varest.getvcs().size()) = varest.getvcs();
			if (!nofix)
			{
				varest.estimateFix();
				fixsBatch[i].resize(varest.getfix().size());
				Eigen::VectorXf::Map(&fixsBatch[i][0], varest.getfix().size()) = varest.getfix();
			}
			printf("The thread %d is finished\n", i);
			LOG(INFO) << "The thread " << i << " is finished";
		}
		catch (const std::exception & err)
		{
			stringstream ss;
			ss << "[Warning]: The thread " << i << " is interrupt, because " << err.what();
			std::cout << ss.str();
			//	printf("%s\n", ss.str().);
			LOG(WARNING) << ss.str();
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
			if (!nofix)
			{
				fixsBatch[i].resize(1);
				fixsBatch[i][0] = -999;
			}
		}
		catch (string & e)
		{
			std::cout << e;
			LOG(ERROR) << e;
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
			if (!nofix)
			{
				fixsBatch[i].resize(1);
				fixsBatch[i][0] = -999;
			}
		}
	}
	for (int i = 0; i < varsBatch.size(); i++)
	{
		stringstream ss;
		ss << "Thread ID: " << i << "\t";
		for (int j = 0; j < varsBatch[i].size(); j++)
		{
			ss << varsBatch[i][j] << "\t";
		}
		LOG(INFO) << ss.str();
	}
	auto it = varsBatch.begin();
	while (it != varsBatch.end())
	{
		auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
		if (itzero != (*it).end())
		{
			it = varsBatch.erase(it);
		}
		else
		{
			++it;
		}
	}

	variances.resize(nkernel + 1);
	for (int i = 0; i < nkernel + 1; i++)
	{
		float sum = 0;
		for (int j = 0; j < varsBatch.size(); j++)
		{
			sum += varsBatch[j][i];
		}
		variances[i] = sum / float(varsBatch.size());
	}
	if (!nofix)
	{
		auto it = fixsBatch.begin();
		while (it != fixsBatch.end())
		{
			auto itzero = std::find_if((*it).cbegin(), (*it).cend(), [](float i) {return i == -999; });
			if (itzero != (*it).end())
			{
				it = fixsBatch.erase(it);
			}
			else
			{
				++it;
			}
		}
		coefs.resize(fixsBatch[0].size());
		for (int i = 0; i < coefs.size(); i++)
		{
			float sum = 0;
			for (int j = 0; j < fixsBatch.size(); j++)
			{
				sum += fixsBatch[j][i];
			}
			coefs[i] = sum / float(fixsBatch.size());
		}
	}

}

void cMINQUE0(MinqueOptions& minque, std::vector<Eigen::MatrixXf*>& Kernels, PhenoData& phe, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs)
{
	MINQUE0 varest(minque.MatrixDecomposition, minque.altMatrixDecomposition, minque.allowPseudoInverse);
	Eigen::VectorXf pheV;
	if (phe.Phenotype.cols() == 1)
	{
		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
	}
	else
	{
		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
	}
	varest.importY(pheV);
	varest.pushback_X(Covs, false);
	for (int i = 0; i < Kernels.size(); i++)
	{
		varest.pushback_Vi(Kernels[i]);
	}
	Eigen::MatrixXf e(phe.Phenotype.size(), phe.Phenotype.size());
	e.setIdentity();
	varest.pushback_Vi(&e);
	std::cout << "starting CPU MINQUE(0) " << std::endl;
	LOG(INFO) << "starting CPU MINQUE(0) ";
	varest.estimateVCs();
	variances = varest.getvcs();
	if (coefs[0] != -999)
	{
		varest.estimateFix();
		coefs = varest.getfix();
	}
	std::stringstream ss;
	ss << std::fixed << "Thread ID: 0" << std::setprecision(3) << "\tIt: " << 0 << "\t" << varest.getvcs().transpose();
	printf("%s\n", ss.str().c_str());
	LOG(INFO) << ss.str();
}