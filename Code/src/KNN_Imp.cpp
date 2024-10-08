#include "../include/KNN_Imp.h"

void BatchMINQUE1(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, 
	Eigen::VectorXf& coefs, float& iterateTimes, int nsplit, int seed, int nthread, bool isecho, bool dataType)
{
	int nkernel = Kernels.size();
	//bool nofix = coefs[0] == -999 ? true : false;
//	Eigen::VectorXf pheV;
//	if (phe.Phenotype.cols() == 1)
//	{
//		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
//	}
//	else
//	{
//		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
//	}
	/*
	for (size_t i = 0; i < nkernel; i++)
	{
		std::stringstream ss;
		ss << "kernel: " << i << "in BatchMINQUE1" << std::endl;
		ss << "First 10x10: \n" << Kernels[i]->block(0, 0, 10, 10) << std::endl;
		ss << "Last 10x10: \n" << Kernels[i]->block(Kernels[i]->rows() - 10, Kernels[i]->cols() - 10, 10, 10);
		LOG(INFO) << ss.str() << std::endl;
	}*/
	
	std::stringstream ss_batch;
	ss_batch << "Generating " << nsplit << " batches for analysis";
	std::cout << ss_batch.str() << std::endl;
	LOG(INFO) << ss_batch.str();
	Batch b = Batch(Kernels, pheV, Covs, nsplit, seed, true);
	b.start(dataType);
	std::cout << "Generated." << std::endl;
	LOG(INFO) << "Generated batch, sizes of each batch are: " << b.getSizesofBatch().transpose() << ".";
	std::vector<std::vector<std::shared_ptr<Eigen::MatrixXf>>> KernelsBatch;
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
		std::shared_ptr<imnq> varest=std::make_shared<imnq>(minque);
		varest->isEcho(isecho);
		varest->setThreadId(i);
	//	varest.setOptions(minque);
		varest->importY(PheBatch[i]);
		varest->pushback_X(CovBatch[i], false);

		std::shared_ptr<Eigen::MatrixXf> e=std::make_shared<Eigen::MatrixXf>(PheBatch[i].size(), PheBatch[i].size());
		e->setIdentity();
		std::stringstream ss_bug;
		for (int j = 0; j < KernelsBatch[i].size(); j++)
		{
			
			ss_bug << "10 *10 Kernel "<< j <<"on thread " << i<<"\n"<< KernelsBatch[i][j]->block(0, 0, 10, 10) << std::endl;
			ss_bug << "Last 10 *10 Kernel " << j << "on thread " << i << "\n" << KernelsBatch[i][j]->block(KernelsBatch[i][j]->rows() - 10, KernelsBatch[i][j]->cols() - 10, 10, 10) << std::endl;
			varest->pushback_Vi(KernelsBatch[i][j]);
		}
		LOG(INFO) << ss_bug.str() << std::endl;

		varest->pushback_Vi(e);
		if (variances.size() != 0)
		{
			varest->pushback_W(variances);
		}
		try
		{
			varest->estimateVCs();
			varsBatch[i].resize(varest->getvcs().size());
			Eigen::VectorXf::Map(&varsBatch[i][0], varest->getvcs().size()) = varest->getvcs();
		/*	if (!nofix)
			{
				varest.estimateFix();
				fixsBatch[i].resize(varest.getfix().size());
				Eigen::VectorXf::Map(&fixsBatch[i][0], varest.getfix().size()) = varest.getfix();
			}
			*/
			printf("The thread %d is finished\n", i);
			LOG(INFO) << "The thread " << i << " is finished";
		}
		catch (const std::exception & err)
		{
			std::stringstream ss;
			ss << "[Warning]: The thread " << i << " is interrupt, because " << err.what();
			printf("%s\n", ss.str().c_str());
			LOG(WARNING) << ss.str();
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
		}
		catch (std::string & e)
		{
			std::cout << e;
			LOG(ERROR) << e;
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
			/*if (!nofix)
			{
				fixsBatch[i].resize(1);
				fixsBatch[i][0] = -999;
			}
			*/
		}
	}

	for (int i = 0; i < varsBatch.size(); i++)
	{
		std::stringstream ss;
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
	/*
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
	*/

	iterateTimes = accumulate(time.begin(), time.end(), 0.0) / time.size(); ;
}

std::shared_ptr<MinqueBase> cMINQUE1(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho)
{
	std::shared_ptr<imnq> varest=std::make_shared<imnq>(minque) ;
	//varest->setOptions(minque);
	varest->isEcho(isecho);
//	Eigen::VectorXf pheV;
//	if (phe.Phenotype.cols() == 1)
//	{
//		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
//	}
//	else
//	{
//		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
//	}
	varest->importY(pheV);
	if (Covs.size() > 0)
	{
		varest->pushback_X(Covs, false);
	}
	
	for (int i = 0; i < Kernels.size(); i++)
	{
		varest->pushback_Vi(Kernels[i]);
	}
	
	std::shared_ptr<Eigen::MatrixXf> e = std::make_shared<Eigen::MatrixXf>(pheV.size(), pheV.size());
	e->setIdentity();
	varest->pushback_Vi(e);
	if (variances.size() != 0)
	{
		varest->pushback_W(variances);
	}
	std::cout << "starting CPU MINQUE(1) " << std::endl;
	varest->estimateVCs();
	variances = varest->getvcs();
	iterateTimes = varest->getIterateTimes();
	/*if (coefs[0] != -999)
	{
		varest.estimateFix();
		coefs = varest.getfix();
	}
	*/
	return varest;
}

void Fixed_estimator(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, float& iterateTimes, bool isecho)
{
	std::shared_ptr<imnq> varest=std::make_shared<imnq>(minque);
//	varest.setOptions(minque);
	varest->isEcho(isecho);
//	Eigen::VectorXf pheV;
//	if (phe.Phenotype.cols() == 1)
//	{
//		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
//	}
//	else
//	{
//		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
//	}
	LOG(INFO) << "Input Y";
	varest->importY(pheV);
	LOG(INFO) << "Input X";
	varest->pushback_X(Covs, false);
	for (int i = 0; i < Kernels.size(); i++)
	{
		LOG(INFO) << "Input Kernel " << i << std::endl;
		varest->pushback_Vi(Kernels[i]);
	}
	std::shared_ptr<Eigen::MatrixXf> e = std::make_shared<Eigen::MatrixXf>(pheV.size(), pheV.size());
	e->setIdentity();
	LOG(INFO) << "Input Kernel error"  << std::endl;
	varest->pushback_Vi(e);
	if (variances.size() != 0)
	{
		varest->pushback_W(variances);
	}
	LOG(INFO) << "estimation starting" << std::endl;
	varest->estimateFix(variances);
	LOG(INFO) << "Get Fixed effect" << std::endl;
	coefs = varest->getfix();
}

void BatchMINQUE0(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, 
	Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs, int nsplit, int seed, int nthread, bool dataType)
{
	int nkernel = Kernels.size();
//	bool nofix = coefs[0] == -999 ? true : false;
//	Eigen::VectorXf pheV;
//	if (phe.Phenotype.cols() == 1)
//	{
//		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
//	}
//	else
//	{
//		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
//	}
	std::stringstream ss_batch;
	ss_batch << "Generating " << nsplit << " batches for analysis";
	std::cout << ss_batch.str() << std::endl;
	LOG(INFO) << ss_batch.str();
	/*
	for (size_t i = 0; i < nkernel; i++)
	{
		std::stringstream ss;
		ss << "kernel: " << i << "in BatchMINQUE1" << std::endl;
		ss << "First 10x10: \n" << Kernels[i]->block(0, 0, 10, 10) << std::endl;
		ss << "Last 10x10: \n" << Kernels[i]->block(Kernels[i]->rows() - 10, Kernels[i]->cols() - 10, 10, 10);
		LOG(INFO) << ss.str() << std::endl;
	}*/
	Batch b = Batch(Kernels, pheV, Covs, nsplit, seed, true);
	b.start(dataType);
	std::cout << "Generated." << std::endl;
	LOG(INFO) << "Generated batch, sizes of each batch are: " << b.getSizesofBatch().transpose() << ".";
	std::vector<std::vector<std::shared_ptr<Eigen::MatrixXf>>> KernelsBatch;
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
		std::shared_ptr<Eigen::MatrixXf> e = std::make_shared<Eigen::MatrixXf>(PheBatch[i].size(), PheBatch[i].size());
		e->setIdentity();
		for (int j = 0; j < KernelsBatch[i].size(); j++)
		{
			varest.pushback_Vi(KernelsBatch[i][j]);
		}
		varest.pushback_Vi(e);
		try
		{
			varest.estimateVCs();
			varsBatch[i].resize(varest.getvcs().size());
			Eigen::VectorXf::Map(&varsBatch[i][0], varest.getvcs().size()) = varest.getvcs();
		/*	if (!nofix)
			{
				varest.estimateFix();
				fixsBatch[i].resize(varest.getfix().size());
				Eigen::VectorXf::Map(&fixsBatch[i][0], varest.getfix().size()) = varest.getfix();
			}
			*/
			printf("The thread %d is finished\n", i);
			LOG(INFO) << "The thread " << i << " is finished";
		}
		catch (const std::exception & err)
		{
			std::stringstream ss;
			ss << "[Warning]: The thread " << i << " is interrupt, because " << err.what();
			std::cout << ss.str();
			//	printf("%s\n", ss.str().);
			LOG(WARNING) << ss.str();
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
		//	if (!nofix)
		//	{
		//		fixsBatch[i].resize(1);
		//		fixsBatch[i][0] = -999;
		//	}
		}
		catch (std::string & e)
		{
			std::cout << e;
			LOG(ERROR) << e;
			varsBatch[i].resize(nkernel + 1);
			varsBatch[i][0] = -999;
		//	if (!nofix)
		//	{
		//		fixsBatch[i].resize(1);
		//		fixsBatch[i][0] = -999;
		//	}
		}
	}
	for (int i = 0; i < varsBatch.size(); i++)
	{
		std::stringstream ss;
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
	/*
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
	*/

}

std::shared_ptr<MinqueBase> cMINQUE0(MinqueOptions& minque, std::vector<std::shared_ptr<Eigen::MatrixXf>> Kernels, Eigen::VectorXf& pheV, Eigen::MatrixXf& Covs, Eigen::VectorXf& variances, Eigen::VectorXf& coefs)
{
	std::shared_ptr< MinqueBase> varest =std::make_shared< MINQUE0>(minque.MatrixDecomposition, minque.altMatrixDecomposition, minque.allowPseudoInverse);
//	Eigen::VectorXf pheV;
//	if (phe.Phenotype.cols() == 1)
//	{
//		pheV = Eigen::Map<Eigen::VectorXf>(phe.Phenotype.data(), phe.Phenotype.rows());
//	}
//	else
//	{
//		throw std::string("KNN can not be applied to the phenotype over 2 dimensions.");
//	}
	varest->importY(pheV);
	if (Covs.size() > 0)
	{
		varest->pushback_X(Covs, false);
	}
	for (int i = 0; i < Kernels.size(); i++)
	{
		varest->pushback_Vi(Kernels[i]);
	}
	std::shared_ptr<Eigen::MatrixXf> e = std::make_shared<Eigen::MatrixXf>(pheV.size(), pheV.size());
	e->setIdentity();
	varest->pushback_Vi(e);
	std::cout << "starting CPU MINQUE(0) " << std::endl;
	LOG(INFO) << "starting CPU MINQUE(0) ";
	varest->estimateVCs();
	variances = varest->getvcs();
/*	if (coefs[0] != -999)
	{
		varest.estimateFix();
		coefs = varest.getfix();
	}*/
	std::stringstream ss;
	ss << std::fixed << "Thread ID: 0" << std::setprecision(3) << "\tIt: " << 0 << "\t" << varest->getvcs().transpose();
	printf("%s\n", ss.str().c_str());
	LOG(INFO) << ss.str();
	return varest;


	

}