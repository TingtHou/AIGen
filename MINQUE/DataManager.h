#pragma once
#include "cLog.h"
#include "CommonFunc.h"
#include <boost/program_options.hpp>
class DataManager
{
public:
	DataManager(boost::program_options::variables_map programOptions, LOG *log);
	void read();
	PhenoData getPhenotype() {	return phe;	};
	std::vector<KernelData> GetKernel() { return KernelList; };
	~DataManager();
private:
	PhenoData phe;
	std::vector<KernelData> KernelList;
	LOG *log;
	boost::program_options::variables_map programOptions;
private:
	
	void readResponse(std::string resopnsefile, PhenoData &phe);
	void readmkernel(std::string mkernel);
	void match(PhenoData &phenotype, KernelData &kernel, std::string prefix);
};

