#pragma once
#include "CommonFunc.h"
#include <boost/program_options.hpp>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include "KernelManage.h"
#include "PlinkReader.h"
#include "CommonFunc.h"
#include "easylogging++.h"
class DataManager
{
public:
	DataManager();
//	void read();
	void readPhe(std::string phefilename);
	void readKernel(std::string prefix);
	void readmKernel(std::string mkernelfilename);
	void readGeno(std::vector<std::string> filelist, bool isImpute);                 //[ped, map] or [bed, bim, fam];
	PhenoData getPhenotype() {	return phe;	};
	std::vector<KernelData> GetKernel() { return KernelList; };
	void SetKernel(std::vector<KernelData> KernelList) { this->KernelList=KernelList; };
	void match();
	GenoData getGenotype() { return geno; };
	~DataManager();
private:
	PhenoData phe;
	GenoData geno;
	std::vector<KernelData> KernelList;
private:
	void readResponse(std::string resopnsefile, PhenoData &phe);
	void readmkernel(std::string mkernel);
	//void match(PhenoData &phenotype, KernelData &kernel, std::string prefix);
	void match(PhenoData &phenotype, KernelData &kernel);
};

