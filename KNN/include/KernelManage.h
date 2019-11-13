#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <map>
#include "../include/KernelManage.h"
#include <iomanip>
#include "../include/CommonFunc.h"
class KernelReader
{
public:
	KernelReader(std::string prefix);
	~KernelReader();
	KernelData getKernel() { return Kernels; };
	std::string print();
	void read();

private:
	std::string BinFileName; //*.grm.bin
	std::string NfileName;   //*.grm.N.bin
	std::string IDfileName;  //*.grm.id
	std::string prefix;
	KernelData Kernels; //store kernel data
	int nind=0;
	void IDfileReader(std::ifstream &fin, KernelData &kdata);   // read *.grm.id
	void BinFileReader(std::ifstream &fin, KernelData &kdata);  // read *.grm.bin
	void NfileReader(std::ifstream &fin, KernelData &kdata);    //read *.grm.N.bin
	
	
};

class KernelWriter
{
public:
	KernelWriter(KernelData kdata);
	~KernelWriter();
	std::string print();
	void setprecision(int datatype);
	void setprefix(std::string prefix);
	void writeText(std::string filename);
	void write(std::string prefix);
	void write();
private:
	std::string BinFileName; //*.grm.bin
	std::string NfileName;   //*.grm.N.bin
	std::string IDfileName;  //*.grm.id
	std::string prefix;
	KernelData Kernels; //store kernel data
	int precision =1 ;  //0 for double; 1 for float;
	int nind=0;
private:
	void IDfileWriter(std::ofstream &fin, KernelData &kdata);   // read *.grm.id
	void BinFileWriter(std::ofstream &fin, KernelData &kdata);  // read *.grm.bin
	void NfileWriter(std::ofstream &fin, KernelData &kdata);    //read *.grm.N.bin

};