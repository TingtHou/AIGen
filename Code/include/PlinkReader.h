#pragma once
#define EIGEN_USE_MKL_ALL
#include <bitset>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <Eigen/Dense>
#include "ToolKit.h"
#include <boost/algorithm/string.hpp>
#include <boost/exception/all.hpp>
#include <exception>
#include <vector>
#include "CommonFunc.h"
#include "Random.h"
#include <iomanip>
class PlinkReader
{

public:
	PlinkReader();
// 	PlinkReader(std::string pedfile, std::string mapfile);
// 	PlinkReader(std::string bedfile, std::string bimfile, std::string famfile);
	~PlinkReader();

	void savePedMap(std::string prefix);
	GenoData GetGeno();
	void test();
	void setImpute(bool isImpute);
	void read(std::string pedfile, std::string mapfile);
	void read(std::string bedfile, std::string bimfile, std::string famfile);
private:
	static const int READER_MASK = 3;
	static const int HOMOZYGOTE_FIRST = 0;
	static const int HOMOZYGOTE_SECOND = 3;
	static const int HETEROZYGOTE = 2;
	static const int MISSING = 1;
	const int MISSINGcode = -9;
	bool isImpute;
private:
	std::string bimfile;
	std::string famfile;
	std::string bedfile;
	std::string pedfile;
	std::string mapfile;
	///////////////////family info////////////////////////////////
	boost::bimap<int, std::string>  fid_iid_index;
	std::vector<std::string> fid;
	std::vector<std::string> iid;
	std::vector<std::string> PaternalID;
	std::vector<std::string> MaternalID;
	std::multimap<std::string, std::string> fid_iid; //pair<std::string, int>(_fid[i],_iid[i])
	std::vector<int> Sex; //(1 = male; 2 = female; -9 = unknown)
	int Sexstat[2];          //male 1 numder in sex[0], female 2 number in sex[1];
	std::vector <int> Phenotype;
	int nind=0;
	///////////////////////Marker info//////////////////////////////////////
	std::vector<int> chr;					    //chromosome 
	std::vector<std::string> marker_name;               //rs# or snp identifier
	std::vector<int> Gene_distance;             //Genetic distance(morgans)
	std::vector<int> bp;                        //Base - pair position(bp units)
	std::vector<std::string> minor_allele;					//corresponding to clear bits in .bed; usually minor
	std::vector<std::string> major_allele;					//corresponding to set bits in .bed; usually major
	std::multimap<int, std::string> chr_marker; //pair<int,std::string>(chr[i],marker_name[i])
	std::map<std::string, int> marker_index;            //pair<std::string, int>(marker[i],id)
	std::vector<float> minor_freq;            //pair<std::string, int>(marker[i],id)
//	std::vector<float> major_freq;
	int nmarker = 0;
	std::vector<bool> MissinginSNP;
	////////////////////////Marker Data//////////////////////////////////////////
	std::vector<std::vector<int>> Marker;          //nind x nmarker  
	bool SNP_major = true;
private:
	void buildpedigree();
	void buildgenemap();
	void savepedfile(std::string pedfile);
	void savemapfile(std::string mapfile);
	void readPedfile(std::string pedfile);
	void ReadMapFile(std::string mapfile);
	void ReadFamFile(std::string famfile);
	void ReadBimFile(std::string bimfile);
	void ReadBedFile(std::string bedfile);
	void buildMAF();
	void Impute();
	void removeMissing();
	//map<std::string, int> _snp_name_map;
	//multimap<std::string, std::string> _fid_pid_map; //pair<std::string, int>(_fid[i],_pid[i])
	std::string out;
};

