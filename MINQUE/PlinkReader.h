#pragma once
#include <bitset>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <Eigen/dense>
class PlinkReader
{

public:
	PlinkReader();
	PlinkReader(std::string bedfile, std::string bimfile, std::string famfile);
	~PlinkReader();

	void savePedMap(std::string prefix);
	Eigen::MatrixXd GetGeno();
	void test();
private:
	static const int READER_MASK = 3;
	static const int HOMOZYGOTE_FIRST = 0;
	static const int HOMOZYGOTE_SECOND = 3;
	static const int HETEROZYGOTE = 2;
	static const int MISSING = 1;
	static const int MISSINGcode = -9;

private:
	std::string bimfile;
	std::string famfile;
	std::string bedfile;
	///////////////////family info////////////////////////////////
	std::map<std::string, int> fid_pid_index;
	std::vector<std::string> fid;
	std::vector<std::string> pid;
	std::vector<std::string> PaternalID;
	std::vector<std::string> MaternalID;
	std::multimap<std::string, std::string> fid_pid; //pair<std::string, int>(_fid[i],_pid[i])
	std::vector<int> Sex; //(1 = male; 2 = female; -9 = unknown)
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
	std::vector<double> minor_freq;            //pair<std::string, int>(marker[i],id)
//	std::vector<double> major_freq;
	int nmarker = 0;
	////////////////////////Marker Data//////////////////////////////////////////
	std::vector<std::vector<int>> Marker;          //nind x nmarker  
	bool SNP_major = true;
private:
	void buildpedigree();
	void buildgenemap();
	void savepedfile(std::string pedfile);
	void savemapfile(std::string mapfile);
	void CalcMAF();
	void readPedfile(std::string pedfile);
	void ReadMapFile(std::string mapfile);
	void ReadFamFile(std::string famfile);
	void ReadBimFile(std::string bimfile);
	void ReadBedFile(std::string bedfile);
	//map<std::string, int> _snp_name_map;
	//multimap<std::string, std::string> _fid_pid_map; //pair<std::string, int>(_fid[i],_pid[i])
	std::string out;
};

