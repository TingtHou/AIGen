#include "pch.h"
#include "PlinkReader.h"
#include "ToolKit.h"
#include "Exception.h"
#include <boost/algorithm/string.hpp>
#include <boost/exception/all.hpp>
#include <exception>
PlinkReader::PlinkReader(std::string bedfile, std::string bimfile, std::string famfile)
{
	this->bedfile = bedfile;
	this->bimfile = bimfile;
	this->famfile = famfile;

}

PlinkReader::PlinkReader()
{

}

PlinkReader::~PlinkReader()
{

}

void PlinkReader::readPedfile(std::string pedfile)
{
	std::string str_buf;
	std::ifstream Ped(pedfile.c_str());
	if (!Ped)
	{
			BOOST_THROW_EXCEPTION(
				NoItemException() <<
				ErrInfoItemType("ped"));
//		std::cout << "Error: can not open the file [" << pedfile << "] to read!" << std::endl;
	//	return;
	}
	std::cout << "Reading Plink map file from [" + pedfile + "]." << endl;
	for (int i=0;i<nmarker;i++)
	{
		minor_allele.push_back("1");
		major_allele.push_back("2");
	}
	std::getline(Ped, str_buf);
	while (str_buf!="")
	{
		
		std::vector<string> SplitVec;
		boost::split(SplitVec, str_buf, boost::is_any_of(" \t"), boost::token_compress_on);
		fid.push_back(SplitVec[0]);
		pid.push_back(SplitVec[1]);
		PaternalID.push_back(SplitVec[2]);
		MaternalID.push_back(SplitVec[3]);
		int _sex = atoi(SplitVec[4].c_str());
		if (_sex != 1 && _sex != 2)
		{
			Sex.push_back(-9);
		}
		else
		{
			Sex.push_back(_sex);
		}
		int _phe = atoi(SplitVec[5].c_str());
		if (_phe != 1 && _phe != 2)
		{
			Phenotype.push_back(MISSINGcode);
		}
		else
		{
			Phenotype.push_back(_phe);
		}
		std::vector<int> Gene;
		for (int i = 6; i < SplitVec.size(); )
		{

			int allele1= atoi(SplitVec[i++].c_str());
			int allele2 = atoi(SplitVec[i++].c_str());
			int alleles = allele1 + allele2;
			switch (alleles)
			{
			case 2:
				Gene.push_back(0);
				break;
			case 3:
				Gene.push_back(1);
				break;
			case 4:
				Gene.push_back(2);
				break;
			default:
				Gene.push_back(-9);
				break;
			}
		}
		Marker.push_back(Gene);
		std::getline(Ped, str_buf);
		nind++;
	}
	buildpedigree();
	Ped.close();
}


void PlinkReader::ReadMapFile(std::string mapfile)
{
	std::string str_buf;
	std::ifstream Map(mapfile.c_str());
	if (!Map)
	{
		std::cout << "Error: can not open the file [" << mapfile << "] to read!" << std::endl;
		return;
	}
	std::cout << "Reading Plink map file from [" + mapfile + "]." << endl;

	chr.clear();
	marker_name.clear();
	Gene_distance.clear();
	bp.clear();
	minor_allele.clear();
	major_allele.clear();
	chr_marker.clear();
	marker_index.clear();
	while (Map) {
		Map >> str_buf;
		if (Map.eof()) break;
		chr.push_back(stoi(str_buf));
		Map >> str_buf;
		marker_name.push_back(str_buf);
		Map >> str_buf;
		Gene_distance.push_back(stoi(str_buf));
		Map >> str_buf;
		bp.push_back(stoi(str_buf));
		nmarker++;
	}
	Map.close();
	std::cout << nmarker << " SNPs to be included from [" + mapfile + "]." << std::endl;
	buildgenemap();
}

void PlinkReader::ReadFamFile(std::string famfile)
{
	std::ifstream Fam(famfile.c_str());
	if (!Fam)
	{
		std::cout << "Error: can not open the file [" << famfile << "] to read." << std::endl;
		return;
	}
	std::cout << "Reading plink fam file from [" << famfile << "]." << endl;
	fid_pid_index.clear();
	PaternalID.clear();
	MaternalID.clear();
	fid_pid.clear();
	Sex.clear();
	Phenotype.clear();
	pid.clear();
	fid_pid.clear();
	int i = 0;
	std::string str_buf;
	nind = 0;
	while (Fam)
	{
		Fam >> str_buf;
		if (Fam.eof()) break;
		fid.push_back(str_buf);
		Fam >> str_buf;
		pid.push_back(str_buf);
		Fam >> str_buf;
		PaternalID.push_back(str_buf);
		Fam >> str_buf;
		MaternalID.push_back(str_buf);
		Fam >> str_buf;
		int _sex = atoi(str_buf.c_str());
		if (_sex!=1&&_sex!=2)
		{
			Sex.push_back(-9);
		}
		else
		{
			Sex.push_back(_sex);
		}
		
		Fam >> str_buf;
		int _phe = atoi(str_buf.c_str());
		if (_phe != 1 && _phe != 2)
		{
			Phenotype.push_back(MISSINGcode);
		}
		else
		{
			Phenotype.push_back(_phe);
		}
//		Phenotype.push_back(stod(str_buf));
		nind++;
	}

}

void PlinkReader::ReadBimFile(std::string bimfile)
{

	std::string str_buf;
	std::ifstream Bim(bimfile.c_str());
	if (!Bim)
	{
		std::cout << "Error: can not open the file [" << bimfile << "] to read!" << std::endl;
		return;
	}
	std::cout << "Reading Plink bim file from [" + bimfile + "]." << endl;

	chr.clear();
	marker_name.clear();
	Gene_distance.clear();
	bp.clear();
	minor_allele.clear();
	major_allele.clear();
	chr_marker.clear();
	marker_index.clear();
	while (Bim) {
		Bim >> str_buf;
		if (Bim.eof()) break;
		chr.push_back(stoi(str_buf));
		Bim >> str_buf;
		marker_name.push_back(str_buf);
		Bim >> str_buf;
		Gene_distance.push_back(stoi(str_buf));
		Bim >> str_buf;
		bp.push_back(stoi(str_buf));
		Bim >> str_buf;
		//StrFunc::to_upper(cbuf);
		minor_allele.push_back(str_buf);
		Bim >> str_buf;
		//StrFunc::to_upper(cbuf);
		major_allele.push_back(str_buf);
		nmarker++;
	}
	Bim.close();
	std::cout << nmarker << " SNPs to be included from [" + bimfile + "]." << std::endl;
	buildgenemap();
}

void PlinkReader::ReadBedFile(std::string bedfile)
{
	std::fstream Bed(bedfile.c_str(), std::ios::in | std::ios::binary);
	if (!Bed)
	{
		std::cout << "Error: can not open the file [" << bedfile << "] to read." << std::endl;
		return;
	}
	char byte_buf;
	for (int i=0;i<2;i++)
	{
		Bed.read(&byte_buf, 1);
	}
	Bed.read(&byte_buf, 1);
	if (byte_buf == 1)
	{
		SNP_major = true;
	}
	else if (byte_buf == 0)
	{
		SNP_major = false;
	}
	else
	{
		std::cout << "Error: bed file [" << bedfile << "] is neither in SNP-major mode nor in individual-major mode." << std::endl;
		return;
	}
	std::cout << "Reading PLINK BED file from [" + bedfile + "] in "<< (SNP_major? "SNP-major format ...": "individual-major format ...") << std::endl;
	int sampleCounter = 0;
	std::vector<int> Gene;
	Gene.clear();
	std::vector < std::vector<int>> _marker;
	_marker.clear();
	int len = SNP_major ? nind : nmarker;
	while (Bed)
	{
		Bed.read(&byte_buf, 1);
		for (int i = 0; i < 4; ++i)
		{

			if (sampleCounter < len)
			{
				switch (byte_buf & READER_MASK)
				{
				case HOMOZYGOTE_FIRST:
					Gene.push_back(0); //1 1
					break;
				case HOMOZYGOTE_SECOND:
					Gene.push_back(2); //2 2
					break;
				case HETEROZYGOTE:
					Gene.push_back(1); //1 2
					break;
				case MISSING:
					Gene.push_back(-9); // 0 0
					break;
				default:
					return;
				}
			}
			byte_buf = byte_buf >>  2;
			++sampleCounter;
		}
		if (sampleCounter == len)
		{
			_marker.push_back(Gene);
			Gene.clear();
			sampleCounter = 0;
		}
	}
	if (SNP_major)
	{
		Marker.resize(nind);
		for (int i=0;i<nind;i++)
		{
			Marker[i].resize(nmarker);
			for (int j=0;j<nmarker;j++)
			{
				Marker[i][j] = _marker[j][i];
			}
		}
	}
	else
	{
		Marker = _marker;
	}
	Bed.close();
	std::cout << nind << " individuals and " << nmarker << " SNPs to be included from [" + bedfile + "]." << endl;
}



void PlinkReader::test()
{
	std::string bedfile = "../example/example.bed";
	std::string bimfile = "../example/example.bim";
	std::string famfile = "../example/example.fam";
	std::string pedfile = "../example/example11.ped";
	std::string mapfile = "../example/example.map";
//	std::string famfile = "../example/example.fam";
	this->bedfile = bedfile;
// 	ReadBimFile(bimfile);
// 	ReadFamFile(famfile);
// 	ReadBedFile(bedfile);
	ReadMapFile(mapfile);
	readPedfile(pedfile);
	savePedMap();
}

void PlinkReader::buildpedigree()
{
	for (int i=0;i<nind;i++)
	{
		fid_pid_index.insert(std::pair<std::string, int>(fid[i] + ":" + pid[i], i));
		fid_pid.insert(std::pair<std::string, std::string>(fid[i], pid[i]));
	}
}

void PlinkReader::buildgenemap()
{
	for (int i=0;i<nmarker;i++)
	{
		chr_marker.insert(std::pair<int, std::string>(chr[i], marker_name[i]));
		marker_index.insert(std::pair<std::string, int>(marker_name[i], i));
	}
}

void PlinkReader::savePedMap()
{
	std::string basename = bedfile.substr(0, bedfile.rfind("."));
	std::string pedfile = basename + "1.ped";
	std::string mapfile = basename + "1.map";
	savemapfile(mapfile);
	savepedfile(pedfile);
}

void PlinkReader::savepedfile(std::string pedfile)
{
	std::ofstream Ped(pedfile.c_str());
	if (!Ped)
	{
		std::cout << "Error: can not open the file [" << pedfile << "] to write." << std::endl;
		return;
	}
	std::cout << "Writing PLINK ped file to [" + pedfile + "] ..." << std::endl;
	for (int i=0;i<nind;i++)
	{
		Ped << fid[i] << "\t" << pid[i] << "\t" << PaternalID[i] << "\t" << MaternalID[i] << "\t" << Sex[i] << "\t" << Phenotype[i];
		for (int j=0;j<nmarker;j++)
		{
			int alleles = (int)Marker[i][j];
			std::string str_alleles;
			switch (alleles)
			{
			case 0:
				str_alleles = minor_allele[j] + " " + minor_allele[j];
				break;
			case 1:
				str_alleles = minor_allele[j] + " " + major_allele[j];
				break;
			case 2:
				str_alleles = major_allele[j] + " " + major_allele[j];
				break;
			default:
				str_alleles = "0 0";
				break;
			}
			Ped <<"\t"<<str_alleles;
		
		}
		Ped << std::endl;
		Ped.flush();
	}
	Ped.close();
	std::cout <<"Writing "<< nind << " individuals and " << nmarker << " markers to [" + pedfile + "]." << endl;
}

void PlinkReader::savemapfile(std::string mapfile)
{
	std::ofstream Map(mapfile.c_str());
	if (!Map)
	{
		std::cout << "Error: can not open the file [" << mapfile << "] to write." << std::endl;
		return;
	}
	std::cout << "Writing PLINK map file to [" + mapfile + "] ..." << std::endl;
	for (int i=0;i<nmarker;i++)
	{
		Map << chr[i] << "\t" << marker_name[i] << "\t" << Gene_distance[i] << "\t" << bp[i] << std::endl;
		Map.flush();
	}
	Map.close();
	std::cout << "Writing "<<nmarker << " markers to [" + mapfile + "]." << endl;
}

void PlinkReader::CalcMAF()
{
}

