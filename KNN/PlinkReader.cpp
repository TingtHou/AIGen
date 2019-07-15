#include "pch.h"
#include "PlinkReader.h"
PlinkReader::PlinkReader(std::string pedfile, std::string mapfile)
{
	this->pedfile = pedfile;
	this->mapfile = mapfile;
	ReadMapFile(mapfile);
	readPedfile(pedfile);
	buildpedigree();
	buildgenemap();
}
// reading bed file, bim file, famfile in plink format
PlinkReader::PlinkReader(std::string bedfile, std::string bimfile, std::string famfile)
{
	this->bedfile = bedfile;
	this->bimfile = bimfile;
	this->famfile = famfile;
	Sexstat[0] = 0;
	Sexstat[1] = 0;
	ReadBimFile(bimfile);
	ReadFamFile(famfile);
	ReadBedFile(bedfile);
	buildpedigree();
	buildgenemap();
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
		throw ("Error: can not open the file [" + pedfile + "] to read!" );
	}
	std::cout << "Reading genotype file from [" + pedfile + "]." << endl;
	for (int i=0;i<nmarker;i++)
	{
		minor_allele.push_back("1");
		major_allele.push_back("2");
	}
	std::getline(Ped, str_buf);
	int indtmp = 0;
	while (str_buf!="")
	{
		
		std::vector<string> SplitVec;
		boost::split(SplitVec, str_buf, boost::is_any_of(" \t"), boost::token_compress_on);
		fid.push_back(SplitVec[0]);
		iid.push_back(SplitVec[1]);
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
				Gene.push_back(2);
				break;
			case 3:
				Gene.push_back(1);
				break;
			case 4:
				Gene.push_back(0);
				break;
			default:
				Gene.push_back(3);
				break;
			}
		}
		Marker.push_back(Gene);
		std::getline(Ped, str_buf);
		nind++;
	}
	Ped.close();
	std::cout << nind << " individuals read from [" << pedfile << "]. " << std::endl;
	if (nind != std::count(Phenotype.begin(), Phenotype.end(), -9))
	{
		std::cout << nind << " individuals with nonmissing phenotypes." << std::endl;
		std::cout << std::count(Sex.begin(), Sex.end(), 1) << " males, " << std::count(Sex.begin(), Sex.end(), 2) << " females, and " << std::count(Sex.begin(), Sex.end(), -9) << " of unspecified sex." << std::endl;

	}
}


void PlinkReader::ReadMapFile(std::string mapfile)
{
	std::string str_buf;
	std::ifstream Map(mapfile.c_str());
	if (!Map)
	{
		throw ("Error: can not open the file [" + mapfile + "] to read!");
		
	}
	std::cout << "Reading map file from [" + mapfile + "]." << endl;

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
	std::cout << nmarker << " markers to be included from [" + mapfile + "]." << std::endl;
}

void PlinkReader::ReadFamFile(std::string famfile)
{
	std::ifstream Fam(famfile.c_str());
	if (!Fam)
	{
		throw ("Error: can not open the file [" + famfile + "] to read.");
	}
	std::cout << "Reading pedigree information from [" << famfile << "]." << endl;
	fid_iid_index.clear();
	PaternalID.clear();
	MaternalID.clear();
	fid_iid.clear();
	Sex.clear();
	Phenotype.clear();
	iid.clear();
	fid_iid.clear();
	int i = 0;
	std::string str_buf;
	nind = 0;
	while (Fam)
	{
		std::string fid_iid;
		Fam >> str_buf;
		if (Fam.eof()) break;
		fid.push_back(str_buf);
		Fam >> str_buf;
		iid.push_back(str_buf);
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
			Sexstat[_sex - 1]++;
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
		nind++;
	}
	Fam.close();
//	std::cout << nind << " individuals (" << Sexstat[0] << " males, " << Sexstat[1] << " females) loaded from .fam." << std::endl;
	std::cout << nind << " individuals read from ["<<famfile<<"]. "<< std::endl;
	if (nind != std::count(Phenotype.begin(), Phenotype.end(), -9))
	{
		std::cout << nind << "individuals with nonmissing phenotypes." << std::endl;
		std::cout << std::count(Sex.begin(), Sex.end(), 1) << " males, " << std::count(Sex.begin(), Sex.end(), 2) << " females, and " << std::count(Sex.begin(), Sex.end(), -9) << " of unspecified sex." << std::endl;
		
	}
}

void PlinkReader::ReadBimFile(std::string bimfile)
{

	std::string str_buf;
	std::ifstream Bim(bimfile.c_str());
	if (!Bim)
	{
		Bim.close();
		throw ("Error: can not open the file [" + bimfile + "] to read!");
	}
	std::cout << "Reading map (extended format) from [" + bimfile + "]." << endl;
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
}

void PlinkReader::ReadBedFile(std::string bedfile)
{
	std::fstream Bed(bedfile.c_str(), std::ios::in | std::ios::binary);
	if (!Bed)
	{
		Bed.close();
		throw ("Error: can not open the file [" + bedfile + "] to read.");
	}
	std::cout << "Reading genotype bitfile from [" + bedfile + "]." << endl;
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
		throw ("Error: bed file [" + bedfile + "] is neither in SNP-major mode nor in individual-major mode.");
	}
	std::cout << "Detected that binary PED file is v1.00 SNP - major mode" << std::endl;
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
					Gene.push_back(2); //1 1
					break;
				case HOMOZYGOTE_SECOND:
					Gene.push_back(0); //2 2
					break;
				case HETEROZYGOTE:
					Gene.push_back(1); //1 2
					break;
				case MISSING:
					Gene.push_back(3); // 0 0
					break;
				default:
					return;
				}
			}
			byte_buf = byte_buf >>  2;
			++sampleCounter;
			if (sampleCounter == len)
			{
				_marker.push_back(Gene);
				Gene.clear();
				sampleCounter = 0;
				break;
			}
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
	std::cout << "Before frequency and genotyping pruning, there are "<<nmarker<< " SNPs." << endl;
}

GenoData PlinkReader::GetGeno()
{
	GenoData gd;
	gd.Geno.resize(nind,nmarker);
	for (int i=0;i<nind;i++)
	{
		for (int j=0;j<nmarker;j++)
		{
			gd.Geno(i, j) = Marker[i][j];
		}
	}
	gd.fid_iid = fid_iid_index;
	return gd;
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
	//savePedMap();
}

void PlinkReader::buildpedigree()
{
	for (int i=0;i<nind;i++)
	{
		fid_iid_index.insert({ i,fid[i] + "_" + iid[i] });
		fid_iid.insert(std::pair<std::string, std::string>(fid[i], iid[i]));
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

void PlinkReader::savePedMap(std::string prefix)
{
	
//	std::string basename = bedfile.substr(0, bedfile.rfind("."));
	std::string pedfile = prefix + ".ped";
	std::string mapfile = prefix + ".map";
	savemapfile(mapfile);
	savepedfile(pedfile);
}

void PlinkReader::savepedfile(std::string pedfile)
{
	std::ofstream Ped(pedfile.c_str());
	if (!Ped)
	{
		Ped.close();
		throw ("Error: can not open the file [" + pedfile + "] to write.");
	}
	std::cout << "Writing PLINK ped file to [" + pedfile + "] ..." << std::endl;
	for (int i=0;i<nind;i++)
	{
		Ped << fid[i] << "\t" << iid[i] << "\t" << PaternalID[i] << "\t" << MaternalID[i] << "\t" << Sex[i] << "\t" << Phenotype[i];
		for (int j=0;j<nmarker;j++)
		{
			int alleles = (int)Marker[i][j];
			std::string str_alleles;
			switch (alleles)
			{
			case 0:
				str_alleles = "2 2";
				break;
			case 1:
				str_alleles = "1 2";
				break;
			case 2:
				str_alleles = "1 1";
				break;
			default:
				str_alleles = "0 0";
				break;
			}

// 			switch (alleles)
// 			{
// 			case 0:
// 				str_alleles = minor_allele[j] + " " + minor_allele[j];
// 				break;
// 			case 1:
// 				str_alleles = minor_allele[j] + " " + major_allele[j];
// 				break;
// 			case 2:
// 				str_alleles = major_allele[j] + " " + major_allele[j];
// 				break;
// 			default:
// 				str_alleles = "0 0";
// 				break;
// 			}
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
		Map.close();
		throw ("Error: can not open the file [" + mapfile + "] to write.");
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

