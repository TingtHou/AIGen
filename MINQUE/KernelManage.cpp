#include "pch.h"
#include "KernelManage.h"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <map>
#include "KernelManage.h"


KernelReader::KernelReader(std::string prefix)
{
	this->prefix = prefix;
	BinFileName = prefix + ".grm.bin";
	NfileName = prefix + ".grm.N.bin";
	IDfileName = prefix + ".grm.id";
}

KernelReader::~KernelReader()
{
}

std::string KernelReader::print()
{
	std::stringstream ss;
	ss << "Kernel for " << nind << " individuals are included from [" + prefix + "].";
	return ss.str();
}

void KernelReader::IDfileReader(std::ifstream &fin, KernelData &kdata)
{
	kdata.fid_iid.clear();
	int i = 0;
	while (!fin.eof())
	{
		std::string line;
		getline(fin, line);
		if (fin.fail())
		{
			continue;
		}
		std::vector<std::string> splitline;
		boost::split(splitline, line, boost::is_any_of(" \t"), boost::token_compress_on);
		std::string fid_iid = splitline[0] + "_" + splitline[1];
		std::pair<int, std::string> each(i, fid_iid);
		std::pair<std::string, int> reach(fid_iid,i++);
		kdata.fid_iid.insert(each);
		kdata.rfid_iid.insert(reach);
		nind++;
	}
}

void KernelReader::BinFileReader(std::ifstream &fin, KernelData & kdata)
{
	
	
	const size_t num_elements = nind * (nind + 1) / 2;
	fin.seekg(0, std::ios::end);
	int fileSize = fin.tellg();
	int bytesize = fileSize / num_elements;
	fin.seekg(0, std::ios::beg);
	if (bytesize!=4&&bytesize!=8)
	{
		throw ("Error: the size of the [" + BinFileName + "] file is incomplete?");
	}
	char *f_buf = new char[bytesize];// (char*)malloc(bytesize * sizeof(char));  //set bytesize bits buffer for data, 4bits for float and 8bits for double
	for (int i = 0; i < nind; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			if (!fin.read(f_buf, bytesize)) throw ("Error: the size of the [" + BinFileName + "] file is incomplete?");
// 			double a= bytesize==4?*(float *)f_buf: *(double *)f_buf; 
// 			std::cout << a << std::endl;
			kdata.kernelMatrix(j, i) = kdata.kernelMatrix(i, j) = bytesize == 4 ? *(float *)f_buf : *(double *)f_buf;
		}
	}

}

void KernelReader::NfileReader(std::ifstream &fin, KernelData & kdata)
{
	
	const size_t num_elements = nind * (nind + 1) / 2;
	fin.seekg(0, std::ios::end);
	int fileSize = fin.tellg();
	int bytesize =  fileSize / num_elements;
	bytesize = !bytesize ? fileSize % num_elements : bytesize;
	fin.seekg(0, std::ios::beg);
	if (bytesize != 4 && bytesize != 8)
	{
		throw ("Error: the size of the [" + NfileName + "] file is incomplete?");
	}
	char *f_buf = new char[bytesize];//set bytesize bits buffer for data, 4bits for float and 8bits for double
	//char *f_buf = (char*)malloc(bytesize * sizeof(char));  
	for (int i = 0; i < nind; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			if (!fin.read(f_buf, bytesize)) throw ("Error: the size of the [" + NfileName + "] file is incomplete?");
			kdata.VariantCountMatrix(j, i) = kdata.VariantCountMatrix(i, j) = bytesize == 4 ? *(float *)f_buf : *(double *)f_buf;
		}
	}
	delete [] f_buf;


	
}

void KernelReader::read()
{
	std::ifstream IDifstream(IDfileName, std::ios::in);
	if (!IDifstream.is_open())
	{
		IDifstream.close();
		throw ("Error: can not open the file [" + IDfileName + "] to read.");
	}
	std::ifstream Binifstream(BinFileName, std::ios::binary);
	if (!IDifstream.is_open())
	{
		Binifstream.close();
		throw ("Error: can not open the file [" + BinFileName + "] to read.");
	}
	std::cout << "Reading the IDs from [" + IDfileName + "]." << std::endl;
	IDfileReader(IDifstream, Kernels);
	Kernels.kernelMatrix.resize(nind, nind);
	Kernels.VariantCountMatrix.resize(nind, nind);
	Kernels.kernelMatrix.setZero();
	Kernels.VariantCountMatrix.setZero();
	std::cout << "Reading the kernel matrix from [" + BinFileName + "]." << std::endl;;
	BinFileReader(Binifstream, Kernels);
	std::ifstream Nifstream(NfileName, std::ios::binary);
	if (Nifstream.is_open())
	{
		std::cout << "Reading the number of SNPs for the kernel from [" + NfileName + "]." << std::endl;
		NfileReader(Nifstream, Kernels);
		
	}
	IDifstream.close();
	Binifstream.close();
	Nifstream.close();

}

KernelWriter::KernelWriter(KernelData kdata)
{
	this->Kernels = kdata;
	nind = kdata.fid_iid.size();

}

KernelWriter::~KernelWriter()
{
}

void KernelWriter::setprecision(int datatype)
{
	this->precision = datatype;
}

void KernelWriter::setprefix(std::string prefix)
{
	this->prefix = prefix;
	BinFileName = prefix + ".grm.bin";
	NfileName = prefix + ".grm.N.bin";
	IDfileName = prefix + ".grm.id";
}

void KernelWriter::write()
{
	std::ofstream IDsfstram;
	IDsfstram.open(IDfileName, std::ios::out);
	if (!IDsfstram.is_open())
	{
		IDsfstram.close();
		throw ("Error: can not open the file [" + IDfileName + "] to write.");
	}
	std::ofstream Binofstream;
	Binofstream.open(BinFileName, std::ios::out | std::ios::binary);
	if (!Binofstream.is_open())
	{
		Binofstream.close();
		throw ("Error: can not open the file [" + BinFileName + "] to write.");
	}
	std::ofstream Nofstream;
	Nofstream.open(NfileName, std::ios::out | std::ios::binary);
	if (!Nofstream.is_open())
	{
		Nofstream.close();
		throw ("Error: can not open the file [" + NfileName + "] to write.");
	}
	IDfileWriter(IDsfstram, Kernels);
	BinFileWriter(Binofstream, Kernels);
	NfileWriter(Nofstream, Kernels);
	IDsfstram.close();
	Binofstream.close();
	std::cout << "Kernel of " << nind << " individuals has been saved in the file [" + prefix + "] (in binary format)." << std::endl;
	Nofstream.close();
	std::cout << "Number of SNPs to calcuate the genetic relationship between each pair of individuals has been saved in the file [" + prefix + "] (in binary format)." << std::endl;

}

void KernelWriter::write(std::string prefix)
{
	this->prefix = prefix;
	BinFileName = prefix + ".grm.bin";
	NfileName = prefix + ".grm.N.bin";
	IDfileName = prefix + ".grm.id";
	std::ofstream IDsfstram;
	IDsfstram.open(IDfileName, std::ios::out);
	if (!IDsfstram.is_open())
	{
		IDsfstram.close();
		throw ("Error: can not open the file [" + IDfileName + "] to write.");
	}
	std::ofstream Binofstream;
	Binofstream.open(BinFileName, std::ios::out|std::ios::binary);
	if (!Binofstream.is_open())
	{
		Binofstream.close();
		throw ("Error: can not open the file [" + BinFileName + "] to write.");
	}
	std::ofstream Nofstream;
	Nofstream.open(NfileName, std::ios::out | std::ios::binary);
	if (!Nofstream.is_open())
	{
		Nofstream.close();
		throw ("Error: can not open the file [" + NfileName + "] to write.");
	}
	IDfileWriter(IDsfstram, Kernels);
	BinFileWriter(Binofstream, Kernels);
	NfileWriter(Nofstream, Kernels);
	IDsfstram.close();
	Binofstream.close();
	std::cout << "Kernel of " << nind << " individuals has been saved in the file [" + prefix + "] (in binary format)." << std::endl;
	Nofstream.close();
	std::cout << "Number of SNPs to calcuate the genetic relationship between each pair of individuals has been saved in the file [" + prefix + "] (in binary format)." << std::endl;
}

void KernelWriter::IDfileWriter(std::ofstream & fin, KernelData & kdata)
{
	for (auto it=kdata.fid_iid.begin();it!=kdata.fid_iid.end();it++)
	{
		std::string fid_iid = it->second;
		std::vector<std::string> splitline;
		boost::split(splitline, fid_iid, boost::is_any_of("_"), boost::token_compress_on);
		fin << splitline[0] << "\t" << splitline[1] << std::endl;
		fin.flush();
	}
}

void KernelWriter::BinFileWriter(std::ofstream & fin, KernelData & kdata)
{
	int bytesize = !precision ? 8 : 4;
	for (int i=0;i<nind;i++)
	{
		for (int j=0;j<=i;j++)
		{
			if (precision)
			{
				float f_buf = kdata.kernelMatrix(i, j);
				fin.write((char *) &f_buf, bytesize);
			}
			else
			{
				double f_buf = kdata.kernelMatrix(i, j);
				fin.write((char *)&f_buf, bytesize);
			}
	//		fin.write((char *) &(!precision ? kdata.kernelMatrix(i, j) : (float)kdata.kernelMatrix(i, j), bytesize);
		}
	}
	

}

void KernelWriter::NfileWriter(std::ofstream & fin, KernelData & kdata)
{
	int bytesize = !precision ? 8 : 4;
	char *f_buf = new char[bytesize];
	for (int i = 0; i < nind; i++)
	{
		for (int j = 0; j <= i; j++)
		{

			if (precision)
			{
				float f_buf = kdata.VariantCountMatrix(i, j);
				fin.write((char *)&f_buf, bytesize);
			}
			else
			{
				double f_buf = kdata.VariantCountMatrix(i, j);
				fin.write((char *)&f_buf, bytesize);
			}
		}
	}
	delete[] f_buf;
}
