#include "pch.h"
#include "cLog.h"


LOG::LOG(std::string logfile)
{
	Logfile.open(logfile, std::ios::out);
	std::time_t currenttime = std::time(0);
	char tAll[255];
	tm Tm;
	localtime_s(&Tm, &currenttime);
	std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S", &Tm);
	Logfile << tAll << std::endl;
	Logfile.flush();
}

LOG::LOG()
{
}

void LOG::setlog(std::string logfile)
{
	Logfile.open(logfile, std::ios::out);
	std::time_t currenttime = std::time(0);
	char tAll[255];
	tm Tm;
	localtime_s(&Tm, &currenttime);
	std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S", &Tm);
	Logfile << tAll << std::endl;
	Logfile.flush();
}

void LOG::write(std::string outline, bool terminateOpt)
{
	std::time_t currenttime = std::time(0);
	char tAll[255];
	tm Tm;
	localtime_s(&Tm, &currenttime);
	std::strftime(tAll, sizeof(tAll), "%Y-%m-%d-%H-%M-%S", &Tm);
	if (terminateOpt)
	{
		std::cout << outline << std::endl;
	}
	Logfile << tAll << " :\t" << outline << std::endl;
	Logfile.flush();
}

LOG::~LOG()
{
	
}

void LOG::close()
{
	if (Logfile.is_open())
	{
		Logfile.close();
	}
	
};