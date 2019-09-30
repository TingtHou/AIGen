#pragma once
#include <iostream>
#include <fstream>
#include <time.h>
#include <iomanip>
#include <string>
class LOG
{
public:
	LOG(std::string logfile);
	LOG();
	void setlog(std::string logfile);
	void write(std::string outline, bool terminateOpt);
	void close();
	~LOG();
private:
	std::ofstream Logfile;

};

