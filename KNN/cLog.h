#pragma once
#include <iostream>
#include <fstream>
class LOG
{
public:
	LOG(std::string logfile);
	LOG();
	void write(std::string outline, bool terminateOpt);
	void close();
	~LOG();
private:
	std::ofstream Logfile;

};

