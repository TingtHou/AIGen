#pragma once
#include <iostream>
#include <fstream>
class LOG
{
public:
	LOG(std::string logfile);
	void write(std::string outline, bool terminateOpt);
	~LOG();
private:
	std::ofstream Logfile;

};

