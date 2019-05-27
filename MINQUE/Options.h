#pragma once
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <stdio.h>
#include <vector>
class Options
{
public:
	Options(int argc, const char * const argv[]);
	~Options();
	boost::program_options::variables_map GetOptions();
	boost::program_options::options_description GetDescription();
private:
	void boostProgramOptionsRoutine(int argc, const char * const argv[]);
	boost::program_options::variables_map programOptions;
	boost::program_options::options_description optsDescCmdLine;
};
