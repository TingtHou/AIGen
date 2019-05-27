#include "pch.h"
#include "Exception.h"
#include <cassert>
#include <fstream>
#include <iostream>
using namespace std;

std::string readFileErrorMessage(const Exception &excpt)
{
	const std::string *fileType =
		boost::get_error_info<ErrInfoFileType>(excpt);
	const std::string *fileName =
		boost::get_error_info<ErrInfoFileName>(excpt);
	const unsigned int *lineNo =
		boost::get_error_info<ErrInfoLineNo>(excpt);

	if (fileName)
	{
		assert(fileType);
		if (lineNo)
		{
			return boost::str(boost::format(
				"An error is detected when the program reads to "
				"line %1% of the %2% file '%3%': ") %
				*lineNo % *fileType % *fileName);
		}
		else
		{
			return boost::str(boost::format(
				"An error is detected when the program is "
				"reading the %1% file '%2%': ") %
				*fileType % *fileName);
		}
	}
	else
	{
		return "";
	}
}

void handleExceptionInMain(const boost::exception &excpt)
{
	try
	{
		if (const Exception *pExcpt =
			dynamic_cast<const Exception *>(&excpt))
		{
			std::cerr << pExcpt->what() << std::endl;
		}
		else
		{
			std::cerr << "An error has occurred." << std::endl;
		}

		std::ofstream diagnosticFile("diagnostic.txt");
		if (diagnosticFile)
		{
			diagnosticFile << boost::diagnostic_information(excpt) << std::endl;
			std::cerr <<
				"Please see 'diagnostic.txt' for more technical details." << std::endl;
		}
		else
		{
			std::cerr <<
				"Unable to write 'diagnostic.txt'. " <<
				"Print the technical details here: " << std::endl;
			std::cerr << boost::diagnostic_information(excpt) << std::endl;
		}
	}
	catch (std::exception &excpt)
	{
		std::cerr << "An exception has occurred: " << excpt.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "An Unknown exception has occurred." << std::endl;
	}

	exit(1);
}


const char * Exception::what() const throw() try
{
    static std::string message;
    message = getMessage();
    return message.c_str();
}
catch (...)
{
    return typeid(*this).name();
}



std::string NoItemException::getMessage() const
{
	const std::string *itemType =
		boost::get_error_info<ErrInfoItemType>(*this);
	assert(itemType);
	return boost::str(boost::format(
		"%sNo %s is read.") %
		readFileErrorMessage(*this) % *itemType);
}
