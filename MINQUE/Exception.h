#include <boost/format.hpp>
#include <boost/exception/all.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

class Exception :
virtual public std::exception,
virtual public boost::exception
{
public:
    const char * what() const throw();
    virtual std::string getMessage() const = 0;
};

// ItemException deals with the error related to
// any data structure consisting of homogeneous elements
class ItemException : virtual public Exception {};

typedef boost::error_info<struct TagFileName, std::string> ErrInfoFileName;
typedef boost::error_info<struct TagFileType, std::string> ErrInfoFileType;
typedef boost::error_info<struct TagLineNo, unsigned int> ErrInfoLineNo;

typedef boost::error_info<struct TagItem, std::string> ErrInfoItem;
typedef boost::error_info<
	struct TagItemType, std::string> ErrInfoItemType;

class NoItemException : virtual public ItemException
{
public:
	std::string getMessage() const;
};
std::string readFileErrorMessage(const Exception &excpt);

void handleExceptionInMain(const boost::exception &excpt);

#define TRY_MAIN()   int main(int argc, char *argv[]) try

#define CATCH_EXCEPTION()   \
    catch (boost::exception &e)  \
    {  \
        Gwas::handleExceptionInMain(e);  \
    }  \
    catch (std::exception &e)  \
    {  \
        std::cerr << "An exception has occurred: " << e.what() << std::endl;  \
        return 1;  \
    }  \
    catch (...)  \
    {  \
        std::cerr << "An unknown exception has occurred." << std::endl;  \
        return 1;  \
    }
