#include "pch.h"
#include "Options.h"
namespace po = boost::program_options;
Options::Options(int argc, const char * const argv[])
{
	boostProgramOptionsRoutine(argc, argv);
}

Options::~Options()
{
}

boost::program_options::variables_map Options::GetOptions()
{
	return programOptions;
}

boost::program_options::options_description Options::GetDescription()
{
	return optsDescCmdLine;
}


void Options::boostProgramOptionsRoutine(int argc, const char * const argv[])
{

	po::options_description optsDescGeneral("General Options");
	optsDescGeneral.add_options()
		("help,h", "display this helping message\n")
		("version,v", "show program version information\n");
	po::options_description optsDescInFiles("Input Files");
	optsDescInFiles.add_options()
		("file",po::value<std::string>()->value_name("{prefix}"),
			"Specify .ped + .map filename prefix.\n")
		("ped", po::value<std::string>()->value_name("[filename]"),
			"Specify full name of .ped file.\n")
		("map", po::value<std::string>()->value_name("[filename]"),
			"Specify full name of .map file.\n")
		("bfile", po::value<std::string>()->value_name("{prefix}"),
			"Specify .bed + .bim + .fam prefix.\n")
		("bed", po::value<std::string>()->value_name("[filename]"),
			"Specify full name of .bed file.\n")
		("bim", po::value<std::string>()->value_name("[filename]"),
			"Specify full name of .bim file.\n")
		("fam", po::value<std::string>()->value_name("[filename]"),
			"Specify full name of .fam file.\n")
		("phe", po::value<std::string>()->value_name("[filename]"),
			"Specify full name of phenotype file\n")
		("kernel", po::value<std::vector<std::string> >()->multitoken()->value_name("[filenames]"),
			"Specify full name of kernel files, different kernel files should be delimited by white-space characters\n");
	po::options_description optsAlgorithm("Algorithm Parameters");
	optsAlgorithm.add_options()
		("iterate", po::value<int>()->value_name("times"), "The iterate times used in iterate minque method.\nDefault to 20.\n")
		("tolerance", po::value<int>()->value_name("value"), "The threshold value used in iterate minque method.\nDefault to 1e-6.\n")
		("inverse", po::value<std::string>()->value_name("mode"), "The matrix decomposition.\n"
			"mode 0: Cholesky decomposition; mode 1: LU decomposition; mode 2: QR decomposition; mode 3: SVD decomposition.")
		("pseudo", po::value<bool>()->value_name("true/fasle"), "The pseudo will be used if the matrix is not invertible.\n"
				"default to True.\n"
				"If this option is set, the inverse mode should be mode 2 or mode 3.")
		("altinverse", po::value<std::string>()->value_name("mode"), "The alternative matrix decomposition if the matrix is not invertible and the pseudo inverse is allowed.\n"
				"Only mode 2: QR decomposition and mode 3 : SVD	decomposition are available.\n"
				"Default to mode 3.");
	po::options_description optsComputerDevice("Computing Device Options");
	optsComputerDevice.add_options()
		("GPU", " Use CPU for computation.\n")
		("e", po::value<int>()->value_name("[dev]"), "CUDA GPUs to use.\n");


	po::options_description optsDescOutFiles("Output Files");
	optsDescOutFiles.add_options()
		("out", po::value<std::string>()->value_name("[filename]"), "Specify full name of output file.\n")
		("log", po::value<std::string>()->value_name("[filename]"), "Specify full name of log file.\n");

	po::options_description optsCheckingParametr("Checking Parameter");
	optsCheckingParametr.add_options()
		("check", "Checking software mode.\n");

	optsDescCmdLine.
		add(optsDescGeneral).
		add(optsDescInFiles).
		add(optsAlgorithm).
		add(optsComputerDevice).
		add(optsDescOutFiles).
		add(optsCheckingParametr);
	try
	{
		po::store(parse_command_line(
			argc, argv, optsDescCmdLine), programOptions);
		po::notify(programOptions);
	}
	catch (po::too_many_positional_options_error &e)
	{
		std::cerr << e.what() << std::endl;
		std::cout << optsDescCmdLine << std::endl;
		exit(1);
	}
	catch (po::error_with_option_name &e)
	{
	// Another usage error occurred
		std::cerr << e.what() << std::endl;
		std::cout << optsDescCmdLine << std::endl;
		exit(1);
	}


}
