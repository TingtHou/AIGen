#include "../include/Options.h"
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

std::string Options::print()
{
	std::stringstream buffer;
	buffer << "Accepted Options:\n";
	for (auto it=programOptions.begin();it!=programOptions.end();it++)
	{
		buffer <<"  --"<< std::left << std::setw(11)<< it->first << "\t";
		auto& value = it->second.value();
		if (auto v = boost::any_cast<int>(&value))
			buffer << *v;
		else if (auto v = boost::any_cast<std::string>(&value))
			buffer << *v;
		else if (auto v = boost::any_cast<bool>(&value))
			buffer << *v;
		else if (auto v = boost::any_cast<float>(&value))
			buffer << *v;
		else if (auto v = boost::any_cast<double>(&value))
			buffer << *v;
		buffer << std::endl;
	}
	return buffer.str();
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
		("kernel", po::value<std::string>()->value_name("{prefix}"),
			"Specify .grm.bin + .grm.N.bin + .grm.id (GCTA triangular binary relationship matrix) filename prefix.\n\n")
	    ("mkernel", po::value<std::string>()->value_name("[filename]"),
				"Input multiple kernel files in binary format.\n")
		("weights", po::value<std::string>()->value_name("[filename]"), "Specify full name of weight vector file for iterative MINQUE.\n")
		("covar", po::value<std::string>()->value_name("[filename]"),
				"Input discrete covariates from a plain text file.\n")
		("qcovar", po::value<std::string>()->value_name("[filename]"),
				"Input quantitative covariates from a plain text file.\n")
		("keep", po::value<std::string>()->value_name("[filename]"),
				"Only use individuals in this file for analysis.\n")
		("intercept", po::value<bool>()->value_name("True/False")->default_value(false), "Specify whether insert a intercept into covariates matrix, default to false.\n");
		po::options_description optsFilesOperation("File Operations");
		optsFilesOperation.add_options()
		("impute", po::value<bool>()->value_name("True/False"),
				"Impute missing genotypes.\n")
		("recode", po::value<std::string>()->value_name("[filename]"),
				"Recode the binary kernel file to text format.\n")
		("precision", po::value<int>()->value_name("precision"),
				"Set precision for output kernel file (binary format); 0 for float, 1 for float.\n")
		("make-bin", po::value<std::string>()->value_name("{prefix}"),
				"Generate .grm.bin + .grm.N.bin + .grm.id (GCTA triangular binary relationship matrix).\n");
	po::options_description optsKernelGenr("Kernel Parameters");
	optsKernelGenr.add_options()
		("make-kernel", po::value<std::string>()->value_name("[kernel name]"), "Compute kernel matrix.\n"
		 "mode 0: CAR kernel; mode 1: Identity kernel; mode 2: Product kernel; mode 3: Polymonial kernel; mode 4: Gaussian kernel; mode 5 IBS.\n")
		("std", "Standardize SNP data.\n")
	//	("weight", po::value<std::string>()->value_name("[filename]"), "Specify full name of weight vector file.\n")
		("scale", po::value<bool>()->value_name("True/False"), "The weight value will be scaled.\n")
		("constant", po::value<float>()->value_name("value"), "The constant value used for polynomial kernel calculating.\n")
		("deg", po::value<float>()->value_name("value"), "The degree value used for polynomial kernel calculating.\n")
		("sigma", po::value<float>()->value_name("value"), "The standard deviation used for Gaussian kernel.\n");
	po::options_description optsKNNAlgorithm("KNN Algorithm Parameters");
	optsKNNAlgorithm.add_options()
		("KNN", "Implement KNN analysis.\n")
	//	("minque1","Use MINQUE(1) for estimate.\n")
		("minque0", "Use MINQUE(0) for estimate.\n")
		("iterate", po::value<int>()->value_name("times"), "The iterate times used in iterate minque method.\nDefault to 100.\n")
		("tolerance", po::value<float>()->value_name("value"), "The threshold value used in iterate minque method.\nDefault to 1e-6.\n")
		("inverse", po::value<std::string>()->value_name("mode"), "The matrix decomposition.\n"
			"mode 0: Cholesky decomposition; mode 1: LU decomposition; mode 2: QR decomposition; mode 3: SVD decomposition.\n")
		("pseudo", po::value<bool>()->value_name("true/fasle"), "The pseudo will be used if the matrix is not invertible.\n"
			"default to True.\n"
			"If this option is set, the inverse mode should be mode 2 or mode 3.\n")
		("ginverse", po::value<std::string>()->value_name("mode"), "The alternative matrix decomposition if the matrix is not invertible and the pseudo inverse is allowed.\n"
				"Only mode 2, QR decomposition and mode 3, SVD decomposition are available.\n"
				"Default to mode 3.\n")
		("predict", po::value<int>()->value_name("mode"), "Prediction according to estimation results.\n"
														"0: BLUP; 1: Leave one out\n")
		("alphaKNN", po::value<int>()->value_name("degree"), "Adopt 2-layer KNN with 1 latent feature h and order alpha polynomial kernels.\n")
		("batch", po::value<int>()->value_name("num"), "Split a super-large kernels into 'num' smaller batch to analysis.\n")
		("seed", po::value<float>()->value_name("num")->default_value(1), "Set seed for random process.\n")
		("echo", po::value<bool>()->value_name("True/False"), "Print the results at each iterate or not")
		("fix",  "Skip fixed effects estimation.\n")
		("thread", po::value<int>()->value_name("num"), "Set a 'num' size thread pool for multi-thread analysis.\n");

	po::options_description optsFNNAlgorithm("FNN Algorithm Parameters");
	optsFNNAlgorithm.add_options()
		("FNN", "Implement functional nerual network analysis.\n")
		("NN",  "Implement traditional nerual network analysis.\n")
		("layer", po::value<std::string>()->value_name("inputs,layer1,layer2, ..., output"), "define the numbers of nodes in the hidden layers.\n"
			"In FNN, the number of layers indicates the number of the basis function used to fit the data in this layer.\n"
			"In NN, the number of layers indicates the data points in this layer\n"
			"If the phenotype is categorical data, the output equals to the number of class.\n")
		("epoch", po::value<double>()->value_name("num")->default_value(3000), "set Epoch number for the training.\n")
		("lr", po::value<double>()->value_name("num")->default_value(0.001), "set learning rate for the training.\n")
		("optim", po::value<int>()->value_name("num")->default_value(1), "set optimizer  for the training, 0 Adam; 1 SGD.\n")
		("lambda", po::value<std::string>()->value_name("num")->default_value("1"), "define the lambda value for training, default 1.\n")
		("loss", po::value<int>()->value_name("[0|1|2]")->default_value(0), "define the loss function for training, 0: MSE; 1: BCE; 2: CrossEntropy.\n")
		("ratio", po::value<float>()->value_name("num")->default_value(0.8), "define the ratio of training dataset to testing dataset\n")
		("basis", po::value<int>()->value_name("[0|1]")->default_value(0), "define the basis function in the hidden layers, 0: Wavelet basis; 1: B spline basis.\n")
		("save", po::value<std::string>()->value_name("[filename]"), "save the model.\n")
		("load", po::value<std::string>()->value_name("[filename]"), "load the model from a file.\n");
	/*
	po::options_description optsComputerDevice("Computing Device Options");
	optsComputerDevice.add_options()
		("GPU", " Use CPU for computation.\n")
		("e", po::value<int>()->value_name("[dev]"), "Select GPUs to use.\n");

*/

	po::options_description optsDescOutFiles("Output Files");
	optsDescOutFiles.add_options()
		("out", po::value<std::string>()->value_name("[filename]"), "Specify full name of output file.\n")
		("log", po::value<std::string>()->value_name("[filename]"), "Specify full name of log file.\n");

// 	po::options_description optsCheckingParametr("Checking Parameter");
// 	optsCheckingParametr.add_options()
// 		("check", "Checking software mode.\n");

	optsDescCmdLine.
		add(optsDescGeneral).
		add(optsDescInFiles).
		add(optsFilesOperation).
		add(optsKernelGenr).
		add(optsKNNAlgorithm).
		add(optsFNNAlgorithm).
	//	add(optsComputerDevice).
		add(optsDescOutFiles);
//		add(optsCheckingParametr);
	try
	{
		po::store(parse_command_line(
			argc, argv, optsDescCmdLine), programOptions);
		po::notify(programOptions);
	}
	catch (po::too_many_positional_options_error &e)
	{
		std::cerr << e.what() << std::endl;
//		std::cout << optsDescCmdLine << std::endl;
		exit(1);
	}
	catch (po::error_with_option_name &e)
	{
	// Another usage error occurred
		std::cerr << e.what() << std::endl;
//		std::cout << optsDescCmdLine << std::endl;
		exit(1);
	}


}
