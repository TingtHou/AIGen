AIGen
=====
AIGen is an artificial intelligence software for complex genetic data analysis. This software based on two newly developed neural networks (i.e., kernel neural networks and functional neural networks) that are capable of modeling complex genotype-phenotype relationships (e.g., interactions) while providing robust performance against high-dimensional genetic data. Moreover, computationally efficient algorithms (e.g., a minimum norm quadratic unbiased estimation approach and batch training) are implemented in the package to accelerate the computation, making them computationally efficient for analyzing large-scale datasets with thousands or even millions of samples. 

Build AIGen from source
=======================

On any operating system, you should get the latest AIGen source the usual way::

$ git clone https://github.com/TingtHou/AIGen.git
$ cd AIGen

Then you need to get dependencies installed and configured for your operating system.

Building on Linux
-----------------

On Debian/Ubuntu you will need to install the dependencies with (Fedora flavors can use a similar command)::

    $ sudo apt install gcc g++ make cmake libboost-all-dev libeigen3-dev

MKL library
^^^^^^^^^^^^^^^^^^^
In addition to those dependencies, the oneMKL package will be required using the following command::

    $ sudo apt update
    $ sudo apt install -y gpg-agent wget 
    $ wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    $ echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
    $ sudo apt update
    $ sudo apt install intel-oneapi-mkl
    $ sudo apt install intel-oneapi-mkl-devel

LibTorch library
^^^^^^^^^^^^^^^^^^^
Libtorch is the PyTorch C++ frontend which is a pure C++ interface to the PyTorch machine learning framework. This library can be installed using the following command::

   $ wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
   $ unzip libtorch-shared-with-deps-latest.zip

Building from source
^^^^^^^^^^^^^^^^^^^^
then::

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
    $ cmake --build . --config Release


Usage
=====

Kernel neural network
---------------------

Building kernel matrix
^^^^^^^^^^^^^^^^^^^^^^

The software accepts input data in either binary or text format. Use the "--bfile" option for binary data and "--file" option for text data. Additionally, the "--make-kernel" and "--make-bin" options enable the generation of a kernel matrix, which is saved in binary file format compatible with GCTA (*.grm.bin, *.grm.id)::

$ ./KNN --bfile ./example/sample --make-kernel 2 --make-bin ./example/product


This command will read PLINK binary files (sample.bim, sample.bed, and sample.fam) located in the "example" folder, generating a product kernel matrix. The output, in binary format, is saved as product.grm.bin and product.grm.id within the same "example" folder.

The software supports various kernel modes as follows:

1. Mode **0**: CAR (Covariance Allele Regression) Kernel
2. Mode **1**: Identity Kernel
3. Mode **2**: Product Kernel
4. Mode **3**: Polynomial Kernel
5. Mode **4**: Gaussian Kernel
6. Mode **5**: IBS (Identity By State)

Estimation and Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^
After generating kernel matrix, the variance component in the KNN model could be estimated using iterative IMINQUE with following command::

$ ./KNN --kernel ./example/kernel1 --phe ../../example/phenW.1.phen  --qcovar ../example/Covs.txt --KNN --iterate 100 --tolerance 1e-5 --out ./result/out

**Command Options** :

- ``--kernel ./example/kernel1`` : Specifies the prefix for kernel matrix files located in the ``./example`` directory.
- ``--phe ../../s3/phenW.1.phen`` : Specifies the phenotype file path.
- ``--qcovar ../../s3/Covs.txt`` : Specifies the quantitative covariates file path.
- ``--KNN`` : Initiates KNN analysis.
- ``--iterate 100`` : Sets the iteration count for the MINQUE algorithm to 100.
- ``--tolerance 1e-5`` : Sets the tolerance threshold for the MINQUE algorithm.
- ``--out ./result/out`` : Specifies the output file location and name in the ``./result`` directory.

To use multiple kernel matrices in the analysis, the ``--mkernel`` option is available. This option allows specifying a file that contains the paths to multiple kernel matrix files. The following command is a example::

$ ./KNN --mkernel ./example/mltgrm --phe ../../example/phenW.1.phen  --qcovar ../example/Covs.txt --KNN --iterate 100 --tolerance 1e-5 --out ./result/out

Here, the file ``./example/mltgrm`` should list the paths to the individual kernel matrix files for use in the analysis.


This software allows for phenotype prediction using the --predict option::

$ ./KNN --mkernel ./example/mltgrm --phe ../../example/phenW.1.phen  --qcovar ../example/Covs.txt --KNN --iterate 100 --tolerance 1e-5 --out ./result/out --predict 0

In this context, **"1"** signifies the Leave-One-Out prediction method, whereas **"0"** denotes the use of BLUP (Best Linear Unbiased Prediction).


Functional neural network
-------------------------







 
