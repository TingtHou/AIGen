AIGen
=====
AIGen is an artificial intelligence software for complex genetic data analysis. This software based on two newly developed neural networks (i.e., kernel neural networks and functional neural networks) that are capable of modeling complex genotype-phenotype relationships (e.g., interactions) while providing robust performance against high-dimensional genetic data. Moreover, computationally efficient algorithms (e.g., a minimum norm quadratic unbiased estimation approach and batch training) are implemented in the package to accelerate the computation, making them computationally efficient for analyzing large-scale datasets with thousands or even millions of samples. 

Downloads:
==========

+ Windows: https://github.com/TingtHou/AIGen/releases/download/v1.1.0/AIGen_windows_x84_64.zip

+ Linux: https://github.com/TingtHou/AIGen/releases/download/v1.1.0/AIGen_linux_x86_64.zip

+ Sample Data: https://github.com/TingtHou/AIGen/releases/download/v1.1.0/example.zip

Build AIGen from source
=======================

Install AIGen
-------------
Windows:
^^^^^^^^
Precompiled Package: We offer a ready-to-use, precompiled package of AIGen that includes all necessary dependencies for libtorch. Simply download and run the package for immediate use.

Linux:
^^^^^^
Executable File: We provide a precompiled executable that relies only on the libtorch library. This version for Linux has no additional dependencies, making it the recommended option for ease of use.

For users who prefer or require manual compilation, detailed instructions are provided to build AIGen from the source. This approach helps ensure the best compatibility with your system's specific configuration.

Building on Linux
-----------------

On Debian/Ubuntu you will need to install the dependencies with (Fedora flavors can use a similar command)::

     sudo apt install gcc g++ make cmake libboost-all-dev libeigen3-dev

MKL library
^^^^^^^^^^^^^^^^^^^
In addition to those dependencies, the oneMKL package will be required using the following command::

     sudo apt update
     sudo apt install -y gpg-agent wget 
     wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
     echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
     sudo apt update
     sudo apt install intel-oneapi-mkl
     sudo apt install intel-oneapi-mkl-devel

LibTorch library
^^^^^^^^^^^^^^^^^^^
Libtorch is the PyTorch C++ frontend which is a pure C++ interface to the PyTorch machine learning framework. This library can be installed using the following command::

   wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
   unzip libtorch-shared-with-deps-latest.zip

Building from source
^^^^^^^^^^^^^^^^^^^^
then::

    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
    cmake --build . --config Release


Usage
=====

Kernel neural network
---------------------

Building kernel matrix
^^^^^^^^^^^^^^^^^^^^^^

The software accepts input data in either binary or text format. Use the "--bfile" option for binary data and "--file" option for text data. Additionally, the "--make-kernel" and "--make-bin" options enable the generation of a kernel matrix, which is saved in binary file format compatible with GCTA (*.grm.bin, *.grm.id)::

 ./AIGen --bfile ./example/sample --make-kernel 2 --make-bin ./example/product


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

 ./AIGen --kernel ./example/kernel1 --phe ./example/phenW.1.phen  --qcovar ./example/Covs.txt --KNN --iterate 100 --tolerance 1e-5 --out ./result/out

**Command Options** :

- ``--kernel ./example/kernel1`` : Specifies the prefix for kernel matrix files located in the ``./example`` directory.
- ``--phe ../../s3/phenW.1.phen`` : Specifies the phenotype file path.
- ``--qcovar ../../s3/Covs.txt`` : Specifies the quantitative covariates file path.
- ``--KNN`` : Initiates KNN analysis.
- ``--iterate 100`` : Sets the iteration count for the MINQUE algorithm to 100.
- ``--tolerance 1e-5`` : Sets the tolerance threshold for the MINQUE algorithm.
- ``--out ./result/out`` : Specifies the output file location and name in the ``./result`` directory.

To use multiple kernel matrices in the analysis, the ``--mkernel`` option is available. This option allows specifying a file that contains the paths to multiple kernel matrix files. The following command is a example::

 ./AIGen --mkernel ./example/mltgrm --phe ./example/phenW.1.phen  --qcovar ./example/Covs.txt --KNN --iterate 100 --tolerance 1e-5 --out ./result/out

Here, the file ``./example/mltgrm`` should list the paths to the individual kernel matrix files for use in the analysis.


This software allows for phenotype prediction using the --predict option::

 ./AIGen --mkernel ./example/mltgrm --phe ./example/phenW.1.phen  --qcovar ./example/Covs.txt --KNN --iterate 100 --tolerance 1e-5 --out ./result/out --predict 0

In this context, **"1"** signifies the Leave-One-Out prediction method, whereas **"0"** denotes the use of BLUP (Best Linear Unbiased Prediction).


Functional neural network
-------------------------

The KNN software provides a comprehensive suite of tools for statistical genetics and machine learning analyses, including the advanced Functional Neural Network (FNN) method. This section  guides users through the process of performing an FNN analysis, using  genetic and phenotype data。

**Example**::

 ./AIGen --bfile ./example/sample --phe ./example/y.txt  --FNN --layer 28,2,1  --basis 0 --optim 0 --epoch 3000 --lambda 0 --lr  0.001 --ratio 0.8

- ``--bfile ../../train/sample`` : Specifies the binary input files (.bed, .bim, .fam) located in the ``../../train/gene`` directory.

- ``--phe ../../train/y.txt`` : Points to the phenotype data file located in the ``../../train`` directory.

- ``--FNN`` : Indicates the analysis should use the Functional Neural Network approach.

- ``--layer 28,2,1`` : Defines the function neural network architecture with 28 nodes in the input layer, 2 nodes in the hidden layer, and 1 node in the output layer.

- ``--basis 0`` : Chooses Wavelet basis functions for the hidden layers (0 denotes Wavelet basis).

- ``--optim 0`` : Selects the Adam optimizer for training (0 for Adam).

- ``--epoch 3000`` : Sets the number of training epochs to 3000.

- ``--lambda 0`` : Specifies no regularization in the loss function (lambda = 0).

- ``--lr 0.001`` : Sets the learning rate to 0.001.

- ``--ratio 0.8`` : Uses 80% of the dataset for training and the remaining 20% for validation/testing.


This example command instructs the software to train an FNN model on genetic data located in ../example/gene, with phenotype outcomes provided in ../example/y.txt. The network is structured with 28 nodes in the input layer, 2 nodes in one hidden layer, and 1 node in the output layer. The Wavelet basis function is used in the FNN, with the Adam optimizer, 3000 epochs, no regularization (lambda set to 0), a learning rate of 0.001, and 80% of the data used for training.


Neural network
-------------------------
The KNN software also offers capabilities for performing analyses with Traditional Neural Networks. This manual section delivers comprehensive guidance on conducting a Traditional NN analysis, utilizing the KNN software's robust features for predicting phenotype given genetic and covariates data.

Here's an example::

 ./AIGen --bfile ./example/sample --phe ./example/y.txt  --NN --layer 50,20,1   --optim 0 --epoch 3000 --lambda 0 --lr  0.001 --ratio 0.8

- ``--NN``: Indicates that the analysis will use a traditional Neural Network approach, as opposed to a Functional Neural Network (FNN) or other methods available in the software.

- ``--layer 50,20,1`` : Specifies the architecture of the neural network, consisting of 28 nodes in the input layer, 2 nodes in the hidden layer, and 1 node in the output layer. **Important:** The number of nodes in the input layer must correspond to the number of genetic variants.

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

**Activation Functions**

In the current version of the software, the activation function for the neural network layers is set to a sigmoid function. In future releases, we plan to expand the available options by including a variety of other activation functions to enhance the model's flexibility and performance in capturing complex patterns within the data.




