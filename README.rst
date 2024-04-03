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




 
