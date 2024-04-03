# AIGen
AIGen is an artificial intelligence software for complex genetic data analysis. This software based on two newly developed neural networks (i.e., kernel neural networks and functional neural networks) that are capable of modeling complex genotype-phenotype relationships (e.g., interactions) while providing robust performance against high-dimensional genetic data. Moreover, computationally efficient algorithms (e.g., a minimum norm quadratic unbiased estimation approach and batch training) are implemented in the package to accelerate the computation, making them computationally efficient for analyzing large-scale datasets with thousands or even millions of samples. 

# Build AIGen from source
On any operating system, you should get the latest AIGen source the usual way::

$ git clone https://github.com/TingtHou/AIGen.git
$ cd AIGen

Then you need to get dependencies installed and configured for your operating system.

Building on Linux
^^^^^^^^^^^^^^^^^

On Debian/Ubuntu you will need to install the dependencies with (Fedora flavors can use a similar command)::

    $ sudo apt install gcc g++ make cmake libboost-all-dev

Install MKL library

To add APT repository access, install the prerequisites:
$ sudo apt update
$ sudo apt install -y gpg-agent wget
 
Set up the repository. To do this, download the key to the system keyring:
$ wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
 

Add the signed entry to APT sources and configure the APT client to use the Intel repository:
$echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
 

Update the packages list and repository index.
$ sudo apt update
 

Download the following from APT:
$ sudo apt install intel-oneapi-mkl

For installation with header files, install the oneMKL development package using the following command:
$ sudo apt install intel-oneapi-mkl-devel


 
