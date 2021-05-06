cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(KNN)

option(USE_MKL "use MKL as pytorch LAPACK backend" ON)
option(USE_STATIC_LIBTORCH "use LibTorch static library" ON)

set(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE( Boost COMPONENTS program_options REQUIRED )
MESSAGE( [Main] " Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIR}") 
MESSAGE( [Main] " Boost_LIBRARIES = ${Boost_LIBRARIES}") 
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIRS} )
MESSAGE( [Main] " Eigen_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 

#set(Libtorch_Home "/mnt/c/Users/tingtinghou/Downloads/libtorch_cpu_mkl")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


if(USE_MKL AND USE_STATIC_LIBTORCH)
    if(NOT DEFINED MKL_HOME)
        set(MKL_HOME "/opt/intel/mkl")
    endif()
    if(NOT EXISTS ${MKL_HOME})
        message(FATAL "unable to locate MKL library, either set MKL_HOME or install MKL")
    endif()
	
	if(NOT DEFINED MKL_ROOT)
		set(MKL_ROOT "/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl")
	endif()
    if(NOT EXISTS ${MKL_ROOT})
        message(FATAL "unable to locate MKL library, either set MKL_HOME or install MKL")
    endif()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(MKL_ARCHITECTURES intel64)
    else()
        set(MKL_ARCHITECTURES ia32)
    endif()
    #must consistent with pytorch static library.
    set(INT_LIB "libmkl_intel_lp64.a")
    set(SEQ_LIB "libmkl_sequential.a")
    #set(THR_LIB "libmkl_intel_thread.a")
    set(THR_LIB "libmkl_gnu_thread.a")
    set(COR_LIB "libmkl_core.a")
    
    find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT}/include)
    
    set(MKL_LIB_PATH ${MKL_HOME}/lib/${MKL_ARCHITECTURES})
    find_library(MKL_INTERFACE_LIBRARY
                 NAMES ${INT_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    find_library(MKL_SEQUENTIAL_LAYER_LIBRARY
                 NAMES ${SEQ_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    find_library(MKL_THREAD_LAYER_LIBRARY
                 NAMES ${THR_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    find_library(MKL_CORE_LIBRARY
                 NAMES ${COR_LIB}
                 PATHS ${MKL_LIB_PATH}
                 NO_DEFAULT_PATH)
    
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_CORE_LIBRARY} ${MKL_THREAD_LAYER_LIBRARY} ${MKL_INTERFACE_LIBRARY} )
    set(MKL_LIBRARIES -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group)
endif()
include_directories(${MKL_INCLUDE_DIRS})
#find_package(MKL REQUIRED)
MESSAGE( [Main] " MKL_INCLUDE_DIRS = ${MKL_INCLUDE_DIRS}") 
MESSAGE( [Main] " MKL_LIBRARIES = ${MKL_LIBRARIES}")  

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})
#MESSAGE( [Main] "TORCH_LIBRARIES = ${TORCH_LIBRARIES}") 
#set(TORCH_LIBRARIES
#    -Wl,--whole-archive
#	 ${Libtorch_Home}/lib/libtorch.a
#        ${Libtorch_Home}/lib/libc10.a
#        ${Libtorch_Home}/lib/libtorch_cpu.a
#    -Wl,--no-whole-archive
#    ${Libtorch_Home}/lib/libnnpack.a
	# ${Libtorch_Home}/lib/libcaffe2_protos.a
	# ${Libtorch_Home}/lib/libmkldnn.a
   # ${Libtorch_Home}/lib/libpthreadpool.a
    # ${Libtorch_Home}/lib/libcpuinfo.a
    # ${Libtorch_Home}/lib/libclog.a
    # ${Libtorch_Home}/lib/libsleef.a
    # ${Libtorch_Home}/lib/libfbgemm.a
   # ${Libtorch_Home}/lib/libasmjit.a
    # /usr/local/lib/libprotobuf.a
    # -Wl,--whole-archive
       # /usr/local/lib/libonnx.a
       # /usr/local/lib/libonnx_proto.a
# )

MESSAGE( [Main] "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}") 
MESSAGE( [Main] "TORCH_LIBRARIES = ${TORCH_LIBRARIES}") 


include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)


#Set Linker flags
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${MKL_LIBRARIES} ${TORCH_LIBRARIES}  ${CMAKE_DL_LIBS})