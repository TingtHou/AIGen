cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(AIGen)
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXE_LINKER_FLAGS "-static")
option(USE_STATIC_LIBTORCH "use LibTorch static library" ON)

set(Boost_USE_STATIC_LIBS ON)


if(NOT DEFINED MKLROOT)
	set(MKLROOT "/opt/intel/oneapi/mkl/2024.1")
endif()

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
INCLUDE_DIRECTORIES( ${MKLROOT}/include )
add_compile_options(-DMKL_ILP64  -m64)

MESSAGE( [Main] " MKL_INCLUDE_DIRS = ${MKLROOT}/include") 
MESSAGE( [Main] " MKL_LIBRARIES = ${MKLROOT}/lib/") 



FIND_PACKAGE( Boost COMPONENTS program_options REQUIRED )
MESSAGE( [Main] " Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIR}") 
MESSAGE( [Main] " Boost_LIBRARIES = ${Boost_LIBRARIES}") 
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#set(EIGEN3_INCLUDE_DIRS "/home/ting/eigen-3.4.0/")
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIRS} )
MESSAGE( [Main] " Eigen_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${TORCH_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( ${TORCH_INCLUDE_DIRS} )
MESSAGE( [Main] "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}") 
MESSAGE( [Main] "TORCH_LIBRARIES = ${TORCH_LIBRARIES}") 


include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)


#Set Linker flags
add_executable(${PROJECT_NAME} ${SOURCES})
#target_compile_options(${PROJECT_NAME} "-DMKL_ILP64  -m64")
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
target_link_libraries(${PROJECT_NAME} "${Boost_LIBRARIES}"  "${CMAKE_DL_LIBS}" "${TORCH_LIBRARIES}" "-m64  -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_gnu_thread.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl -fopenmp -lrt -lnuma")
set_property(TARGET ${PROJECT_NAME}  PROPERTY CXX_STANDARD 17)
