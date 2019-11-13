cmake_minimum_required(VERSION 2.8)
PROJECT(KNN)                                   
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
SET(BUILD_SHARED_LIBS OFF)
SET(CMAKE_EXE_LINKER_FLAGS "-static")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(Boost_USE_STATIC_LIBS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -static -pthread -O3")


FIND_PACKAGE(Boost COMPONENTS program_options  REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
##mkl
#find_package(MKL REQUIRED)
set(BLA_DEFINITIONS)
message("MKL ROOt:  $ENV{MKLROOT}")
if ("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
endif()

# user defined options for MKL
option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
option(BLA_STATIC "MKL_LINK_TOOL_COMMAND" True)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

if (BLA_VENDOR MATCHES "_seq")
    set(MKL_USE_parallel OFF)
else()
    set(MKL_USE_parallel ON)
endif()

find_package(MKL REQUIRED)

if (MKL_FOUND)
    set(BLA_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLA_DEFINITIONS USE_MKL)
endif()

message(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")

include_directories(${PROJECT_SOURCE_DIR})

INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include/")


########################################
FILE(GLOB SOURCE_1 "${CMAKE_SOURCE_DIR}/src/*.cpp")
FILE(GLOB SOURCE_2 "${CMAKE_SOURCE_DIR}/*.cpp")

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_1} ${SOURCE_2})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${BLA_DEFINITIONS})

target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})

target_include_directories(${PROJECT_NAME} 
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BLA_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} -static-libgcc -static-libstdc++ ${Boost_LIBRARIES})
