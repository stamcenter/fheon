cmake_minimum_required (VERSION 3.5.1)

project(FHEON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option( BUILD_STATIC "Set to ON to include static versions of the library" OFF)

find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
endif()

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES} stdc++fs)
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES} stdc++fs)
endif()

# Test Models
add_executable(lenet5
        HEmodels/LeNet5.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(lenet5 PRIVATE DEFAULT_ARG=100)

add_executable(resnet20Basic
        HEmodels/ResNet20Basic.cpp
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(resnet20Basic PRIVATE DEFAULT_ARG=100)

add_executable(resnet20DataPreprocessed
        HEmodels/ResNet20DataPreprocessed.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(resnet20DataPreprocessed PRIVATE DEFAULT_ARG=100)

add_executable(resnet20MultiChannels
        HEmodels/ResNet20MultiChannels.cpp
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(resnet20MultiChannels PRIVATE DEFAULT_ARG=100)

add_executable(resnet20MultiChannelsAndSlots
        HEmodels/ResNet20MultiChansAndSlots.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(resnet20MultiChannelsAndSlots PRIVATE DEFAULT_ARG=100)

add_executable(resnet20Optimized
        HEmodels/ResNet20Optimized.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(resnet20Optimized PRIVATE DEFAULT_ARG=250)

add_executable(resnet34
        HEmodels/ResNet34.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(resnet34 PRIVATE DEFAULT_ARG=100)

add_executable(resnet34Optimized
        HEmodels/ResNet34Optimized.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(resnet34Optimized PRIVATE DEFAULT_ARG=100)

add_executable(vgg11
        HEmodels/VGG11.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(vgg11 PRIVATE DEFAULT_ARG=100)

add_executable(vgg11Optimized
        HEmodels/VGG11Optimized.cpp 
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(vgg11Optimized PRIVATE DEFAULT_ARG=100)

add_executable(vgg16
        HEmodels/VGG16.cpp
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
)
target_compile_definitions(vgg16 PRIVATE DEFAULT_ARG=100)

add_executable(vgg16Optimized
        src/Utils.h src/UtilsData.h 
        src/FHEONHEController.h src/FHEONHEController.cpp 
        src/FHEONANNController.h src/FHEONANNController.cpp
        HEmodels/VGG16Optimized.cpp 
)
target_compile_definitions(vgg16Optimized PRIVATE DEFAULT_ARG=100)
