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

# Grouping include directories
include_directories(
    ${OPENMP_INCLUDES}
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Grouping link directories
link_directories(
    ${OpenFHE_LIBDIR}
    ${OPENMP_LIBRARIES}
)

# Setting compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
endif()

# Handling static vs shared library linking
if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES} stdc++fs)
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES} stdc++fs)
endif()

# Optional HEXL support
if(TARGET HEXL::hexl)
    message(STATUS "HEXL found, linking HEXL::hexl")
    link_libraries(HEXL::hexl)
endif()

################################################################################
############################## SINGLE INPUT MODELS #############################
################################################################################

# Grouping HESingleModels source files
set(HESingleModels_SOURCES
    LeNet5.cpp
    ResNet20Basic.cpp
    ResNet20DataPreprocessed.cpp
    ResNet20MultiChannels.cpp
    ResNet20MultiChansAndSlots.cpp
    ResNet20Optimized.cpp
    ResNet34.cpp
    ResNet34Optimized.cpp
    VGG11.cpp
    VGG11Optimized.cpp
    VGG16.cpp
    VGG16Optimized.cpp
)

foreach(model ${HESingleModels_SOURCES})
    string(REPLACE ".cpp" "" model_name ${model})
    add_executable(${model_name}
        HESingleModels/${model}
        include/Utils.h include/UtilsData.h
        include/FHEONHEController.h src/FHEONHEController.cpp
        include/FHEONANNController.h src/FHEONANNController.cpp
    )
    target_compile_definitions(${model_name} PRIVATE DEFAULT_ARG=10)
endforeach()


###############################################################################
############################# BATCH MODELS ####################################
###############################################################################

# Grouping HEBatchModels source files
set(HEBatchModels_SOURCES
    TResNet20N16.cpp
    TResNet20N64.cpp
    TResNet20N256.cpp
    TResNet20N32.cpp
    TResNet20N128.cpp
    TResNet20N512.cpp
    TResNet34N16.cpp
    TResNet34N64.cpp
    TResNet34N128.cpp
    TResNet34N32.cpp
    TResNet34N128P2.cpp
    TResNet34N256.cpp
    TResNet20N64JoinedKeys.cpp
    TResNet20N256JoinedKeys.cpp
    TResNet34N16JoinedKeys.cpp
)

foreach(model ${HEBatchModels_SOURCES})
    string(REPLACE ".cpp" "" model_name ${model})
    add_executable(${model_name}
        HEBatchModels/${model}
        include/Utils.h include/UtilsData.h
        include/FHEONHEController.h src/FHEONHEController.cpp
        include/FHEONANNController.h src/FHEONANNController.cpp
        include/FHEONANNBatchController.h src/FHEONANNBatchController.cpp
    )
    if(model_name MATCHES "N16|N64|N128|N256|N512")
        string(REGEX MATCH "[0-9]+" batch_size ${model_name})
        target_compile_definitions(${model_name} PRIVATE DEFAULT_ARG=${batch_size})
    endif()
endforeach()