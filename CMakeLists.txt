cmake_minimum_required (VERSION 3.5.1)

project(FHEON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
option( BUILD_STATIC "Set to ON to include static versions of the library" OFF)

################################################################################
################################# BUILD OPTIONS ###############################
################################################################################

# Pipeline mode: SINGLE_INPUTS, BATCH_INPUTS, or ALL
set(MODE "ALL" CACHE STRING "Pipeline mode: SINGLE_INPUTS, BATCH_INPUTS, or ALL")
set_property(CACHE MODE PROPERTY STRINGS "ALL" "SINGLE_INPUTS" "BATCH_INPUTS")

# For SINGLE_INPUTS: specific model to build (ALL, BasicMLP, LeNet5, ResNet20, VGG11, VGG16, ResNet34)
set(SINGLE_MODEL "ALL" CACHE STRING "Single-input model to build: ALL, BasicMLP, LeNet5, ResNet20, VGG11, VGG16, ResNet34")
set_property(CACHE SINGLE_MODEL PROPERTY STRINGS "ALL" "BasicMLP" "LeNet5" "ResNet20" "VGG11" "VGG16" "ResNet34")

# Default test size for single-input models
set(TEST_SIZE "10" CACHE STRING "Default test size / batch size for single-input models")

# For BATCH_INPUTS: specific resnet to build (ALL, ResNet20, ResNet34) and batch size
set(BATCH_MODEL "ALL" CACHE STRING "Batch model to build: ALL, ResNet20, ResNet34")
set_property(CACHE BATCH_MODEL PROPERTY STRINGS "ALL" "ResNet20" "ResNet34")

set(BATCH_SIZE "ALL" CACHE STRING "Batch size: ALL, 16, 32, 64, 128, 256, 512")
set_property(CACHE BATCH_SIZE PROPERTY STRINGS "ALL" "16" "32" "64" "128" "256" "512")

# Validate inputs
if(NOT MODE MATCHES "SINGLE_INPUTS|BATCH_INPUTS|ALL")
    message(FATAL_ERROR "MODE must be SINGLE_INPUTS, BATCH_INPUTS, or ALL")
endif()

if(MODE STREQUAL "SINGLE_INPUTS" OR MODE STREQUAL "ALL")
    if(NOT SINGLE_MODEL MATCHES "ALL|BasicMLP|LeNet5|ResNet20|VGG11|VGG16|ResNet34")
        message(FATAL_ERROR "SINGLE_MODEL must be ALL, BasicMLP, LeNet5, ResNet20, VGG11, VGG16, or ResNet34")
    endif()
endif()

if(MODE STREQUAL "BATCH_INPUTS" OR MODE STREQUAL "ALL")
    if(NOT BATCH_MODEL MATCHES "ALL|ResNet20|ResNet34")
        message(FATAL_ERROR "BATCH_MODEL must be ALL, ResNet20, or ResNet34")
    endif()
    if(NOT BATCH_SIZE MATCHES "ALL|16|32|64|128|256|512")
        message(FATAL_ERROR "BATCH_SIZE must be ALL, 16, 32, 64, 128, 256, or 512")
    endif()
endif()

###############################################################################

################################################################################
# Find OpenFHE (and HEXL if available)
################################################################################

# Try to locate HEXL first. If it is not installed separately, this will not fail.
find_package(HEXL CONFIG QUIET)

find_package(OpenFHE CONFIG REQUIRED)

if(OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")

    message(STATUS "OpenFHE Shared Libraries: ${OpenFHE_SHARED_LIBRARIES}")
    message(STATUS "OpenFHE Static Libraries: ${OpenFHE_STATIC_LIBRARIES}")

    if(TARGET HEXL::hexl)
        message(STATUS "HEXL target found.")
    else()
        message(STATUS "HEXL target NOT found.")
    endif()

else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

################################################################################
# Build configuration
################################################################################

message(STATUS "Build Configuration:")
message(STATUS "  Mode: ${MODE}")

if(MODE STREQUAL "SINGLE_INPUTS" OR MODE STREQUAL "ALL")
    message(STATUS "  Single Model: ${SINGLE_MODEL}")
endif()

if(MODE STREQUAL "BATCH_INPUTS" OR MODE STREQUAL "ALL")
    message(STATUS "  Batch Model: ${BATCH_MODEL}")
    message(STATUS "  Batch Size: ${BATCH_SIZE}")
endif()

set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -O3 -DNDEBUG -march=native")

################################################################################
# Include directories
################################################################################

include_directories(
    ${OPENMP_INCLUDES}
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

################################################################################
# Link directories
################################################################################

link_directories(
    ${OpenFHE_LIBDIR}
    ${OPENMP_LIBRARIES}
)

################################################################################
# Compiler flags
################################################################################

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
endif()

################################################################################
# Libraries
################################################################################

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES} stdc++fs)
else()
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS}")
    link_libraries(${OpenFHE_SHARED_LIBRARIES} stdc++fs)
endif()

if(TARGET HEXL::hexl)
    link_libraries(HEXL::hexl)
endif()


################################################################################
############################## SINGLE INPUT MODELS #############################
################################################################################

if(MODE STREQUAL "SINGLE_INPUTS" OR MODE STREQUAL "ALL")
    # Grouping HESingleModels source files
    set(HESingleModels_SOURCES
        BasicMLP.cpp
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
        
        # Check if we should build this model
        # Extract base model name (e.g., "ResNet20" from "ResNet20Basic")
        string(REGEX MATCH "^[A-Za-z]+[0-9]+" base_name ${model_name})
        
        set(should_build FALSE)
        if(SINGLE_MODEL STREQUAL "ALL")
            set(should_build TRUE)
        elseif(SINGLE_MODEL STREQUAL "BasicMLP" AND base_name STREQUAL "BasicMLP")
            set(should_build TRUE)
        elseif(SINGLE_MODEL STREQUAL "LeNet5" AND base_name STREQUAL "LeNet5")
            set(should_build TRUE)
        elseif(SINGLE_MODEL STREQUAL "ResNet20" AND base_name STREQUAL "ResNet20")
            set(should_build TRUE)
        elseif(SINGLE_MODEL STREQUAL "ResNet34" AND base_name STREQUAL "ResNet34")
            set(should_build TRUE)
        elseif(SINGLE_MODEL STREQUAL "VGG11" AND base_name STREQUAL "VGG11")
            set(should_build TRUE)
        elseif(SINGLE_MODEL STREQUAL "VGG16" AND base_name STREQUAL "VGG16")
            set(should_build TRUE)
        endif()
        
        if(should_build)
            add_executable(${model_name}
                HESingleModels/${model}
                include/Utils.h include/UtilsData.h
                include/FHEONHEController.h src/FHEONHEController.cpp
                include/FHEONANNController.h src/FHEONANNController.cpp
            )
            target_compile_definitions(${model_name} PRIVATE DEFAULT_TEST_SIZE=${TEST_SIZE})
        endif()
    endforeach()
endif()

###############################################################################
############################# BATCH MODELS ####################################
###############################################################################

if(MODE STREQUAL "BATCH_INPUTS" OR MODE STREQUAL "ALL")
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
        
        # Extract model type (TResNet20 or TResNet34) and batch size
        string(REGEX MATCH "TResNet[0-9]+" model_type ${model_name})
        string(REGEX MATCH "N[0-9]+" batch_num ${model_name})
        string(REGEX MATCH "[0-9]+" extracted_batch_size ${batch_num})
        
        # Check if we should build this model
        set(should_build FALSE)
        
        # Filter by BATCH_MODEL and BATCH_SIZE
        if(model_name MATCHES "JoinedKeys")
            # Skip JoinedKeys variants for now or include if BATCH_MODEL is ALL
            if(BATCH_MODEL STREQUAL "ALL" AND (BATCH_SIZE STREQUAL "ALL" OR extracted_batch_size STREQUAL BATCH_SIZE))
                set(should_build TRUE)
            endif()
        elseif(BATCH_MODEL STREQUAL "ALL" AND (BATCH_SIZE STREQUAL "ALL" OR extracted_batch_size STREQUAL BATCH_SIZE))
            set(should_build TRUE)
        elseif(BATCH_MODEL STREQUAL "ResNet20" AND model_type STREQUAL "TResNet20" AND (BATCH_SIZE STREQUAL "ALL" OR extracted_batch_size STREQUAL BATCH_SIZE))
            set(should_build TRUE)
        elseif(BATCH_MODEL STREQUAL "ResNet34" AND model_type STREQUAL "TResNet34" AND (BATCH_SIZE STREQUAL "ALL" OR extracted_batch_size STREQUAL BATCH_SIZE))
            set(should_build TRUE)
        endif()
        
        if(should_build)
            add_executable(${model_name}
                HEBatchModels/${model}
                include/Utils.h include/UtilsData.h
                include/FHEONHEController.h src/FHEONHEController.cpp
                include/FHEONANNController.h src/FHEONANNController.cpp
                include/FHEONANNBatchController.h src/FHEONANNBatchController.cpp
            )
            target_compile_definitions(${model_name} PRIVATE DEFAULT_TEST_SIZE=${extracted_batch_size})
        endif()
    endforeach()
endif()
