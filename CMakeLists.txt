cmake_minimum_required (VERSION 3.5.1)

# link-time optimization through INTERPROCEDURAL_OPTIMIZATION (FHEON_LTO) needs the CMake 3.9 behaviour
if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif()

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

# Pipeline mode:
#   STANDARD_SINGLE_INPUT : HESingleModels/* with the non-threaded single-input primitives (FHEONANNController), sequential
#   BATCH_INPUTS          : HEBatchModels/* with the threaded batch primitives (FHEONANNBatchController)
#   SINGLE_INPUTS         : alias kept for compatibility with existing scripts
#   TRAINING              : encrypted training in the standard layout (HETrainingModels/: tuneResNet20, trainModules)
#   TESTING               : build a single TESTING_SRC target for local validation
#   ALL                   : public default build: all supported single-input, batch, and training targets
set(MODE "ALL" CACHE STRING "Pipeline mode: STANDARD_SINGLE_INPUT, SINGLE_INPUTS, BATCH_INPUTS, TESTING, TRAINING or ALL")
set_property(CACHE MODE PROPERTY STRINGS "ALL" "STANDARD_SINGLE_INPUT" "SINGLE_INPUTS" "BATCH_INPUTS" "TESTING" "TRAINING")

# Execution of the pipelines (fixed): the single-input models run sequentially with FHEONANNController; the batch models
# run threaded by default with FHEONANNBatchController, and the encrypted training models use the threaded standard
# training controllers. The multithreaded programs take --threads N at run time to cap the worker threads (and their memory).

# Model to build (base name of the source): ResNet20 for ResNet20Optimized.cpp, TResNet20N64.cpp, etc.
#   single-input modes : ALL, BasicMLP, LeNet5, ResNet20, VGG11, VGG16, ResNet34
#   batch modes        : ALL, LeNet5, ResNet20, ResNet34, VGG11, VGG16
set(MODEL "ALL" CACHE STRING "Model to build: ALL, BasicMLP, LeNet5, ResNet20, VGG11, VGG16, ResNet34")

# Default test size for single-input models
set(TEST_SIZE "10" CACHE STRING "Default test size / batch size for single-input models")

# For BATCH_INPUTS: batch size of the fixed-batch models (B<model>N<size>). The pipelines (B<model>Pipeline) take any
# number of images at run time (--test-size) and are built for every BATCH_SIZE.
set(BATCH_SIZE "ALL" CACHE STRING "Batch size: ALL, 16, 32, 64, 128, 256, 512")

# Security level of the CKKS contexts (FHEON_SECURITY_LEVEL, include/FHEONHEController.h). It is compiled into the
# controllers, so each level needs its own build directory (the Studio uses build/std128, build/std192, build/std256).
#   PERFORMANCE : performance efficient, not a standard level: the ring dimension and slots of each model, sparse
#                 ternary secret, HEStd_NotSet (the parameters FHEON was tuned with)
#   128|192|256 : HEStd_<n>_classic: OpenFHE picks the smallest ring dimension the HE security standard allows for the
#                 log QP of each model's modulus chain (uniform ternary secret, SPARSE_ENCAPSULATED for bootstrapping);
#                 the model's slot count is kept, so its packing does not change
set(SECURITY_LEVEL "PERFORMANCE" CACHE STRING "Security level: PERFORMANCE (not standard), 128, 192 or 256 bits")
set_property(CACHE SECURITY_LEVEL PROPERTY STRINGS "PERFORMANCE" "128" "192" "256")
string(TOUPPER "${SECURITY_LEVEL}" _fheon_sec)
if(_fheon_sec MATCHES "^(PERFORMANCE|NOTSET|0)$")
    set(FHEON_SECURITY_BITS 0)
elseif(_fheon_sec MATCHES "^(128|192|256)$")
    set(FHEON_SECURITY_BITS ${_fheon_sec})
else()
    message(FATAL_ERROR "SECURITY_LEVEL must be PERFORMANCE, 128, 192 or 256 (got '${SECURITY_LEVEL}')")
endif()
add_compile_definitions(FHEON_SECURITY_LEVEL=${FHEON_SECURITY_BITS})
if(FHEON_SECURITY_BITS EQUAL 0)
    message(STATUS "FHEON security level: performance efficient (HEStd_NotSet)")
else()
    message(STATUS "FHEON security level: ${FHEON_SECURITY_BITS}-bit (HEStd_${FHEON_SECURITY_BITS}_classic)")
endif()

# Testing mode: single source file compile support
set(TESTING_SRC "" CACHE STRING "Source file to compile in TESTING mode")
set(TESTING_TARGET "TestingExe" CACHE STRING "Executable name for MODE=TESTING")

# Performance of the binaries. Almost all of the run time of a FHEON program is spent inside OpenFHE (NTTs, key switching,
# bootstrapping), so how OpenFHE itself was built matters far more than these flags (see "Compiler flags" below). Every
# option checks that the compiler and linker support it and is skipped otherwise, so the defaults build everywhere.
#   FHEON_NATIVE : -march=native for the FHEON sources. The binaries then only run on CPUs with the instruction sets of the
#                  build machine. (An OpenFHE built with WITH_NATIVEOPT=ON passes -march=native to FHEON already.)
#   FHEON_LTO    : link-time optimization of the controllers and the models
#   FHEON_WERROR : keep the -Werror that OpenFHE_CXX_FLAGS carries (OFF: a warning of a newer compiler does not stop the build)
option(FHEON_NATIVE "Compile the FHEON sources for the CPU of the build machine (-march=native)" OFF)
option(FHEON_LTO "Link-time optimization of the FHEON controllers and models" OFF)
option(FHEON_WERROR "Treat compiler warnings as errors (the -Werror of OpenFHE_CXX_FLAGS)" OFF)

# Validate inputs
if(DEFINED THREADED)
    # build folders configured with an earlier CMakeLists.txt still have it in their cache
    message(STATUS "THREADED is no longer an option (single-input models: sequential, standard batch models: multithreaded)")
    unset(THREADED CACHE)
endif()
if(NOT MODE MATCHES "^(ALL|STANDARD_SINGLE_INPUT|SINGLE_INPUTS|BATCH_INPUTS|TESTING|TRAINING)$")
    message(FATAL_ERROR "MODE must be STANDARD_SINGLE_INPUT, SINGLE_INPUTS, BATCH_INPUTS, TESTING, TRAINING, or ALL")
endif()

set(BUILD_STANDARD_SINGLE OFF)
set(BUILD_STANDARD_BATCH OFF)
if(MODE MATCHES "^(ALL|STANDARD_SINGLE_INPUT|SINGLE_INPUTS)$")
    set(BUILD_STANDARD_SINGLE ON)
endif()
if(MODE MATCHES "^(ALL|BATCH_INPUTS)$")
    set(BUILD_STANDARD_BATCH ON)
endif()

if(BUILD_STANDARD_SINGLE)
    if(NOT MODEL MATCHES "^(ALL|BasicMLP|LeNet5|ResNet20|VGG11|VGG16|ResNet34)$")
        message(FATAL_ERROR "MODEL must be ALL, BasicMLP, LeNet5, ResNet20, VGG11, VGG16, or ResNet34")
    endif()
endif()

if(BUILD_STANDARD_BATCH)
    if(MODE STREQUAL "ALL")
        set(batch_models "ALL|BasicMLP|LeNet5|ResNet20|VGG11|VGG16|ResNet34")
    else()
        set(batch_models "ALL|LeNet5|ResNet20|ResNet34|VGG11|VGG16")
    endif()
    if(NOT MODEL MATCHES "^(${batch_models})$")
        message(FATAL_ERROR "MODEL must be ALL, LeNet5, ResNet20, ResNet34, VGG11 or VGG16 for the batch models")
    endif()
    if(NOT BATCH_SIZE MATCHES "^(ALL|16|32|64|128|256|512)$")
        message(FATAL_ERROR "BATCH_SIZE must be ALL, 16, 32, 64, 128, 256, or 512")
    endif()
endif()

###############################################################################

################################################################################
########### Find OpenFHE (and HEXL if available)
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
# Controllers of the pipelines
################################################################################

# single-input pipelines: sequential; standard batch pipeline: multithreaded

################################################################################
# Build configuration
################################################################################

message(STATUS "Build Configuration:")
message(STATUS "  Mode: ${MODE}")
if(BUILD_STANDARD_SINGLE)
    message(STATUS "  Standard single-input controller: FHEONANNStandardController (sequential)")
endif()
if(BUILD_STANDARD_BATCH)
    message(STATUS "  Standard batch controller: FHEONANNStandardBatchThreadedController (multithreaded)")
endif()

if(BUILD_STANDARD_SINGLE)
    message(STATUS "  Single Model: ${MODEL}")
endif()

if(BUILD_STANDARD_BATCH)
    message(STATUS "  Batch Model: ${MODEL}")
    message(STATUS "  Batch Size: ${BATCH_SIZE} (pipelines: any size at run time)")
endif()

if(MODE STREQUAL "TESTING" OR MODE STREQUAL "ALL")
    message(STATUS "  Testing Source: ${TESTING_SRC}")
    message(STATUS "  Testing Target: ${TESTING_TARGET}")
endif()

if(MODE STREQUAL "TRAINING" OR MODE STREQUAL "ALL")
    message(STATUS "  Training: trainModules, tuneResNet20 (standard layout, multithreaded)")
endif()

################################################################################
# Compiler flags
################################################################################
# The flags OpenFHE was built with (-O3, -fopenmp, MATHBACKEND, ...) come first; flags given on the command line
# (-DCMAKE_CXX_FLAGS=...) are kept after them, and the build type adds its own (Release: -O3 -DNDEBUG).
#
# What makes FHEON programs faster is mostly the OpenFHE build (measured on ResNet20Diagonal, Intel Core Ultra 7 265K, GCC 13,
# OpenFHE 1.5.1: -march=native for OpenFHE or FHEON, LTO, mimalloc, jemalloc, huge pages and OpenMP settings all stay within
# 2% of the default build):
#   * OpenFHE in Release, with OpenMP (WITH_OPENMP=ON, the default)
#   * WITH_NATIVEOPT=ON on the machine that runs the programs (CPUs with AVX-512 gain the most)
#   * WITH_INTEL_HEXL=ON on Intel CPUs with AVX512-IFMA (Ice Lake server and newer): HEXL only speeds up FHEON when OpenFHE
#     itself is built with it, linking it into FHEON alone changes nothing
# The status lines below report how the OpenFHE found here was built.
set(FHEON_OPENFHE_CXX_FLAGS "${OpenFHE_CXX_FLAGS}")
if(NOT FHEON_WERROR)
    string(REGEX REPLACE "(^| )-Werror( |$)" " " FHEON_OPENFHE_CXX_FLAGS "${FHEON_OPENFHE_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${FHEON_OPENFHE_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

if(FHEON_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" FHEON_HAS_MARCH_NATIVE)
    check_cxx_compiler_flag("-mcpu=native" FHEON_HAS_MCPU_NATIVE)
    if(FHEON_HAS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    elseif(FHEON_HAS_MCPU_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    else()
        message(STATUS "FHEON_NATIVE: the compiler supports neither -march=native nor -mcpu=native, the option is ignored")
    endif()
endif()

if(FHEON_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT FHEON_HAS_IPO OUTPUT ipo_output LANGUAGES CXX)
    if(FHEON_HAS_IPO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "FHEON_LTO: link-time optimization is not supported here, the option is ignored")
    endif()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} (${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS_${build_type_upper}})")
message(STATUS "  OpenFHE built with WITH_NATIVEOPT=${OpenFHE_NATIVEOPT}, OpenMP=${OpenFHE_OPENMP}, tcmalloc=${OpenFHE_TCM}")
if(NOT OpenFHE_NATIVEOPT)
    message(STATUS "  (an OpenFHE built with -DWITH_NATIVEOPT=ON on the machine that runs the programs can be faster, on AVX-512 CPUs above all)")
endif()

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
# Libraries
################################################################################

# std::filesystem needs libstdc++fs only with GCC older than 9 (and the library does not exist with libc++)
set(FHEON_FS_LIBRARY "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    set(FHEON_FS_LIBRARY stdc++fs)
endif()

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES} ${FHEON_FS_LIBRARY})
else()
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS}")
    link_libraries(${OpenFHE_SHARED_LIBRARIES} ${FHEON_FS_LIBRARY})
    # the programs load the OpenFHE they were built against (another OpenFHE earlier in the loader path, e.g. the system one,
    # is not picked up silently); LD_LIBRARY_PATH still takes precedence
    list(APPEND CMAKE_BUILD_RPATH "${OpenFHE_LIBDIR}")
    list(APPEND CMAKE_INSTALL_RPATH "${OpenFHE_LIBDIR}")
endif()

if(TARGET HEXL::hexl)
    link_libraries(HEXL::hexl)
endif()

################################################################################
# Controllers, compiled once
################################################################################
# The controller sources do not depend on FHEON_THREADED (only the controller alias of the batch header does), so every
# source the enabled pipelines need is compiled once into fheon_controllers. A pipeline is an interface library on top of it
# that defines FHEON_THREADED for its models when it is multithreaded (the standard batch pipeline).
set(FHEON_CONTROLLER_SOURCES src/FHEONHEController.cpp)
if(BUILD_STANDARD_SINGLE)
    list(APPEND FHEON_CONTROLLER_SOURCES
        src/FHEONANNController.cpp
        src/FHEONANNStandardController.cpp
        src/FHEONANNDiagonalController.cpp
    )
endif()
if(BUILD_STANDARD_BATCH)
    list(APPEND FHEON_CONTROLLER_SOURCES src/FHEONANNBatchController.cpp)
endif()
list(REMOVE_DUPLICATES FHEON_CONTROLLER_SOURCES)
if(BUILD_STANDARD_SINGLE OR BUILD_STANDARD_BATCH)
    add_library(fheon_controllers STATIC ${FHEON_CONTROLLER_SOURCES})
endif()

# fheon_pipeline_library(<name> <threaded>): the controllers of a pipeline for its models
function(fheon_pipeline_library name threaded)
    add_library(${name} INTERFACE)
    target_link_libraries(${name} INTERFACE fheon_controllers)
    if(threaded)
        target_compile_definitions(${name} INTERFACE FHEON_THREADED)
    endif()
endfunction()

# fheon_model_selected(<file> <result>): whether MODEL selects a model source, by the base name of the file
# (ResNet20 for ResNet20Optimized.cpp, BResNet20N64.cpp or ResNet20Diagonal.cpp; BasicMLP for any MLP)
function(fheon_model_selected file result)
    string(REPLACE ".cpp" "" model_name ${file})
    string(REGEX REPLACE "^B([A-Z])" "\\1" model_name ${model_name})
    string(REGEX MATCH "^[A-Za-z]+[0-9]+" base_name ${model_name})
    if(model_name MATCHES "MLP")
        set(base_name "BasicMLP")
    endif()
    if(MODEL STREQUAL "ALL" OR MODEL STREQUAL base_name)
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()

################################################################################
############################## SINGLE INPUT MODELS #############################
################################################################################
# fheon_model_selected filters by the base name of the model (e.g. "ResNet20" for ResNet20Optimized.cpp, "BasicMLP" for any MLP).

if(BUILD_STANDARD_SINGLE)
    fheon_pipeline_library(fheon_standard OFF)

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
        fheon_model_selected(${model} should_build)
        if(should_build)
            string(REPLACE ".cpp" "" model_name ${model})
            add_executable(${model_name} HESingleModels/${model})
            target_link_libraries(${model_name} fheon_standard)
            target_compile_definitions(${model_name} PRIVATE DEFAULT_TEST_SIZE=${TEST_SIZE})
        endif()
    endforeach()

    set(HEDiagonalSingleModels_SOURCES
        MLPDiagonal.cpp
        LeNet5Diagonal.cpp
        LeNet5KeyOptimizedDiagonal.cpp
        ResNet20Diagonal.cpp
        ResNet20OptimizedDiagonal.cpp
        ResNet34Diagonal.cpp
        VGG11Diagonal.cpp
        VGG16Diagonal.cpp
    )
    foreach(model ${HEDiagonalSingleModels_SOURCES})
        fheon_model_selected(${model} should_build)
        if(should_build)
            string(REPLACE ".cpp" "" model_name ${model})
            if(NOT model_name MATCHES "Diagonal$")
                set(model_name "${model_name}Diagonal")
            endif()
            add_executable(${model_name} HEDiagonalSingleModels/${model})
            target_link_libraries(${model_name} fheon_standard)
            target_compile_definitions(${model_name} PRIVATE DEFAULT_TEST_SIZE=${TEST_SIZE})
        endif()
    endforeach()
endif()

###############################################################################
############################# BATCH MODELS ####################################
###############################################################################

if(BUILD_STANDARD_BATCH)
    fheon_pipeline_library(fheon_batch ON)

    # fixed-batch models: T<model>N<batch size>[variant]
    set(HEBatchModels_SOURCES
        TResNet20N16.cpp
        TResNet20N16JoinedKeys.cpp
        TResNet20N64.cpp
        TResNet20N64JoinedKeys.cpp
        TResNet20N256.cpp
        TResNet20N256JoinedKeys.cpp
        TResNet20N32.cpp
        TResNet20N128.cpp
        TResNet20N512.cpp
        TResNet34N16.cpp
        TResNet34N16JoinedKeys.cpp
        TResNet34N64.cpp
        TResNet34N128.cpp
        TResNet34N32.cpp
        TResNet34N128P2.cpp
        TResNet34N256.cpp
    )
    foreach(model ${HEBatchModels_SOURCES})
        string(REPLACE ".cpp" "" model_name ${model})
        string(REGEX MATCH "N[0-9]+" batch_num ${model_name})
        string(REGEX MATCH "[0-9]+" extracted_batch_size ${batch_num})
        fheon_model_selected(${model} should_build)
        if(should_build AND (BATCH_SIZE STREQUAL "ALL" OR extracted_batch_size STREQUAL BATCH_SIZE))
            add_executable(${model_name} HEBatchModels/${model})
            target_link_libraries(${model_name} fheon_batch)
            target_compile_definitions(${model_name} PRIVATE DEFAULT_TEST_SIZE=${extracted_batch_size})
        endif()
    endforeach()
endif()

#################################################################################

# -------------------------
# TRAINING MODELS
# -------------------------
# Encrypted training in the standard layout (HETrainingModels/):
#   trainModules : the validation modules, the per-module forward and backward passes checked against a plaintext
#                  reference (TrainingModules.cpp), on the controllers trainModel uses
#   tuneResNet20 : one ResNet-20 fine-tuning step (forward, cross-entropy loss, backward, SGD) on a batch of 16 encrypted
#                  CIFAR-10 images, with the multithreaded standard training controllers

if(MODE STREQUAL "TRAINING" OR MODE STREQUAL "ALL")

    set(COMMON_SRCS
        include/Utils.h
        include/UtilsData.h
        include/FHEONHEController.h
        src/FHEONHEController.cpp
        include/FHEONANNController.h
        src/FHEONANNController.cpp
        src/FHEONTrainForwardPassController.cpp
        src/FHEONTrainSupportController.cpp
        src/FHEONTrainBackwardPassController.cpp
    )

    # Validation modules
    add_executable(trainModules
        HETrainingModels/TrainingModules.cpp
        ${COMMON_SRCS}
    )
    target_compile_definitions(trainModules PRIVATE DEFAULT_TEST_SIZE=16)

    # Multithreaded standard training controllers: FHEONTrainStandard{Forward,Backward}PassController, derived from the
    # controllers above, compiled once into a library
    add_library(fheon_training_threaded STATIC
        ${COMMON_SRCS}
        src/FHEONTrainStandardForwardPassController.cpp
        src/FHEONTrainStandardBackwardPassController.cpp
    )

    # ResNet-20 fine-tuning, standard layout (pretrained CSVs of the weights folder, see tuneResNet20.cpp)
    add_executable(tuneResNet20 HETrainingModels/tuneResNet20.cpp)
    target_link_libraries(tuneResNet20 fheon_training_threaded)
    target_compile_definitions(tuneResNet20 PRIVATE DEFAULT_TEST_SIZE=16)

endif()

