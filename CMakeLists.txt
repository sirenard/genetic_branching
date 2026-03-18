cmake_minimum_required(VERSION 3.15)
project(GeneticBranching)

option(ENABLE_PYTHON_MODULE "Enable python features" ON)

if(ENABLE_PYTHON_MODULE)
    set(MACRO_STATUS "#define USE_PYTHON")
    message(Prep python module)

else()
    set(MACRO_STATUS "/* #undef USE_PYTHON */")
    message(Prep lib)
endif()


configure_file(
        "${PROJECT_SOURCE_DIR}/src/config.h.in"
        "${PROJECT_BINARY_DIR}/src/config.h"
)

find_package(scip REQUIRED)
set(SOURCE_FILES
        src/ArrayView.cpp
        src/DynamicFeaturesObs.cpp
        src/StaticFeaturesObs.cpp
        src/statistics.cpp
        src/TreeFeaturesObs.cpp
        src/utils.cpp
)

# 3. Conditional Target Definition
if(ENABLE_PYTHON_MODULE)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)

    # Build as a Python Module
    pybind11_add_module(gb_features ${SOURCE_FILES} src/gb_features.cpp)

    install(TARGETS gb_features LIBRARY DESTINATION ${Python3_SITEARCH})
else()
    # Build as a Static or Shared Library for other C++ projects
    add_library(gb_features STATIC ${SOURCE_FILES})
    set_target_properties(gb_features PROPERTIES POSITION_INDEPENDENT_CODE ON) # Add fPIC flag

    file(GLOB MY_HEADERS "${PROJECT_SOURCE_DIR}/src/*.h")
    list(APPEND MY_HEADERS "${PROJECT_BINARY_DIR}/src/config.h")

    # 3. Set the PUBLIC_HEADER property using the list
    set_target_properties(gb_features PROPERTIES
            PUBLIC_HEADER "${MY_HEADERS}"
    )

    # Standard installation for a C++ library
    install(TARGETS gb_features
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            PUBLIC_HEADER DESTINATION include
    )

    target_compile_options(gb_features PRIVATE
            -fno-exceptions
            -fno-rtti
    )
endif()

target_compile_options(gb_features PRIVATE
        -march=native
        # DO NOT add -Ofast or -ffast-math here!
)

# Enable Link-Time Optimization (LTO) the "Modern CMake" way.
# This is much safer than passing -flto manually, as CMake checks if the compiler supports it first.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set_property(TARGET gb_features PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO / LTO is not supported by your compiler: ${error}")
endif()

# 4. Common Requirements for both Target Types
target_include_directories(gb_features PUBLIC
        "${PROJECT_BINARY_DIR}/src"      # To find config.h
)
target_link_libraries(gb_features PUBLIC scip)