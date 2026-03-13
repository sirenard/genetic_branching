cmake_minimum_required(VERSION 3.15)
project(MyProject)

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
    src/Obs.cpp
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
    pybind11_add_module(my_module ${SOURCE_FILES} src/my_module.cpp)

    install(TARGETS my_module LIBRARY DESTINATION ${Python3_SITEARCH})
else()
    # Build as a Static or Shared Library for other C++ projects
    add_library(my_module STATIC ${SOURCE_FILES})
    set_target_properties(my_module PROPERTIES POSITION_INDEPENDENT_CODE ON) # Add fPIC flag
    # Standard installation for a C++ library
    install(TARGETS my_module
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
endif()

# 4. Common Requirements for both Target Types
target_include_directories(my_module PUBLIC
        "${PROJECT_BINARY_DIR}/src"      # To find config.h
)
target_link_libraries(my_module PUBLIC scip)