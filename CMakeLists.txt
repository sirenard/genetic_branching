cmake_minimum_required(VERSION 3.15)
project(MyProject)


find_package(pybind11)
find_package(scip)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(${SCIP_INCLUDE_DIRS})
message("SCIP include: ${SCIP_INCLUDE_DIRS}")

FILE(GLOB SOURCES src/*.cpp)

pybind11_add_module(my_module ${SOURCES})

install(TARGETS my_module
        LIBRARY DESTINATION ${Python3_SITEARCH}  # Install to site-packages
)

target_link_libraries(my_module PUBLIC scip)

#add_executable(my_module ${SOURCES})
