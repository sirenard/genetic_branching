# Compiler and flags
CXX := g++
PYTHON := python3
PYBIND11_INCLUDE := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
SCIP_INCLUDE := ${CONDA_PREFIX}/include/
SCIP_LIB := ${CONDA_PREFIX}/lib

# Auto-detect Python extension suffix
TARGET_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Compiler flags
CXXFLAGS := -O3 -Wall -shared -std=c++20 -fPIC $(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE) -I$(SCIP_INCLUDE)
LDFLAGS := -L$(SCIP_LIB) -lscip -lreadline -lz -lgmp -lm

SOURCES := ../src/ArrayView.cpp ../src/Obs.cpp ../src/StaticFeaturesObs.cpp ../src/TreeFeaturesObs.cpp ../src/DynamicFeaturesObs.cpp ../src/utils.cpp ../src/statistics.cpp

OBJS := $(patsubst ../src/%.cpp, ./build/%.o, $(SOURCES))

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Disable all implicit rules
.SUFFIXES:
%: %,v
%: RCS/%,v
%: RCS/%
%: s.%
%: SCCS/s.%

# Keep .o files after build
.PRECIOUS: ./build/%.o ./build/%.cpp

./build/%.o: ../src/%.cpp
	@mkdir -p $(@D)  # Ensure dir exists
	$(CXX) $(CXXFLAGS) -c $< -o $@

./build/%.cpp:
	python ${THIS_DIR}/../python/generator.py -i ${THIS_DIR}/../individuals/$* -o ./build

# Pattern rule to build any module
%: ./build/%.cpp $(OBJS)
	@mkdir -p lib
	$(CXX) $(CXXFLAGS) $^ -o ./lib/$@$(TARGET_SUFFIX) $(LDFLAGS)


# Clean rule
clean:
	rm -f ./build/*
	rm -f *$(TARGET_SUFFIX)

.PHONY: clean