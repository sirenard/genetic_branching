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
LDFLAGS := -L$(SCIP_LIB) -lscip -lreadline -lgmp -lm

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

# -----------------------------------------------------------
# 1. Compile Static Objects
# -----------------------------------------------------------
./build/%.o: ../src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -----------------------------------------------------------
# 2. Generate CPP File (The Fix)
# -----------------------------------------------------------
# This rule tells Make: if we need 'build/X.cpp', look for '../individuals/X'
# We use $< to represent the dependency (the individual file)
./build/%.cpp: ../individuals/%
	@mkdir -p $(@D)
	$(PYTHON) $(THIS_DIR)/../python/generator.py -i $< -o ./build

# -----------------------------------------------------------
# 3. Main Target (Link Everything)
# -----------------------------------------------------------
# Usage: make best_0
# This matches any target name (%), checks if a .cpp exists in build,
# and compiles it along with the static OBJS.
%: ./build/%.cpp $(OBJS)
	@mkdir -p lib
	$(CXX) $(CXXFLAGS) $^ -o ./lib/$@$(TARGET_SUFFIX) $(LDFLAGS)

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
clean:
	rm -f ./build/*
	rm -f ./lib/*$(TARGET_SUFFIX)

.PHONY: clean
