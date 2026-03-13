# Compiler and flags
CXX := g++ -std=c++23
PYTHON := python3

BUILD_PYTHON=ON

FEATURES_INCLUDE=../src/


ifeq ($(BUILD_PYTHON), ON)
	PYBIND11_INCLUDE := $(shell $(PYTHON) -m pybind11 --includes)
	PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
	# Auto-detect Python extension suffix
    TARGET_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
	PYTHON_FLAG=-DUSE_PYTHON=ON
    OUT_FILE = ./lib/$@$(TARGET_SUFFIX)
else
    OUT_FILE = ./lib/$@.so
    PYTHON_FLAG =
endif

SCIP_INCLUDE := ${CONDA_PREFIX}/include/
SCIP_LIB := ${CONDA_PREFIX}/lib
MY_MODULE_PATH=../build



# Compiler flags
CXXFLAGS := $(PYTHON_FLAG) -O3 -Wall -shared -std=c++20 -fPIC $(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE) -I$(SCIP_INCLUDE) -I$(MY_MODULE_PATH)/src -I$(FEATURES_INCLUDE)
LDFLAGS := -L$(SCIP_LIB) -lscip -lreadline -lgmp -lm -L $(MY_MODULE_PATH) -lgb_features

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Disable all implicit rules
.SUFFIXES:
%: %,v
%: RCS/%,v
%: RCS/%
%: s.%
%: SCCS/s.%

.PRECIOUS: ./build/%.o ./src/%.cpp

# -----------------------------------------------------------
# Generate CPP File
# -----------------------------------------------------------
# This rule tells Make: if we need 'src/X.cpp', look for '../individuals/X'
# We use $< to represent the dependency (the individual file)
./src/%.cpp: ../individuals/%
	@mkdir -p $(@D)
	$(PYTHON) $(THIS_DIR)/../python/generator.py -i $< -o ./src

# -----------------------------------------------------------
# Main Target (Link Everything)
# -----------------------------------------------------------
# Usage: make best_0
# This matches any target name (%), checks if a .cpp exists in src,
# and compiles it along with the static OBJS.
%: ./src/%.cpp
	@mkdir -p lib
	$(CXX) $(CXXFLAGS) $^ -o $(OUT_FILE) $(LDFLAGS)

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
clean:
	rm -f ./build/*
	rm -f ./lib/*
	rm -f ./src/*

.PHONY: clean
