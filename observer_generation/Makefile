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


# Pattern rule to build any module
%: %.cpp
	$(CXX) $(CXXFLAGS) $< $(SOURCES) -o $@$(TARGET_SUFFIX) $(LDFLAGS)


# Clean rule
clean:
	rm -f *$(TARGET_SUFFIX) *.so

.PHONY: clean