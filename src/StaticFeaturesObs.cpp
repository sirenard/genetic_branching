#include "StaticFeaturesObs.h"

#include "ArrayView.h"

StaticFeaturesObs::StaticFeaturesObs(SCIP *scip): Obs(scip) {}

#ifdef USE_PYTHON
StaticFeaturesObs::StaticFeaturesObs(py::object py_scip) : StaticFeaturesObs(
        static_cast<SCIP *>(PyCapsule_GetPointer(py_scip.ptr(), "scip"))
    ) {
    if (!scip) {
        throw py::error_already_set();
    }
}
#endif

std::array<double, 1> StaticFeaturesObs::computeObjCoefficient() {
    return {(SCIPvarGetObj(var))};
}

std::array<double, 9> StaticFeaturesObs::computeNonZeroCoefficientsStatistics() {
    auto col = SCIPvarGetCol(var);
    int count = SCIPcolGetNLPNonz(col);
    auto data = ArrayView(SCIPcolGetVals(col), count);

    auto positiveStats = statistics<ArrayView<double>, double>(data, [](double val) { return val > 0;; });
    auto negativeStats = statistics<ArrayView<double>, double>(data, [](double val) { return val < 0; });

    return {
        static_cast<double>(count),
        positiveStats.mean,
        positiveStats.stdev,
        positiveStats.min,
        positiveStats.max,
        negativeStats.mean,
        negativeStats.stdev,
        negativeStats.min,
        negativeStats.max,
    };
}

std::array<double, 4> StaticFeaturesObs::computeConstraintsDegreeStatistics() {
    auto col = SCIPvarGetCol(var);
    auto const n_rows = SCIPcolGetNNonz(col);
    std::vector<double> degrees(n_rows);
    auto rows = SCIPcolGetRows(col);

    for (int i = 0; i < n_rows; i++) {
        auto row = rows[i];
        degrees.push_back(SCIProwGetNNonz(row));
    }

    auto stats = statistics<std::vector<double>, double>(degrees);

    return {
        stats.mean,
        stats.stdev,
        stats.min,
        stats.max,
    };
}

void StaticFeaturesObs::compute(int index) {
    std::vector<double> tmp;
    if (index < 1) {
        assign_features(computeObjCoefficient(), 0);
    } else if (index < 10) {
        assign_features(computeNonZeroCoefficientsStatistics(), 1);
    } else if (index < 14) {
        assign_features(computeConstraintsDegreeStatistics(), 10);
    }
}