#include "StaticFeaturesObs.h"

StaticFeaturesObs::StaticFeaturesObs(SCIP *scip): Obs(scip, size) {}

StaticFeaturesObs::StaticFeaturesObs(py::object py_scip) : StaticFeaturesObs(
        static_cast<SCIP *>(PyCapsule_GetPointer(py_scip.ptr(), "scip"))
    ) {
    if (!scip) {
        throw py::error_already_set();
    }
}

std::vector<double> StaticFeaturesObs::computeObjCoefficient() {
    return {(SCIPvarGetObj(var))};
}

std::vector<double> StaticFeaturesObs::computeNonZeroCoefficientsStatistics() {
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

std::vector<double> StaticFeaturesObs::computeConstraintsDegreeStatistics() {
    std::vector<double> degrees;
    auto col = SCIPvarGetCol(var);
    auto const n_rows = SCIPcolGetNNonz(col);
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
    int start = 0;
    if (index < 1) {
        tmp = computeObjCoefficient();
    } else if (index < 10) {
        start = 1;
        tmp = computeNonZeroCoefficientsStatistics();
    } else if (index < 14) {
        start = 10;
        tmp = computeConstraintsDegreeStatistics();
    }

    for (int i = 0; i < tmp.size(); i++) {
        features[i + start] = tmp[i];
        computed[i + start] = true;
    }
}