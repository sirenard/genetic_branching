//
// Created by simon on 21/02/25.
//

#include "utils.h"

template<typename T>
T safe_div(T a, T b) {
    if (b == 0) return 0;
    return a / b;
}

template double safe_div<double>(double a, double b);

