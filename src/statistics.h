//
// Created by simon on 14/02/25.
//

#ifndef STATISTICS_H
#define STATISTICS_H

#include <functional>

template <typename T, typename N> struct statistics {
  statistics(T data, std::function<bool(N)> func);
  statistics(T data);
  float mean;
  float stdev;
  float min;
  float max;
  int count;
};


#endif //STATISTICS_H
