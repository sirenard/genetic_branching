//
// Created by simon on 14/02/25.
//
#include <limits>
#include <math.h>

#include "statistics.h"

#include "ArrayView.h"

template<typename T, typename N> statistics<T, N>::statistics(T data, std::function<bool(N)> func): mean(0), stdev(0), min(std::numeric_limits<float>::infinity()), max(-std::numeric_limits<float>::infinity()){
  int n=0;
  for(float value:data){
    if(!func(value)) continue;

    ++n;
    if(value < min) min = value;
    if(value > max) max = value;

    // Correct Welford's Algorithm
    float delta = value - mean;
    mean += delta / n;
    float delta2 = value - mean;
    stdev += delta * delta2;
  }

  if(n < 2){
    stdev = 0;
  } else{
    stdev = std::sqrt(stdev / n);
  }

  // Prevent +/- Infinity from destroying ML models if no elements matched
  if (n == 0) {
    min = 0.0;
    max = 0.0;
  }
  count = n;
}

template<typename T, typename N> statistics<T, N>::statistics(T data){
  statistics(data, [](float value){return true;});
}

template class statistics<ArrayView<double>, double>;
template class statistics<std::vector<double>, double>;