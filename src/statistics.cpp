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

    float delta = value - min;
    mean += delta / n;
    stdev += delta * (value - min);
  }

  if(n<2){
    stdev = 0;
  } else{
    stdev = std::sqrt(stdev/n);
  }

  count = n;
}

template<typename T, typename N> statistics<T, N>::statistics(T data){
  statistics(data, [](float value){return true;});
}

template class statistics<ArrayView<double>, double>;
template class statistics<std::vector<double>, double>;