//
// Created by simon on 14/02/25.
//

#include "ArrayView.h"

#include <iostream>
#include <ostream>


template<typename T>
ArrayView<T>::ArrayView(T* arr, int size) : data(arr), size(size) {}

// Begin and end functions for range-based for loop
template<typename T>
T* ArrayView<T>::begin() { return data; }

template<typename T>
T* ArrayView<T>::end() { return data + size; }

template class ArrayView<double>;
// Const versions for read-only access
//template<typename T>
//const T* ArrayView<T>::begin() { return data; }
//
//template<typename T>
//T* ArrayView<T>::end() { return data + size; }
