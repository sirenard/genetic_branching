//
// Created by simon on 14/02/25.
//

#ifndef ARRAYVIEW_H
#define ARRAYVIEW_H



// Wrapper class for a C-style array
template <typename T>
class ArrayView {
    T* data; // Pointer to the raw array
    int size;
public:
    // Constructor to initialize with a raw array
    ArrayView(T* arr, int size);

    // Begin and end functions for range-based for loop
    T* begin();
    T* end();

    // Const versions for read-only access
    // const T* begin();
    // const T* end();
};



#endif //ARRAYVIEW_H
