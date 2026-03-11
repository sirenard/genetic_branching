//
// Created by simon on 20/02/25.
//

#ifndef OBS_H
#define OBS_H

#include <vector>
#include <scip/scip.h>

class Obs {
protected:
    std::vector<double> features;
    std::vector<bool> computed;
    SCIP* scip;
    SCIP_Var* var {};

    virtual void compute(int index)=0;

    template<size_t S>
    void assign_features(const std::array<double, S>& tmp, int start ) {
        for (size_t i = 0; i < S; i++) {
            features[i + start] = tmp[i];
            computed[i + start] = true;
        }
    }
public:
    Obs(SCIP* scip, int size);

    double operator[](int index);
    void reset();
    void setVar(int probIndex);

    virtual ~Obs()=default;
};


#endif //OBS_H
