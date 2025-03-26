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
    virtual void compute(int index)=0;
    SCIP* scip;
    SCIP_Var* var {};
public:
    Obs(SCIP* scip, int size);

    double operator[](int index);
    void reset();
    void setVar(int probIndex);

    virtual ~Obs()=default;
};


#endif //OBS_H
