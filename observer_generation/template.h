#pragma once

#include <iostream>
#include <map>
#include <algorithm>
#include <memory>
#include <objscip/objbranchrule.h>
#include <scip/scip.h>

#include "DynamicFeaturesObs.h"
#include "StaticFeaturesObs.h"
#include "TreeFeaturesObs.h"

class template_name : public scip::ObjBranchrule {
  std::vector<std::unique_ptr<StaticFeaturesObs>> static_features;
  std::unique_ptr<TreeFeaturesObs> tree_features;

  class FeaturesWrapper {
    StaticFeaturesObs &staticFeatures;
    TreeFeaturesObs &treeFeatures;
    DynamicFeaturesObs &dynamicFeatures;

  public:
    FeaturesWrapper(StaticFeaturesObs &staticFeatures,
                    TreeFeaturesObs &treeFeatures,
                    DynamicFeaturesObs &dynamicFeatures);

    double operator[](int index);
  };

public:
  template_name(SCIP *scip, int priority = 0);

  SCIP_DECL_BRANCHINITSOL(scip_initsol) override;
  SCIP_DECL_BRANCHEXECLP(scip_execlp) override;
  SCIP_DECL_BRANCHEXECPS(scip_execps) override;
  SCIP_DECL_BRANCHEXITSOL(scip_exitsol) override;
};

void include_template_name(SCIP *scip, int priority=0);