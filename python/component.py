import ecole
import pyscipopt
import my_module

from boundml.components import BranchingComponent
from pyscipopt import Model, SCIP_RESULT

from observation import Observation, ObservationWrapper


class CustomComponent(BranchingComponent):
    def __init__(self):
        super().__init__()
        self.static_observations = {}
        self.tree_observation = None
        self.dynamic_observations = {}
        self.first = True

    def reset(self, model: Model) -> None:
        self.static_observations = {}
        self.tree_observation = None
        self.dynamic_observations = {}
        self.first = True

    def callback(self, model: Model, passive: bool=True) -> SCIP_RESULT:
        candidates, *_ = model.getLPBranchCands()

        p = model.to_ptr(False)
        if self.first:
            self.tree_observation = ObservationWrapper(my_module.TreeFeaturesObs, p)
            self.first = False

        self.tree_observation.reset()

        self.observation = [None] * len(candidates)

        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            index = var.getCol().getLPPos()
            if index not in self.static_observations:
                self.static_observations[index] = ObservationWrapper(my_module.StaticFeaturesObs, p)
                self.dynamic_observations[index] = ObservationWrapper(my_module.DynamicFeaturesObs, p)

            self.dynamic_observations[index].reset()

            self.static_observations[index].set_var(index)
            self.dynamic_observations[index].set_var(index)

            self.observation[i] = [val for val in Observation(self.static_observations[index], self.tree_observation, self.dynamic_observations[index])]

        return SCIP_RESULT.DIDNOTRUN

    def __len__(self):
        return my_module.StaticFeaturesObs.size() + my_module.DynamicFeaturesObs.size() + my_module.TreeFeaturesObs.size()
        # return my_module.DynamicFeaturesObs.size() + my_module.TreeFeaturesObs.size()