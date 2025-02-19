import ecole
import my_module
from objproxies import LazyProxy


class StaticObservation:
    size = 14
    def __init__(self, model: ecole.scip.Model, prob_index):
        self.data = [None] * self.size
        self.observation = my_module.StaticFeaturesObs(model.get_scip_ptr(), prob_index)

    def compute_feature(self, i):
        if self.data[i] is not None:
            return

        if i < 1:
            self.data[0:1] = self.observation.computeObjCoefficient()
        elif i < 10:
            self.data[1:10] = self.observation.computeNonZeroCoefficientsStatistics()
        elif i < 14:
            self.data[10:14] = self.observation.computeConstraintsDegreeStatistics()

    def __getitem__(self, i):
        self.compute_feature(i)
        return self.data[i]

    def __len__(self):
        return self.size

class TreeObservation:
    size = 4
    def __init__(self, model: ecole.scip.Model):
        observation = my_module.TreeFeaturesObs(model.get_scip_ptr())
        self.data = observation.computeFeatures()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return self.size

class DynamicObservation:
    size = 9
    def __init__(self, model: ecole.scip.Model, prob_index: int):
        self.observation = my_module.DynamicFeaturesObs(model.get_scip_ptr(), prob_index)
        self.data = [None] * self.size

    def compute_feature(self, i):
        if self.data[i] is not None:
            return

        if i < 3:
            self.data[:3] = self.observation.getPseudoCosts()
        elif i < 7:
            self.data[3:7] = self.observation.getInfeasibilityStatistics()
        elif i < 14:
            self.data[7:14] = self.observation.getStrongBranchingScore()

    def __getitem__(self, i):
        self.compute_feature(i)
        return self.data[i]

    def __len__(self):
        return self.size

class Observation:
    size = StaticObservation.size + TreeObservation.size + DynamicObservation.size
    def __init__(self, static_observation, tree_observation, dynamic_observation):
        self.static_observations = static_observation
        self.tree_observation = tree_observation
        self.dynamic_observation = dynamic_observation

    def __getitem__(self, i):
        for observation in [self.static_observations, self.tree_observation, self.dynamic_observation]:
            if i < len(observation):
                return LazyProxy(lambda :observation[i])

            i -= len(observation)

        raise IndexError

    def __len__(self):
        return len(self.static_observations) + len(self.tree_observation) + len(self.dynamic_observation)
