import ecole
import pyscipopt
import my_module

from boundml.observers import Observer

from observation import Observation, ObservationWrapper


class MyObserver(Observer):
    def __init__(self):
        super().__init__()
        self.static_observations = {}
        self.tree_observation = None
        self.dynamic_observations = {}
        self.first = True

    def before_reset(self, instance_path, seed=None):
        self.static_observations = {}
        self.tree_observation = None
        self.dynamic_observations = {}
        self.first = True

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        p = m.to_ptr(False)
        if self.first:
            self.tree_observation = ObservationWrapper(my_module.TreeFeaturesObs, p)
            self.first = False

        self.tree_observation.reset()

        res = [None] * n_vars

        for index in prob_indexes:
            if index not in self.static_observations:
                self.static_observations[index] = ObservationWrapper(my_module.StaticFeaturesObs, p)
                self.dynamic_observations[index] = ObservationWrapper(my_module.DynamicFeaturesObs, p)

            self.dynamic_observations[index].reset()

            self.static_observations[index].set_var(index)
            self.dynamic_observations[index].set_var(index)

            res[index] = [val for val in Observation(self.static_observations[index], self.tree_observation, self.dynamic_observations[index])]

        return res

    def __len__(self):
        return my_module.StaticFeaturesObs.size() + my_module.DynamicFeaturesObs.size() + my_module.TreeFeaturesObs.size()
        # return my_module.DynamicFeaturesObs.size() + my_module.TreeFeaturesObs.size()

class KhalilObserver(Observer):
    def __init__(self):
        super().__init__()
        self.obs = ecole.observation.Khalil2016()

    def reset(self, instance_path, seed=None):
        self.obs.reset(instance_path, seed)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())
        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        obs = self.obs.extract(model, done).features

        res = [None] * n_vars

        for index in prob_indexes:
            res[index] = [float(v) for v in obs[index]]
        return res

    def __len__(self):
        return 72

    def __getstate__(self):
        return []

    def __setstate__(self, _):
        self.__init__()

