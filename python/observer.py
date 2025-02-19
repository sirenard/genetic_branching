import ecole
import pyscipopt
import my_module

from boundml.observers import Observer

from observation import StaticObservation, TreeObservation, Observation, DynamicObservation


class MyObserver(Observer):
    def __init__(self):
        super().__init__()
        self.static_observations = {}
        self.first = True

    def reset(self, instance_path, seed=None):
        self.first = True
        self.static_observations = {}

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())
        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])


        tree_observations = TreeObservation(model)

        res = [None] * n_vars

        for index in prob_indexes:
            if index not in self.static_observations:
                self.static_observations[index] = StaticObservation(model, index)

            dynamic_obsevation = DynamicObservation(model, index)
            res[index] = [val for val in Observation(self.static_observations[index], tree_observations, dynamic_obsevation)]
            # res[index] = [val for val in Observation(None, tree_observations, dynamic_obsevation)]

        return res

    def __len__(self):
        return 21

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

