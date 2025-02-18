import ecole
import numpy as np
import pyscipopt
from boundml.observers import Observer


class FunctorObserver(Observer):
    def __init__(self, scoring_functor, feature_observer=ecole.observation.Khalil2016()):
        super().__init__()
        self.scoring_functor = scoring_functor
        self.feature_observer = feature_observer

    def before_reset(self, model):
        self.feature_observer.before_reset(model)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        prob_indexes = [var.getCol().getLPPos() for var in candidates]

        res = np.zeros(m.getNVars())
        res[:] = np.nan

        features = self.feature_observer.extract(model, done)
        for i in prob_indexes:
            res[i] = self.scoring_functor(*features[i])

        return res

    def __str__(self):
        return "Func"
