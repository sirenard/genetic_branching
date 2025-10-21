import ecole
import numpy as np
import pyscipopt
from boundml.components import ScoringBranchingStrategy, BranchingComponent
from pyscipopt import Model, SCIP_RESULT


class FunctorComponent(ScoringBranchingStrategy):
    def __init__(self, scoring_functor, feature_observer: BranchingComponent):
        super().__init__()
        self.scoring_functor = scoring_functor
        self.feature_observer = feature_observer

    def reset(self, model: Model) -> None:
        self.feature_observer.reset(model)

    def compute_scores(self, model: Model):
        scores = super().compute_scores(model)
        candidates, *_ = model.getLPBranchCands()

        self.feature_observer.callback(model, passive=True)
        features = self.feature_observer.observation

        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            scores[i] = self.scoring_functor(*features[i])

        return scores

    def done(self, model: Model) -> None:
        self.feature_observer.done(model)

    def __str__(self):
        return "Func"
