import importlib
import sys

from boundml.solvers import ClassicSolver


class ClassicSolverCustomBranching(ClassicSolver):
    def __init__(self, branching_rule_name, scip_params={}):
        sys.path.append("observer_generation")
        module = importlib.import_module(branching_rule_name)
        super().__init__(
            branching_rule_name,
            scip_params,
            config_function=lambda model: module.add_branching(model.to_ptr(False))
        )