import importlib
import sys

from boundml.solvers import ClassicSolver


class ClassicSolverCustomBranching(ClassicSolver):
    def __init__(self, branching_rule_name, scip_params={}):
        sys.path.append("observer_generation")
        self.module = importlib.import_module(branching_rule_name)
        sys.path.remove("observer_generation")
        super().__init__(
            branching_rule_name,
            scip_params,
            config_function=lambda model: self.module.add_branching(model.to_ptr(False))
        )

        self.state = (branching_rule_name, scip_params)

    def get_expression(self):
        return self.module.to_str()

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state)