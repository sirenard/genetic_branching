import importlib
import sys

from boundml.solvers import ClassicSolver


class ClassicSolverCustomBranching(ClassicSolver):
    def __init__(self, branching_rule_name, scip_params={}):
        sys.path.append("observer_generation/lib")
        module = importlib.import_module(branching_rule_name)
        self.expression = module.to_str()
        sys.path.remove("observer_generation/lib")
        super().__init__(
            branching_rule_name,
            scip_params,
            config_function=lambda model: module.add_branching(model.to_ptr(False))
        )

        self.state = (branching_rule_name, scip_params)

    def get_expression(self):
        return self.expression

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state)