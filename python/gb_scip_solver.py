import importlib
import sys

from boundml.solvers import DefaultScipSolver


class GbScipSolver(DefaultScipSolver):
    LIB_PATH = "observer_generation/lib"
    def __init__(self, branching_rule_name, scip_params={}):
        sys.path.append(GbScipSolver.LIB_PATH)
        module = importlib.import_module(branching_rule_name)
        self.expression = module.to_str()
        sys.path.remove(GbScipSolver.LIB_PATH)
        super().__init__(
            branching_rule_name,
            scip_params,
            configure=lambda model: module.add_branching(model.to_ptr(False))
        )

        self.state = (branching_rule_name, scip_params)

    @staticmethod
    def set_lib_path(lib_path):
        GbScipSolver.LIB_PATH = lib_path

    def get_expression(self):
        return self.expression

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state)