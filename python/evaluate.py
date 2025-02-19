import pickle

import ecole
from boundml.evaluation_tools import evaluate_solvers
from boundml.solvers import ClassicSolver, EcoleSolver

from functor_observer import FunctorObserver
from observer import MyObserver
from train import toolbox


n_instances = 15

individial = pickle.load(open("best_gen_khalil", "rb"))
print(individial)
f = toolbox.compile(expr=individial)

if __name__ == '__main__':
    instances = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
    instances.seed(0)

    scip_params = {
        "limits/time": 200,
    }

    solvers = [
        ClassicSolver("relpscost", scip_params),
        ClassicSolver("pscost", scip_params),
        EcoleSolver(FunctorObserver(f, MyObserver()), scip_params),
    ]

    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))