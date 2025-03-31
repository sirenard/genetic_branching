import pickle

import ecole
from boundml.evaluation_tools import evaluate_solvers
from boundml.solvers import ClassicSolver

from ClassicSolverCustomBranching import ClassicSolverCustomBranching

n_instances = 15

if __name__ == '__main__':
    instances = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
    instances.seed(0)

    scip_params = {
        "limits/time": 200,
    }

    solvers = [
        ClassicSolver("relpscost", scip_params),
        ClassicSolverCustomBranching("best_ca", scip_params),
    ]

    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)
    pickle.dump(data, open("data", "wb"))