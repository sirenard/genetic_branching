import pickle

import ecole
from boundml.evaluation_tools import evaluate_solvers, SolverEvaluationResults
from boundml.solvers import ClassicSolver

from ClassicSolverCustomBranching import ClassicSolverCustomBranching

n_instances = 10

if __name__ == '__main__':
    instances = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
    # instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100)
    instances.seed(1212)

    scip_params = {
        "limits/time": 300,
    }

    custom_solvers = [
        ClassicSolverCustomBranching("best_ca", scip_params),
        ClassicSolverCustomBranching("best_ca_10_10", scip_params),
        ClassicSolverCustomBranching("best_ca_10_30", scip_params),
    ]

    for solver in custom_solvers:
        print(solver.get_expression())

    solvers = [
        ClassicSolver("relpscost", scip_params),
    ]

    solvers.extend(custom_solvers)

    metrics = ["nnodes", "time", "gap"]

    data = evaluate_solvers(solvers, instances, n_instances, metrics)

    r = data.report(
        SolverEvaluationResults.sg_metric("nnodes", 10),
        SolverEvaluationResults.sg_metric("time", 1),
        SolverEvaluationResults.nwins("time"),
        SolverEvaluationResults.nsolved(),
        SolverEvaluationResults.auc_score("time"))
    print(r)

    pickle.dump(data, open("data", "wb"))