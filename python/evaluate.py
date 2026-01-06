import argparse
import pickle

import ecole
import torch
from boundml.components import EcoleComponent
from boundml.components.gnn_component import GnnBranching
from boundml.evaluation import evaluate_solvers, SolverEvaluationResults

from boundml.solvers import DefaultScipSolver, ModularSolver

from boundml.instances import *

from gb_scip_solver import GbScipSolver
from folder_instance import FolderInstanceGenerator

if __name__ == '__main__':
    import multiprocess.context as ctx

    ctx._force_start_method('spawn')

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Evaluate a set of solvers on an instance generator")

    # Solvers
    parser.add_argument("--classic_solvers", nargs='+', default=[], type=str,
                        help="List of branching strategies names to use (e.g. 'relpscost', 'pscost'")
    parser.add_argument("--genetic_solvers", nargs='+', default=[], type=str,
                        help="List of branching strategies name to use that use genetic branching. It must consist of a python module in folder observer_generation/lib")
    parser.add_argument("--gnn_solvers", nargs='+', default=[], type=str,
                        help="List of GNN solvers")

    # Instances
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy", type=str,
                        help="Difficulty of the instances. Useful only if --instances is provided")
    parser.add_argument("--seed", default=-1, type=int, help="Seed of the instance generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instances", type=str, help="Type of instances to use")
    group.add_argument("--external_instances", type=str, help="Pickle file of a custom instance generator")
    group.add_argument("--folder_instances", type=str, help="Folder containing instances to use")

    parser.add_argument("--n_instances", type=int, help="Number of instances to solve", required=True)

    parser.add_argument("--ncpu", type=int, default=1, help="Number of CPU cores to use")
    parser.add_argument("--time", type=int, default=300, help="Time limit for each instance")
    parser.add_argument("--out", type=str, help="Pickle output file of the resulting SolverEvaluationResult")

    args = parser.parse_args()

    if args.instances is not None:
        match args.instances:
            case "ca":
                kwargs = {
                    "easy": {"n_items": 200, "n_bids": 1000},
                    "medium": {"n_items": 300, "n_bids": 1500},
                    "hard": {"n_items": 400, "n_bids": 2000}
                }[args.difficulty]
                instances = CombinatorialAuctionGenerator(**kwargs)
            case "cfl":
                kwargs = {
                    "easy": {"n_customers": 200},
                    "medium": {"n_customers": 600},
                    "hard": {"n_customers": 1000},
                }[args.difficulty]
                instances = CapacitatedFacilityLocationGenerator(**kwargs)
            case "sc":
                kwargs = {
                    "easy": {"n_rows": 1000, "n_cols": 1000},
                    "medium": {"n_rows": 1500, "n_cols": 1000},
                    "hard": {"n_rows": 2000, "n_cols": 1000},
                }[args.difficulty]
                instances = SetCoverGenerator(**kwargs)
            case "mis":
                kwargs = {
                    "easy": {"n_nodes": 500},
                    "medium": {"n_nodes": 1000},
                    "hard": {"n_nodes": 1500},
                }[args.difficulty]
                instances = IndependentSetGenerator(**kwargs)
            case _:
                instances = FolderInstances(args.instances)
    elif args.external_instances:
        instances = pickle.load(open(args.external_instances, "rb"))
    else:
        instances = FolderInstanceGenerator(args.folder_instances)

    if args.seed is not None:
        instances.seed(1212)

    scip_params = {
        #"timing/clocktype": 1,
        "limits/time": args.time,
        "estimation/method": "c",
        "estimation/completiontype": "m"
    }


    custom_solvers = [GbScipSolver(name, scip_params) for name in args.genetic_solvers]

    for solver in custom_solvers:
        print(solver.get_expression())

    solvers = [DefaultScipSolver(name, scip_params) for name in args.classic_solvers]

    external_solvers = []

    for path in args.gnn_solvers:
        component = GnnBranching(
            path,
            EcoleComponent(ecole.observation.NodeBipartite()),
            try_use_gpu=True
        )

        external_solvers.append(
            ModularSolver(
                component,
                scip_params=scip_params
            )
        )

    solvers.extend(custom_solvers)
    solvers.extend(external_solvers)

    metrics = ["nnodes", "time", "gap", "obj"]


    data = evaluate_solvers(solvers, instances, args.n_instances, metrics, args.ncpu)


    if args.out is not None:
        pickle.dump(data, open(args.out, "wb"))
