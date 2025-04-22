import argparse
import itertools

import ecole
from mpipool import MPIExecutor
from mpi4py.MPI import COMM_WORLD

from observer import MyObserver
from utils import train, create_tool_box
import pickle

if __name__ == "__main__":
    observer = MyObserver()

    if COMM_WORLD.rank:
        create_tool_box(observer=observer)

    with MPIExecutor() as pool:
        pool.workers_exit()

        parser = argparse.ArgumentParser(description="Evaluate a set of solvers on an instance generator")

        # Instances
        parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy", type=str,
                            help="Difficulty of the instances. Useful only if --instances is provided")
        parser.add_argument("--seed", default=-1, type=int, help="Seed of the instance generator")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--instances", choices=["ca", "cfl", "sc", "mis"], type=str, nargs='+',
                           help="Type of instances to use")
        group.add_argument("--external_instances", type=str, help="Pickle file of a custom instance generator")
        parser.add_argument("--n_instances", type=int, help="Number of instances to solve", required=True)

        parser.add_argument("--time", type=int, default=300, help="Time limit for each instance")
        parser.add_argument("--out", type=str, help="Pickle output file of the resulting SolverEvaluationResult",
                            required=True)

        args = parser.parse_args()

        if args.instances is not None:
            instances = []
            for instances_name in args.instances:
                match instances_name:
                    case "ca":
                        kwargs = {"easy": {"n_items": 100, "n_bids": 500}, "medium": {"n_items": 200, "n_bids": 1000},
                                  "hard": {"n_items": 300, "n_bids": 1500}, }[args.difficulty]
                        instances_gen = ecole.instance.CombinatorialAuctionGenerator(**kwargs)
                    case "cfl":
                        kwargs = \
                        {"easy": {"n_customers": 100}, "medium": {"n_customers": 200}, "hard": {"n_customers": 300}, }[
                            args.difficulty]
                        instances_gen = ecole.instance.CapacitatedFacilityLocationGenerator(**kwargs)
                    case "sc":
                        kwargs = {"easy": {"n_rows": 500, "n_cols": 1000}, "medium": {"n_rows": 1000, "n_cols": 1000},
                                  "hard": {"n_rows": 1500, "n_cols": 1000}, }[args.difficulty]
                        instances_gen = ecole.instance.SetCoverGenerator(**kwargs)
                    case "mis":
                        kwargs = {"easy": {"n_nodes": 500}, "medium": {"n_nodes": 1000}, "hard": {"n_nodes": 1500}, }[
                            args.difficulty]
                        instances_gen = ecole.instance.IndependentSetGenerator(**kwargs)
                    case _:
                        raise NotImplementedError

                if args.seed is not None:
                    instances_gen.seed(args.seed)

                instances.extend(itertools.islice(instances_gen, args.n_instances))
        else:
            instances = itertools.islice(pickle.load(open(args.external_instances, "rb")), args.n_instances)

        scip_params = {
            "limits/time": args.time,
            "estimation/method": "c",
            "estimation/completiontype": "m"
        }

        train(
            pool,
            observer=observer,
            instances=instances,
            pop_size=50,
            n_generations=200,
            best_individual_path=args.out,
            scip_params=scip_params
        )

    # Wait for all the workers to finish and continue together
    COMM_WORLD.Barrier()



