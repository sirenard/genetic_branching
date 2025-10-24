import argparse
import itertools
import tempfile

from boundml.instances import *
from mpipool import MPIExecutor
from mpi4py.MPI import COMM_WORLD

from component import CustomComponent
from folder_instance import FolderInstanceGenerator
from utils import train, create_tool_box
import pickle

if __name__ == "__main__":
    observer = CustomComponent()

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
        group.add_argument("--folder_instances", type=str, help="Folder containing instances to use")

        parser.add_argument("--n_instances", type=int, help="Number of instances to solve", required=True)

        parser.add_argument("--time", type=int, default=300, help="Time limit for each instance")
        parser.add_argument("--out", type=str, help="Pickle output file of the resulting SolverEvaluationResult",
                            required=True)

        parser.add_argument("--n_gen", type=int, help="Number of generations to use", default=100)
        parser.add_argument("--pop_size", type=int, help="Population size", default=30)

        parser.add_argument("--tmp_folder", type=str, help="Folder path used for temp files, by default use system path")

        args = parser.parse_args()

        if args.tmp_folder:
            tempfile.tempdir = args.tmp_folder

        if args.instances is not None:
            instances = []
            for instances_name in args.instances:
                match instances_name:
                    case "ca":
                        kwargs = {"easy": {"n_items": 200, "n_bids": 1000},
                                  "hard": {"n_items": 300, "n_bids": 1500}, }[args.difficulty]
                        instances_gen = CombinatorialAuctionGenerator(**kwargs)
                    case "cfl":
                        kwargs = \
                        {"easy": {"n_customers": 200}, "hard": {"n_customers": 300}, }[
                            args.difficulty]
                        instances_gen = CapacitatedFacilityLocationGenerator(**kwargs)
                    case "sc":
                        kwargs = {"easy": {"n_rows": 1000, "n_cols": 1000},
                                  "hard": {"n_rows": 1500, "n_cols": 1000}, }[args.difficulty]
                        instances_gen = SetCoverGenerator(**kwargs)
                    case "mis":
                        kwargs = {"easy": {"n_nodes": 500}, "medium": {"n_nodes": 1000}, "hard": {"n_nodes": 1500}, }[
                            args.difficulty]
                        instances_gen = IndependentSetGenerator(**kwargs)
                    case _:
                        raise NotImplementedError

                if args.seed is not None:
                    instances_gen.seed(args.seed)

                instances.extend(itertools.islice(instances_gen, args.n_instances))
        elif args.external_instances:
            instances = itertools.islice(pickle.load(open(args.external_instances, "rb")), args.n_instances)
        else:
            instances = itertools.islice(FolderInstanceGenerator(args.folder_instances), args.n_instances)

        scip_params = {
            "timing/clocktype": 1,
            "limits/time": args.time,
            "estimation/method": "c",
            "estimation/completiontype": "m"
        }

        train(
            pool,
            observer=observer,
            instances=instances,
            pop_size=args.pop_size,
            n_generations=args.n_gen,
            best_individual_path=args.out,
            scip_params=scip_params
        )

    # Wait for all the workers to finish and continue together
    COMM_WORLD.Barrier()



