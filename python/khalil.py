from boundml.instances import *
from boundml.dataset_generator import DatasetGenerator
from boundml.components import StrongBranching

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # instances = CombinatorialAuctionGenerator(100, 500)
    # instances = IndependentSetGenerator(n_nodes = 500)
    # instances = CapacitatedFacilityLocationGenerator(n_customers=100)
    instances = SetCoverGenerator(n_rows = 500, n_cols = 1000)
    instances.seed(args.seed)

    expert_probability = 0.1

    scip_params = {
    }


    generator = DatasetGenerator(
        instances,
        expert_observer=StrongBranching(),
        exploration_observer=ecole.observation.Pseudocosts(),
        expert_probability=expert_probability,
        scip_params=scip_params,
        )

    generator.generate("samples/sc", max_samples=100000//26, sample_prefix=args.sample_prefix)

