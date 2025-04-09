import ecole
from mpipool import MPIExecutor
from mpi4py.MPI import COMM_WORLD

from observer import MyObserver
from utils import train, create_tool_box

n_instances = 1


instances_ca = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
instances_ca.seed(12)

instances_sc = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000)
instances_sc.seed(12)

instances_cfl = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100)
instances_cfl.seed(12)

instances_mis = ecole.instance.IndependentSetGenerator(500)
instances_mis.seed(12)


instances = [instance for _, *instances in zip(range(n_instances), instances_ca, instances_sc, instances_cfl, instances_mis) for instance in instances]

scip_params = {
    "limits/time": 60,
    "estimation/method": "c",
    "estimation/completiontype": "m"
}

if __name__ == "__main__":
    observer = MyObserver()

    if COMM_WORLD.rank:
        create_tool_box(observer=observer)

    with MPIExecutor() as pool:
        pool.workers_exit()

        train(
            pool,
            observer=observer,
            instances=instances,
            pop_size=5,
            n_generations=3,
            best_individual_path="./individuals/test",
            scip_params=scip_params
        )

    # Wait for all the workers to finish and continue together
    COMM_WORLD.Barrier()



