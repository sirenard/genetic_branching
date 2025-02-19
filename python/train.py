import ecole

from observer import MyObserver
from utils import train

n_instances = 3

instances_ca = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
instances_ca.seed(12)

instances_sc = ecole.instance.SetCoverGenerator(n_rows=400, n_cols=1000)
instances_sc.seed(12)

instances_cfl = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100)
instances_cfl.seed(12)

instances = [instance for _, instance in zip(range(n_instances), instances_ca)]

scip_params = {
    "limits/time": 60,
    "estimation/method": "c",
    "estimation/completiontype": "m"
}

if __name__ == "__main__":
    train(
        observer=MyObserver(),
        instances=instances,
        pop_size=10,
        n_generations=20,
        best_individual_path="./individuals/best",
        scip_params=scip_params
    )

