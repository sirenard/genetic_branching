import ecole

from observer import MyObserver, KhalilObserver
# from ucig_generator import UcigGenerator
from utils import train
# from folder_instance import FolderInstanceGenerator

n_instances = 5

instances_ca = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
instances_ca.seed(12)

instances_sc = ecole.instance.SetCoverGenerator(n_rows=750, n_cols=1000)
instances_sc.seed(12)

instances_cfl = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100)
instances_cfl.seed(12)

# instances = FolderInstanceGenerator("./instances/MVC-easy/train")
# instances.seed(0)

# instances = UcigGenerator("./instances/uc_hetrogeneous_5_1")

instances = [instance for _, instance in zip(range(30), instances_ca)]

#easy_instances = ['/home/simon/Téléchargements/benchmark/irp.mps', '/home/simon/Téléchargements/benchmark/roll3000.mps', '/home/simon/Téléchargements/benchmark/nursesched-sprint02.mps', '/home/simon/Téléchargements/benchmark/swath1.mps', '/home/simon/Téléchargements/benchmark/neos-3083819-nubu.mps', '/home/simon/Téléchargements/benchmark/neos-1445765.mps', '/home/simon/Téléchargements/benchmark/neos-1122047.mps', '/home/simon/Téléchargements/benchmark/beasleyC3.mps', '/home/simon/Téléchargements/benchmark/neos8.mps', '/home/simon/Téléchargements/benchmark/air05.mps', '/home/simon/Téléchargements/benchmark/neos-911970.mps', '/home/simon/Téléchargements/benchmark/neos-860300.mps', '/home/simon/Téléchargements/benchmark/drayage-100-23.mps', '/home/simon/Téléchargements/benchmark/binkar10_1.mps', '/home/simon/Téléchargements/benchmark/pg.mps', '/home/simon/Téléchargements/benchmark/piperout-27.mps', '/home/simon/Téléchargements/benchmark/cbs-cta.mps', '/home/simon/Téléchargements/benchmark/n5-3.mps', '/home/simon/Téléchargements/benchmark/neos-1582420.mps', '/home/simon/Téléchargements/benchmark/mik-250-20-75-4.mps', '/home/simon/Téléchargements/benchmark/timtab1.mps']
#instances = [ecole.scip.Model.from_file(p) for p in easy_instances]

scip_params = {
    "limits/time": 200,
    "estimation/method": "c",
    "estimation/completiontype": "m"
}

if __name__ == "__main__":
    train(
        observer=MyObserver(),
        # observer=KhalilObserver(),
        instances=instances,
        n_instances=n_instances,
        pop_size=30,
        n_generations=100,
        best_individual_path="./individuals/best_ca4_lazy",
        scip_params=scip_params
    )

