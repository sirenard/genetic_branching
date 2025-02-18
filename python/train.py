import multiprocessing
import operator
import pickle
import tempfile
from multiprocessing import Pool

import ecole
import numpy as np
from boundml import solvers
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from functor_observer import FunctorObserver
from observer import MyObserver


def solve(path, individual):
    func = toolbox.compile(expr=individual)
    solver = solvers.EcoleSolver(FunctorObserver(func, MyObserver()), scip_params=scip_params)
    solver.solve(path)
    return solver["estimate_nnodes"]

def evalSymbReg(individual, pool: multiprocessing.Pool):
    async_results = [pool.apply_async(solve, args=(p, individual)) for p in instances_path]
    def get_res():
        nnodes = [ar.get() for ar in async_results]
        for i in range(len(nnodes)):
            nnodes[i] /= baseline_node[i]

        return sum(nnodes) / len(nnodes) + individual.height**0.5/100,

    return get_res

def evalSymbRegSingleCore(individual):
    nnodes = []
    for p in instances_path:
        nnodes.append(solve(p, individual))

    for i in range(len(nnodes)):
        nnodes[i] /= baseline_node[i]

    return sum(nnodes) / len(nnodes) + individual.height ** 0.5 / 100,


def mapp(f, args):
    callbacks = []
    with Pool() as pool:
        for arg in args:
            callbacks.append(f(arg, pool))

        pool.close()
        pool.join()

    result = [callback() for callback in callbacks]

    return result




# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def if_then_else(cond, then, default):
    if cond:
        return then
    else:
        return default

n_instances = 5

instances_ca = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
instances_ca.seed(12)

instances_sc = ecole.instance.SetCoverGenerator(n_rows=400, n_cols=1000)
instances_sc.seed(12)

instances_cfl = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100)
instances_cfl.seed(12)

instances = [instance for _, instance in zip(range(n_instances), instances_ca)]

# for _, *instance in zip(range(5), instances_ca, instances_sc, instances_cfl):
#     instances.extend(instance)


instances_path = []
baseline_node = []
files = []

scip_params = {
    "limits/time": 60,
    "estimation/method": "c",
    "estimation/completiontype": "m"
}

def solve_rb(path):
    solver = solvers.ClassicSolver("relpscost", scip_params)
    solver.solve(path)
    return solver["estimate_nnodes"]


pset = gp.PrimitiveSet("MAIN", 21)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.lt, 2)
pset.addPrimitive(operator.gt, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(if_then_else, 3)
pset.addTerminal(10000)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evalSymbReg)
toolbox.register("map", mapp)


toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


if __name__ == "__main__":

    # toolbox.register("map", mapp)

    for i,instance in enumerate(instances):
        prob_file = tempfile.NamedTemporaryFile(suffix=".lp")
        model = instance.as_pyscipopt()
        model.writeProblem(prob_file.name)

        instances_path.append(prob_file.name)
        # instances_path[i] = str.encode(prob_file.name)
        files.append(prob_file)

    with Pool(n_instances) as pool:
        nnodes = pool.map(solve_rb, instances_path)
    for n in nnodes:
        baseline_node.append(n)


    print("Starting evolution...")
    print(instances_path)


    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    ngen = 100
    for i in range(ngen):
        print(i)
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 1, stats=mstats,
                                       halloffame=hof, verbose=True)

        best = min(pop, key=lambda ind: ind.fitness.values[0])
        pickle.dump(best, open("best_gen2", "wb"))


    print(best)
    # print log



    for file in files:
        file.close()