import multiprocessing
import operator
import random
import numpy as np


import dill as pickle
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
import deap
from functor_observer import FunctorObserver



def shifted_geometric_mean(values, shift=1.0):
    values = np.array(values)
    geom_mean = np.exp(np.mean(np.log(values + shift))) - shift
    return geom_mean

def solve(path, individual, feature_observer, scip_params):
    toolbox = create_tool_box()
    func = toolbox.compile(expr=individual)
    # solver = solvers.EcoleSolver(FunctorObserver(func, MyObserver()), scip_params=scip_params)
    solver = solvers.EcoleSolver(FunctorObserver(func, feature_observer), scip_params=scip_params)
    solver.solve(path)
    return solver["estimate_nnodes"], solver["nnodes"], solver["time"]

def evalSymbReg(individual, pool: multiprocessing.Pool, instances_path, baseline_average_time, *args):
    async_results = [pool.apply_async(solve, args=(p, individual, *args)) for p in instances_path]
    def get_res():
        # score = 0
        estimate_times = []
        for estimate_nnodes, nnodes, time in [ar.get() for ar in async_results]:
            estimate_time = estimate_nnodes * (time/nnodes)
            if estimate_time < time:
                estimate_time = 10*time
            estimate_times.append(estimate_time)

            # print(nnodes, estimate_nnodes, time, estimate_time)
            # score += max(estimate_time, time)

        return shifted_geometric_mean(estimate_times, 1)/baseline_average_time, # * (1 + individual.height/100),

    return get_res

# def evalSymbRegSingleCore(individual):
#     nnodes = []
#     for p in instances_path:
#         nnodes.append(solve(p, individual))
#
#     for i in range(len(nnodes)):
#         nnodes[i] /= baseline_node[i]
#
#     return sum(nnodes) / len(nnodes) + individual.height ** 0.5 / 100,


def mapp(f, individuals, args):
    args = args[:]
    callbacks = []
    instances, nnodes_baseline, n = args.pop(0)

    if n is not None:
        indexes = random.choices(range(len(instances)), k=n)
        instances = [instances[i] for i in indexes]
        nnodes_baseline = [nnodes_baseline[i] for i in indexes]

    baseline_average_time = shifted_geometric_mean(nnodes_baseline, 1)

    with Pool() as pool:
        for individual in individuals:
            callbacks.append(f(individual, pool, instances, baseline_average_time, *args))

        pool.close()
        pool.join()

    result = [callback() for callback in callbacks]

    return result

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


def create_tool_box(observer = None, instances_paths = None, scip_params = None, n_instance=None, nnodes_baseline = None):
    if not hasattr(create_tool_box, "toolbox"):
        create_tool_box.toolbox = None

    if observer is None:
        return create_tool_box.toolbox

    pset = gp.PrimitiveSet("MAIN", len(observer))
    # pset = gp.PrimitiveSet("MAIN", 72)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.lt, 2)
    pset.addPrimitive(operator.gt, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(if_then_else, 3)

    for i in range(-3, 5):
        pset.addTerminal(2**i)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    toolbox.register("evaluate", evalSymbReg)

    assert (instances_paths is None and scip_params is None) or (instances_paths is not None and scip_params is not None), "If one argument is provided, the 3 must be set"
    if instances_paths is not None:
        args = [(instances_paths, nnodes_baseline, n_instance), observer, scip_params]
        toolbox.register("map", mapp, args=args)

    create_tool_box.toolbox = toolbox
    return toolbox


def solve_rb(path, scip_params={}):
    solver = solvers.ClassicSolver("relpscost", scip_params)
    solver.solve(path)
    return solver["estimate_nnodes"]

def train(observer, instances, pop_size, n_generations, best_individual_path="best_ind", scip_params = {}, n_instances=None):
    instances_path = []
    files = []

    for i,instance in enumerate(instances):
        prob_file = tempfile.NamedTemporaryFile(suffix=".lp")
        model = instance.as_pyscipopt()
        model.writeProblem(prob_file.name)

        instances_path.append(prob_file.name)
        files.append(prob_file)

    with Pool() as pool:
        nnodes_baseline = pool.map(solve_rb, instances_path)

    print("Starting evolution...")

    toolbox = create_tool_box(observer, instances_path, scip_params, n_instances, nnodes_baseline)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    for i in range(n_generations):
        print(i)
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 1, stats=mstats,
                                       halloffame=hof, verbose=True)

        # best = min(pop, key=lambda ind: ind.fitness.values[0])
        best = hof.items[0]
        pickle.dump(best, open(best_individual_path, "wb"))

    print(best)

    for file in files:
        file.close()