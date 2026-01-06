import operator
import tempfile

import numpy as np
import pyscipopt
from boundml import solvers
from boundml.solvers import ModularSolver
from deap import base
from deap import creator
from deap import gp
from deap import tools
from mpipool import MPIExecutor
from objproxies import LazyProxy

import ea
from functor_observer import FunctorComponent


def shifted_geometric_mean(values, shift=1.0):
    values = np.array(values)
    geom_mean = np.exp(np.mean(np.log(values + shift))) - shift
    return geom_mean


def solve(path, individual, feature_observer, scip_params):
    toolbox = create_tool_box()
    func = toolbox.compile(expr=individual)
    solver = ModularSolver(FunctorComponent(func, feature_observer), scip_params=scip_params,
                                 configure=lambda model: model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF))
    solver.solve(path)
    return solver["estimate_nnodes"], solver["nnodes"], solver["time"]


def evalSymbReg(individual, pool: MPIExecutor, instances_path, *args):
    async_results = [pool.submit(solve, p, individual, *args) for p in instances_path]

    def get_res():
        estimate_times = []
        for estimate_nnodes, nnodes, time in [ar.result() for ar in async_results]:
            estimate_time = estimate_nnodes * (time / nnodes)
            if estimate_time < time:
                estimate_time = 10 * time
            estimate_times.append(estimate_time)

        return shifted_geometric_mean(estimate_times, 1),

    return get_res


def mapp(f, individuals, args):
    args = args[:]
    callbacks = []
    instances = args.pop(0)
    pool: MPIExecutor = args.pop(0)

    for individual in individuals:
        callbacks.append(f(individual, pool, instances, *args))

    result = [callback() for callback in callbacks]

    return result


def protectedDiv(left, right):
    if right != 0.0:
        return left / right
    else:
        return 1


def if_then_else(cond, then, default):
    if cond:
        return then
    else:
        return default


def create_tool_box(pool: MPIExecutor = None, observer=None, instances_paths=None, scip_params=None):
    if not hasattr(create_tool_box, "toolbox"):
        create_tool_box.toolbox = None

    if observer is None:
        return create_tool_box.toolbox

    pset = gp.PrimitiveSetTyped("MAIN", [np.float64] * len(observer), np.float64)

    # pset = gp.PrimitiveSet("MAIN", 72)
    def add_operator(operator, args_type, type, name, lazy=False):
        if lazy:
            f = lambda *args: LazyProxy(lambda: np.float64(operator(*args)))
        else:
            f = lambda *args: np.float64(operator(*[np.float64(arg) for arg in args]))
        pset.addPrimitive(f, args_type, type, name)

    add_operator(operator.add, [np.float64, np.float64], np.float64, "add")
    add_operator(operator.sub, [np.float64, np.float64], np.float64, "sub")
    add_operator(operator.mul, [np.float64, np.float64], np.float64, "mul")
    add_operator(protectedDiv, [np.float64, np.float64], np.float64, "div")
    add_operator(lambda x, y: x and y, [bool, bool], bool, "and_")
    add_operator(lambda x, y: x or y, [bool, bool], bool, "or_")
    add_operator(lambda x: not x, [bool], bool, "not")
    add_operator(operator.lt, [np.float64, np.float64], bool, "lt")
    add_operator(operator.gt, [np.float64, np.float64], bool, "gt")
    add_operator(operator.neg, [bool], bool, "neg")
    add_operator(if_then_else, [bool, np.float64, np.float64], np.float64, "if_then_else", lazy=True)
    add_operator(min, [np.float64, np.float64], np.float64, "min")
    add_operator(max, [np.float64, np.float64], np.float64, "max")
    add_operator(np.round, [np.float64], np.float64, "round")

    pset.addTerminal(2, np.float64)
    pset.addTerminal(4, np.float64)
    pset.addTerminal(6, np.float64)
    pset.addTerminal(8, np.float64)
    pset.addTerminal(128, np.float64)
    pset.addTerminal(256, np.float64)
    pset.addTerminal(512, np.float64)

    pset.addTerminal(True, bool)
    pset.addTerminal(False, bool)

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

    toolbox.register("simplify", simplify)

    assert (instances_paths is None and scip_params is None) or (
            instances_paths is not None and scip_params is not None), "If one argument is provided, the 3 must be set"
    if instances_paths is not None:
        args = [instances_paths, pool, observer, scip_params]
        toolbox.register("map", mapp, args=args)

    create_tool_box.toolbox = toolbox
    return toolbox, pset


def get_known_individuals(pset):
    relpscost1 = creator.Individual(
        gp.PrimitiveTree.from_string("if_then_else(lt(min(ARG31,ARG32),6), ARG30, ARG23)", pset))
    relpscost2 = creator.Individual(
        gp.PrimitiveTree.from_string("if_then_else(lt(min(ARG31,ARG32),4), ARG30, ARG23)", pset))
    pscost = creator.Individual(
        gp.PrimitiveTree.from_string("ARG23", pset))
    hybrid1 = creator.Individual(
        gp.PrimitiveTree.from_string("if_then_else(lt(ARG18,6), ARG30, ARG23)", pset))
    hybrid2 = creator.Individual(
        gp.PrimitiveTree.from_string("if_then_else(lt(ARG18,4), ARG30, ARG23)", pset))

    return [relpscost1, relpscost2, pscost, hybrid1, hybrid2]


def simplify(individual: gp.PrimitiveTree):
    """Simplify a DEAP GP expression (PrimitiveTree)."""

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    def _eval(elements):
        """Recursively simplify the GP expression. Each call to _val consume some elements in the list "elements" """
        expr = elements.pop(0)

        if isinstance(expr, gp.Primitive):
            args_elements = [_eval(elements) for _ in range(expr.arity)]
            args = [els[0] for els in args_elements]

            name = expr.name

            # Try simplifying based on known patterns
            if name in ["min", "max"] and args[0] == args[1]:
                return args_elements[0]
            elif name in ["lt", "gt"] and args[0] == args[1]:
                return gp.Terminal("False", False, bool)
            elif name == "if_then_else":
                cond, a, b = args
                if isinstance(cond, gp.Terminal) and isinstance(cond.value, bool):
                    return args_elements[1] if cond.value else args_elements[2]
                elif a == b:
                    return args_elements[0]
            # No simplification rule applies, return as is
            return [gp.Primitive(expr.name, args, expr.ret)] + list(flatten(args_elements))
        else:
            return [expr]  # constants, variables

    new_elements = _eval(individual[:])

    new_individual = creator.Individual(gp.PrimitiveTree(new_elements))
    new_individual.fitness = individual.fitness
    return new_individual


def train(pool: MPIExecutor, observer, instances, pop_size, n_generations, best_individual_path="best_ind",
          scip_params={}):
    instances_path = []
    files = []

    print("Presolving.....")
    for i, instance in enumerate(instances):
        prob_file = tempfile.NamedTemporaryFile(suffix=".mps")
        if type(instance) is str:
            model = pyscipopt.Model()
            model.setParam("display/verblevel", 0)
            model.readProblem(instance)
        else:
            model = instance

        model.presolve()
        model.writeProblem(prob_file.name, trans=True)

        instances_path.append(prob_file.name)
        files.append(prob_file)

    print("Starting evolution...")

    toolbox, pset = create_tool_box(pool, observer, instances_path, scip_params)

    known_individuals = get_known_individuals(pset)
    pop = toolbox.population(n=pop_size - len(known_individuals))
    pop.extend(known_individuals)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    ea.evolution_algorithm(pop, toolbox, 0.5, 0.2, 0.0, n_generations, stats=mstats,
                           halloffame=hof, verbose=True, best_individual_path=best_individual_path)

    print(hof[0])

    for file in files:
        file.close()
