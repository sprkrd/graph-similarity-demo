from random import random
from bitstring import BitArray
from deap import base, creator, tools, algorithms
from .utils import default_f_score, default_g_cost
from .algorithms import m_score, m_similarity
from .classes import VertexMapping
import numpy as np


def random_bit(p=0.5):
    return random() <= p


def random_bitstring(N, p=0.5):
    bits = BitArray(N)
    for idx in range(N):
        bits[idx] = random_bit(p)
    return bits


def select_using_mask(elements, mask):
    filtered_pairs = filter(lambda pair: pair[1], zip(elements, mask))
    return list(map(lambda pair: pair[0], filtered_pairs))


def mask_to_mapping(relations, mask):
    selected = select_using_mask(relations, mask)
    mapping = VertexMapping(*selected)
    return mapping


def evaluate(graph1, graph2, f, g, relations, mask):
    mapping = mask_to_mapping(relations, mask)
    return (m_score(graph1, graph2, mapping, f, g),)


creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', BitArray, fitness=creator.FitnessMax)


def ga_mapping(graph1, graph2, f=default_f_score, g=default_g_cost,
        p=0.5, lambda_=None, mu=5, max_fitness_eval=1000,
        mutation_strength=1.0, cxpb=0.3333, mutpb=0.3333,
        track_statistics=False, method='simple', initial_pop=None):
    relations = [(u, v) for u in graph1.V().keys() for v in graph2.V().keys()]
    lambda_ = mu if method == 'simple' else (lambda_ or mu*2)
    ngen = max_fitness_eval//lambda_
    pbflip = min(1.0, mutation_strength/len(relations))
    toolbox = base.Toolbox()
    toolbox.register('attr_bit', random_bit, p=p)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
            toolbox.attr_bit, n=len(relations))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('select', tools.selTournament, tournsize=lambda_//3)
    toolbox.register('mutate', tools.mutFlipBit, indpb=pbflip)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('evaluate', evaluate, graph1, graph2, f, g, relations)
    initial_pop = toolbox.population(mu)
    hof = tools.HallOfFame(1)
    if method == 'simple':
        algorithm = algorithms.eaSimple
        args = (initial_pop, toolbox, cxpb, mutpb, ngen) 
    elif method == 'plus':
        algorithm = algorithms.eaMuPlusLambda
        args = (initial_pop, toolbox, mu, lambda_, cxpb, mutpb, ngen)
    elif method == 'comma':
        algorithm = algorithms.eaMuCommaLambda
        args = (initial_pop, toolbox, mu, lambda_, cxpb, mutpb, ngen)
    if track_statistics:
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("mean", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max) 
        final_pop, logbook = algorithm(*args, verbose=False, halloffame=hof,
                stats=stats)
    else:
        final_pop, _ = algorithm(*args, verbose=False, halloffame=hof)
    best_mask = hof[0]
    mapping = mask_to_mapping(relations, best_mask)
    similarity = m_similarity(graph1, graph2, mapping, f, g)
    if track_statistics:
        return mapping, similarity, logbook
    else:
        return mapping, similarity

