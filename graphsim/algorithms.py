from .classes import Graph, VertexMapping
from .utils import default_f_score, default_g_cost, exists, forall
from time import clock
from random import choice


def m_score(graph1, graph2, mapping, f=default_f_score, g=default_g_cost):
    intersection = graph1.full_intersection(graph2, mapping)
    return intersection.score(f) - mapping.cost(g)


def m_similarity(graph1, graph2, mapping, f=default_f_score, g=default_g_cost):
    score = m_score(graph1, graph2, mapping, f, g)
    union = graph1.union(graph2)
    return float(max(0, score)) / union.score(f)


def candidate_relations(graph1, graph2):
    """
    Computes the set of relations between vertices that have at least one
    feature in common.
    """
    relations = set()
    for f1, f2 in ((f1, f2) for f1 in graph1.features
                            for f2 in graph2.D.get((len(f1), f1[-1]), [])):
        relations.add((f1[0], f2[0]))
        if len(f1) == 3: relations.add((f1[1], f2[1]))
    return relations


def best_mapping(graph1, graph2, f=default_f_score, g=default_g_cost):
    """
    Computes the greatest similarity and the associated map between two graphs
    using given custom metrics. This algorithm is exhaustive and uses a B&B
    (Branch and Bound) approach. It introduces small improvements over the
    outlined B&B methodology proposed by Champin and Solnon [1], like a better
    B&B upper bound function (bound of the best possible score in the
    considered branch) and a reduced list of candidate relations that includes
    only those relations of vertices that have at least one features in common 
    vertex label or {in|out}coming edge with same label). Parameters:
        - graph1, graph2: graphs to compare
        - f: metric used in the intersection and union of graph features. It
             should be a monotonic increasing function over the cardinality of
             the considered cell.
        - g: metric used to penalize certain relations. Champin and Solnon [1]
             penalize exclusively the splits, we consider a more general
             framework in which the g function is applied to the whole mapping
             with the possibility of ignoring non-splits. g should be a
             monotonic increasing function over the cardinality of the set of
             relations.
    Returns:
        mapping, similarity: a tuple in which the first element is the best
                             found mapping and the second is the associated
                             similarity
    """
    relations = list(candidate_relations(graph1, graph2))
    full_map = VertexMapping(*relations)
    f_max = graph1.full_intersection(graph2, full_map).score(f)
    best_score = float('-inf')

    def best_mapping_aux(mapping=VertexMapping(), idx=0):
        nonlocal best_score
        if f_max - mapping.cost(g) <= best_score: # check lower bound
            return None, float('-inf')
        elif idx == len(relations):
            score = m_score(graph1, graph2, mapping, f, g)
            if score > best_score:
                best_score = score
                return mapping.copy(), score
            else:
                return None, float('-inf')
        v1, v2 = relations[idx]
        mapping1, score1 = best_mapping_aux(mapping, idx+1)
        mapping.add_relation(v1, v2)
        mapping2, score2 = best_mapping_aux(mapping, idx+1)
        mapping.remove_relation(v1, v2)
        return (mapping1, score1) if score1 >= score2 else (mapping2, score2)

    mapping, score = best_mapping_aux()
    similarity = float(max(0, score)) / graph1.union(graph2).score(f)
    return mapping, similarity


def greedy_mapping(graph1, graph2, f=default_f_score, g=default_g_cost):
    def include_evaluation(relation):
        u1, u2 = relation
        m.add_relation(u1, u2)
        intersection = graph1.full_intersection(graph2, m)
        score = intersection.score(f) - m.cost(g)
        m.remove_relation(u1, u2)
        return (score, u1, u2, intersection)

    def look_ahead(packed):
        score, u1, u2, intersection = packed
        direct_look_ahead = set()
        inverse_look_ahead = set()
        for f1 in graph1.r_e():
            for f2 in graph2.D[(len(f1), f1[-1])]:
                same_label = f1[2] == f2[2]
                same_src = f1[0] == u1 and f2[0] == u2
                same_dst = f1[1] == u1 and f2[1] == u2
                if same_label and (same_src or same_dst):
                    direct_look_ahead.add(f1)
                    inverse_look_ahead.add(f2)
        direct_look_ahead.difference_update(intersection._direct)
        inverse_look_ahead.difference_update(intersection._inverse)
        return (f(direct_look_ahead, inverse_look_ahead), score, u1, u2)

    relations = candidate_relations(graph1, graph2)
    m = VertexMapping()
    best_m = VertexMapping()
    best_score = 0
    score = 0
    while True:
        evaluated = list(map(include_evaluation, relations))
        max_score = max(evaluated)[0]
        cand = filter(lambda t: max_score-t[0] < 1e-2, evaluated)
        evaluated_ = list(map(look_ahead, cand))
        max_f = max(evaluated_)[0]
        cand_ = list(filter(lambda t: max_f-t[0] < 1e-2, evaluated_))
        if forall(lambda t: t[0] < 1e-2 and t[1] < score, cand_): break
        _, score, u1, u2 = choice(cand_)
        m.add_relation(u1, u2)
        relations.remove((u1, u2))
        if score > best_score:
            best_score = score
            best_m = m.copy()

    similarity = float(max(0, best_score)) / graph1.union(graph2).score(f)
    return best_m, similarity


def r_greedy_mapping(graph1, graph2, f=default_f_score, g=default_g_cost, r=10):
    best_m, best_similarity = None, 0
    for i in range(r):
        m, similarity = greedy_mapping(graph1, graph2, f, g)
        if similarity > best_similarity:
            best_similarity = similarity
            best_m = m
    return best_m, best_similarity

# [1] Champin, P., & Solnon, C. (2003). Measuring the similarity of labeled
# graphs. In International Conference on Case-Based Reasoning (pp. 80–95).
# Springer Berlin Heidelberg.
