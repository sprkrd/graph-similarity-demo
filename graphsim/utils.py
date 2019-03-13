import graphviz as gv

from functools import reduce

def list_to_pretty_str(l):
    l = l.copy()
    l.sort()
    return reduce(lambda acc, e: acc + ', ' + str(e), l[1:], str(l[0]))


def add_value_to_dict(d, key, value):
    try:             d[key].append(value)
    except KeyError: d[key] = [value]


def remove_value_from_dict(d, key, value):
    try:
        d[key].remove(value)
        removed = True
    except (KeyError, ValueError):
        removed = False
    if removed and not d[key]:
        del d[key]
    return removed


def match_feature(feature, pattern):
    if len(feature) != len(pattern):
        return False
    for x, y in zip(feature, pattern):
        if y is not None and x != y:
            return False
    return True


def exists(cond, iterable):
    it_exists = False
    for element in iterable:
        if cond(element):
            it_exists = True
            break
    return it_exists


def forall(cond, iterable):
    is_forall = True
    for element in iterable:
        if not cond(element):
            is_forall = False
            break
    return is_forall


def index_by_length_and_label(features):
    D = {}
    for f in features:
        add_value_to_dict(D, (len(f), f[-1]), f) # index by length and label
    return D


def graph_match_dot(graph1, graph2, mapping=None):
    gv_graph = gv.Digraph()
    gv_graph.graph_attr['rankdir'] = 'LB'

    with gv_graph.subgraph(name='cluster-0') as c:
        c.node_attr['shape'] = 'hexagon'
        c.graph_attr['label'] = 'Source graph'
        c.subgraph(graph1._dot_())

    with gv_graph.subgraph(name='cluster-1') as c:
        c.node_attr.update({'style': 'filled', 'fillcolor': '#cccccc'})
        c.graph_attr['label'] = 'Target graph'
        c.subgraph(graph2._dot_())

    if mapping is not None:
        gv_graph3 = mapping._dot_()
        gv_graph3.edge_attr['constraint'] = 'false'
        gv_graph.subgraph(gv_graph3)

    return gv_graph


def default_g_cost(m, m_inv):
    """
    Default g function. It ignores whether it is being computed on the direct
    or in the inverse mapping. The cost is the sum of the cardinalities of all
    the target sets s_v minus 1 for (v, s_v) in relations. This is, the cost
    will be 0 for all the non-splits; for splits the cost is the cardinality of
    the split minus one. The split definition is that of [1]
    """
    g1 = reduce(lambda acc, s_v: acc + len(s_v)-1, m.values(), 0)
    g2 = reduce(lambda acc, s_v: acc + len(s_v)-1, m_inv.values(), 0)
    return g1 + g2


def default_f_score(direct, inverse):
    return len(direct) + len(inverse)

# [1] Champin, P., & Solnon, C. (2003). Measuring the similarity of labeled
# graphs. In International Conference on Case-Based Reasoning (pp. 80–95).
# Springer Berlin Heidelberg.

