import graphviz as gv

from .utils import list_to_pretty_str, add_value_to_dict, exists,\
                   remove_value_from_dict, default_g_cost, default_f_score,\
                   index_by_length_and_label, match_feature

from copy import deepcopy


class Graph:
    def __init__(self, *args):
        self.features = set(args)
        self.D = index_by_length_and_label(args)

    def V(self, vertex=None, label=None):
        r_v = self.r_v(vertex, label)
        vertices = dict()
        for vertex, label in r_v:
            add_value_to_dict(vertices, vertex, label)
        return vertices

    def E(self, from_=None, to=None, label=None):
        edges = dict()
        r_e = self.r_e(from_, to, label)
        for u, v, label in r_e:
            add_value_to_dict(edges, (u, v), label)
        return edges
   

    def r_v(self, vertex=None, label=None, f=None):
        return filter(lambda f: match_feature(f, (vertex, label)), self.features)


    def r_e(self, from_=None, to=None, label=None, f=None):
        return filter(lambda f: match_feature(f, (from_, to, label)), self.features)


    def partial_intersection(self, graph, mapping):
        intersection = set()
        for feature in self.features:
            if len(feature) == 2: # vertex
                mappedFeatures = ((mVertex, feature[1])
                                  for mVertex in mapping.m(feature[0]))
            else: # assume len(feature) == 3 => edge
                mappedFeatures = ((mVertex1, mVertex2, feature[2])
                                  for mVertex1 in mapping.m(feature[0])
                                  for mVertex2 in mapping.m(feature[1]))
            if exists(graph.has_feature, mappedFeatures):
                intersection.add(feature)
        return intersection

    def full_intersection(self, graph, mapping):
        direct = self.partial_intersection(graph, mapping)
        inverse = graph.partial_intersection(self, mapping.I)
        return FeatureSelection(direct, inverse)

    def union(self, graph):
        return FeatureSelection(self.features, graph.features)

    def has_feature(self, feature):
        return feature in self.features

    def _dot_(self): 
        gv_graph = gv.Digraph()
        V = self.V()
        for vertex, labels in V.items():
            gv_graph.node(str(vertex), label='{}: {}'.format(
                vertex, list_to_pretty_str(labels)))
        E = self.E()
        for (from_, to), labels in E.items():
            gv_graph.edge(str(from_), str(to), list_to_pretty_str(labels))
        return gv_graph

    def _repr_svg_(self):
        dot = self._dot_()
        return dot._repr_svg_()

    def __str__(self):
        return str(self.features)


class VertexMapping:

    def __init__(self, *args):
        self._m = dict()
        self._m_inv = dict()
        for u, v in args: self.add_relation(u, v)

    def add_relation(self, u, v):
        add_value_to_dict(self._m, u, v)
        add_value_to_dict(self._m_inv, v, u)

    def remove_relation(self, u, v):
        removed1 = remove_value_from_dict(self._m, u, v)
        removed2 = remove_value_from_dict(self._m_inv, v, u)
        return removed1 and removed2

    def cost(self, g=default_g_cost):
        return 0 if g is None else g(self._m, self._m_inv)

    @property
    def relations(self):
        relations = set()
        for u in self._m:
            for v in self._m[u]:
                relations.add((u, v))
        return relations

    @property
    def I(self):
        mapping = VertexMapping()
        mapping._m = self._m_inv
        mapping._m_inv = self._m
        return mapping

    def copy(self):
        mapping = VertexMapping()
        mapping._m = deepcopy(self._m)
        mapping._m_inv = deepcopy(self._m_inv)
        return mapping

    def m(self, v):
        return self._m.get(v, [])

    def _dot_(self):
        gv_graph = gv.Digraph()
        for vertex in self._m:
            gv_graph.node(str(vertex))
        for vertex in self._m_inv:
            gv_graph.node(str(vertex))
        for from_ in self._m:
            for to_ in self._m[from_]:
                gv_graph.edge(str(from_), str(to_))
        gv_graph.edge_attr.update({'arrowhead': 'none', 'style': 'dashed'})
        return gv_graph

    def _repr_svg_(self):
        dot = self._dot_()
        return dot._repr_svg_()

    def __str__(self):
        return str(self.relations)


class FeatureSelection:
    
    def __init__(self, direct, inverse):
        self._direct = direct
        self._inverse = inverse

    def score(self, f=default_f_score):
        return f(self._direct, self._inverse)

    def __str__(self):
        return "direct: {}\ninverse: {}\ndef. score: {}".format(self._direct,
                                                                self._inverse,
                                                                self.score())



