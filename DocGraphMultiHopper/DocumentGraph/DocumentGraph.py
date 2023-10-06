from .SentenceNode import SentenceNode
from .SentenceGraph import SentenceGraph

def cross_link_ner_nodes(graph1, graph2):
    common_ners = set.intersection(
        set(graph1.ner_roots.keys()), set(graph2.ner_roots.keys())
    )

    for ner in common_ners:
        for e in graph2.ner_clusters[ner]:
            graph2.node_from_sent(e.sent).children.append(graph1.ner_roots[ner])
        for e in graph1.ner_clusters[ner]:
            graph1.node_from_sent(e.sent).children.append(graph2.ner_roots[ner])

class DocumentGraph(object):

    def __init__(
        self, 
        doc, 
        doc_id=0, 
        root_token="Root", 
        directed=True,
        featurizer=None,
        self_loop=False,
    ): 

        self.doc = doc
        self.doc_id = doc_id
        self.directed = directed
        self.featurizer = featurizer
        self.self_loop = self_loop

        self.graph = SentenceNode(
            sent=root_token, sent_id=doc_id, featurizer=featurizer, self_loop=False,
        )

        self.sub_graphs = list()
        for i, para in enumerate(self.doc):
            sub_graph = SentenceGraph(
                para, 
                para_id=(doc_id, i), 
                directed=directed,
                featurizer=featurizer,
                self_loop=self_loop,
            )
            self.sub_graphs.append(sub_graph)
            self.graph.children.append(sub_graph.root)

        for i in range(len(self.sub_graphs)-1):
            for j in range(i+1, len(self.sub_graphs)):
                cross_link_ner_nodes(self.sub_graphs[i], self.sub_graphs[j])

    def __repr__(self):
        return str(self.doc_id)
