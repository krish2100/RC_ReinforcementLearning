import re
import spacy
import neuralcoref
from itertools import groupby
from functools import partial

from .SentenceNode import SentenceNode


class SentenceGraph(object):

    def __init__(
        self, 
        para, 
        para_id=0, 
        directed=True,
        to_space_regex="\s+|-", 
        featurizer=None,
        self_loop=False,
    ):

        self.para = para
        self.para_id = para_id
        self.directed = directed
        self.to_space_regex = to_space_regex
        self.featurizer = featurizer
        self.self_loop = self_loop

        self.engine = spacy.load("en")
        neuralcoref.add_to_pipe(self.engine)
        self.doc = self.engine(re.sub(to_space_regex, " ", para))
        
        self.sent_nodes = [
            SentenceNode(
                sent=sent, 
                sent_id=(self.para_id, i), 
                featurizer=featurizer,
                self_loop=self_loop,
            )
            for i, sent in enumerate(self.doc.sents)
        ]

        self.root = self.init_graph()
        self.update_graph_from_coref()

        self.ner_clusters = {
            i:list(j) for i,j in groupby(
                sorted(
                    self.doc.ents, key=lambda x: str(x)
                ),
                key=lambda x: str(x)
            )
        }
        
        self.update_graph_from_ner(self.ner_clusters)
        self.ner_roots = {
            i: self.node_from_sent(min(j).sent) for i,j in self.ner_clusters.items()
        }

    def init_graph(self):
        for i in range(1, len(self.sent_nodes)):
            self.sent_nodes[i-1].children.append(self.sent_nodes[i])
            if not self.directed:
                self.sent_nodes[i].children.append(self.sent_nodes[i-1])
        return self.sent_nodes[0]

    def node_from_sent(self, sent):
        return self.sent_nodes[self.sent_nodes.index(SentenceNode(sent=sent))]

    def update_adjacency(self, src_sent, tgt_sent):
        src = self.node_from_sent(sent=src_sent)
        tgt = self.node_from_sent(sent=tgt_sent)
        if tgt not in src.children:
            src.children.append(tgt)

    def update_graph(self, ent_clusters):

        for entities in ent_clusters:
            for i in range(len(entities)-1):
                for j in range(i+1, len(entities)):

                    sent1 = entities[i].sent
                    sent2 = entities[j].sent

                    if sent1 != sent2:
                        if self.directed:
                            if sent1 < sent2:
                                self.update_adjacency(sent1, sent2)
                            else:
                                self.update_adjacency(sent2, sent1)
                        else:
                            self.update_adjacency(sent1, sent2)
                            self.update_adjacency(sent2, sent1)

    def update_graph_from_coref(self):
        self.update_graph(
            [coref.mentions for coref in self.doc._.coref_clusters]
        )

    def update_graph_from_ner(self, ner_clusters):
        self.update_graph(ner_clusters.values())
