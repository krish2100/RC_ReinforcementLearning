import re
import spacy
from functools import total_ordering

@total_ordering
class SentenceNode(object):

    def __init__(self, sent, sent_id=0, featurizer=None, self_loop=False): 
        self.sent = sent
        self.sent_id = sent_id
        self.featurizer = featurizer
        self.children = list()
        self.self_loop = self_loop
        if self_loop:
            self.children.append(self)

    @property
    def embedding(self):
        if self.featurizer:
            return self.featurizer(str(self.sent))

    def __eq__(self, other):
        return self.sent == other.sent

    def __lt__(self, other):
        return self.sent < other.sent

    def __repr__(self):
        return f"{self.sent_id} -- {self.sent}"
