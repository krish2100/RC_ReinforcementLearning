import random
from ..DocumentGraph import SentenceNode, DocumentGraph


class Env(object):
    def __init__(
        self, 
        dataset, 
        subset="train", 
        max_steps=10, 
        seed=42,
        directed=True,
        self_loop=False,
        featurizer=None,
    ):

        self.dataset = dataset
        self.subset = subset
        self.max_steps = max_steps
        self.iterator = (
            iter(range(len(self.dataset[self.subset]))) 
            if seed is None else random.Random(seed)
        )

        self.directed = directed
        self.self_loop = self_loop
        self.featurizer = featurizer

        self.reset()

    def reset(self):
        if isinstance(self.iterator, random.Random):
            doc_idx = self.iterator.randrange(len(self.dataset[self.subset]))
        else:
            try:
                doc_idx = next(self.iterator)
            except StopIteration:
                self.iterator = iter(range(len(self.dataset[self.subset])))
                doc_idx = next(self.iterator)

        self.doc = self.dataset[self.subset][doc_idx]
        
        self.query = SentenceNode(
            sent=self.doc["question"], 
            sent_id=self.doc["id"],
            featurizer=self.featurizer,
        )

        self.doc_graph = DocumentGraph(
            doc=self.doc["supports"], 
            doc_id=self.doc["id"], 
            directed=self.directed,
            featurizer=self.featurizer,
            self_loop=self.self_loop,
        )

        self.cur_node = self.doc_graph.graph
        self.timestep = 0
        self.reward = 0
        self.done = False
        
    
    def step(self, action):
        self.cur_node = action
        self.timestep += 1
        
        if self.timestep==self.max_steps:
            self.done = True
            if self.is_answer_node(self.cur_node):
                self.reward = 1

        return self.cur_node, self.done, self.reward

    def is_answer_node(self, cur_node):
        ans_words = set(self.doc["answer"].lower().split())
        node_words = set(str(cur_node.sent).lower().split())
        intersection = set.intersection(ans_words, node_words)
        return len(intersection) == len(ans_words)
    
    @property
    def state(self):
        return self.cur_node

    def get_actions(self, state):
        return state.children
