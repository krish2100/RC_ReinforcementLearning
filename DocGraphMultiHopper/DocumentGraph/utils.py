import torch
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SENT_MODEL = SentenceTransformer(
    "sentence-transformers/bert-base-nli-mean-tokens"
).to(DEVICE)

def sentence_featurizer(sent, model=SENT_MODEL):
    return torch.tensor(model.encode(sent))