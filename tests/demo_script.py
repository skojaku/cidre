# %%
import sys
import numpy as np
from scipy import sparse
import pandas as pd

edge_file = "../data/edges.csv"
df = pd.read_csv("../data/edges.csv")
src, trg, w = df["source"].values, df["target"].values, df["weight"].values

N = np.maximum(np.max(src), np.max(trg)) + 1
A = sparse.csr_matrix((w, (src, trg)), shape=(N, N))
group_membership = np.random.randint(0, 3, N)

# %%
import cidre

model = cidre.Cidre()
groups = model.detect(A, threshold=0.16)
# %%
# groups[0].get_donors()
# groups[0].get_recipients()
groups[0].get_donor_recipients()
# %%
