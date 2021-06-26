import logging
import sys
import unittest

import numpy as np
from scipy import sparse
import pandas as pd

import cidre


class Test(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv("../data/edges.csv")
        src = df["source"].values
        trg = df["target"].values
        w = df["weight"].values
        self.N = np.maximum(np.max(src), np.max(trg)) + 1
        self.A = sparse.csr_matrix((w, (src, trg)), shape=(self.N, self.N))
        self.group_membership = np.random.randint(0, 3, self.N)

    def test_edge_filtering(self):
        efilter = cidre.EdgeFilter()
        efilter.fit(self.A, group_membership=self.group_membership)

    def test_detect(self):
        model = cidre.CIDRE(group_membership=self.group_membership)
        groups = model.detect(self.A, threshold=0.15)


if __name__ == "__main__":
    unittest.main()
