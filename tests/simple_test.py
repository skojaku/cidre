import logging
import sys
import unittest

import numpy as np
import cidre


class Test(unittest.TestCase):
    def setUp(self):
        A, node_labels = cidre.utils.read_edge_list("data/synthe/edge-table.csv")
        self.N = A.shape[0] 
        self.A = A 
        self.group_membership = np.random.randint(0, 3, self.N)

    def test_detect(self):
        model = cidre.Cidre(group_membership=self.group_membership)
        groups = model.detect(self.A, threshold=0.15)


if __name__ == "__main__":
    unittest.main()
