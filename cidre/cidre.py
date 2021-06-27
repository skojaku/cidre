import numpy as np
from scipy import sparse
import pandas as pd
import networkx as nx
from cidre import utils
from cidre import filters


class Group:
    """Data class for anomalous node groups

    A node group consists of donors and recipients.
    Donors excessively provide edges to other nodes in the group.
    Recipients excessively receive edges from nodes in the group.

    The level of donorness and recipientness are measured by
    donor and recipient scores provided by an external algorithm.

    Then, donor and recipient nodes are identified by a threshold.
    """

    def __init__(self, node_ids, donor_scores, recipient_scores, threshold):
        """Initialize the group

        :param node_ids: ids of nodes
        :type node_ids: numpy.array
        :param donor_scores: donor scores.
        :type donor_scores: numpy.array
        :param recipient_scores: recipient scores
        :type recipient_scores: numpy.array
        :param threshold: threshold for classifying nodes into donors and recipients
        :type threshold: float
        """
        self.node_ids = node_ids.copy()
        self.recipient_score = recipient_scores.copy()
        self.donor_score = donor_scores.copy()
        self.threshold = threshold

        is_donor, is_recipient = (
            donor_scores >= threshold,
            recipient_scores >= threshold,
        )
        donors, recipients = node_ids[is_donor], node_ids[is_recipient]
        donor_scores, recipient_scores = (
            donor_scores[is_donor],
            recipient_scores[is_recipient],
        )
        self.donors = dict(zip(donors, donor_scores))
        self.recipients = dict(zip(recipients, recipient_scores))

    def size(self):
        return len(self.node_ids)

    def set_within_net(self, A, node_ids):
        brow = A[:, node_ids].sum(axis=0)
        bcol = A[node_ids, :].sum(axis=1)
        As = A[:, node_ids][node_ids, :].toarray()

        self.num_within_edges = As.sum() - As.diagonal().sum()

        # Add 'Other' node to the adjacency matrix
        B = np.block([[As, bcol], [brow, np.array([0])]])
        self.A = np.array(B)

    def get_within_net(self):
        return self.A

    def get_num_edges(self):
        return self.num_within_edges

    def get_donors(self):
        return self.donors

    def get_recipients(self):
        return self.recipients

    def get_donor_recipients(self):
        ids = set(list(self.donors.keys())).intersection(
            set(list(self.recipients.keys()))
        )
        return {
            i: {"donor": self.donors[i], "recipient": self.recipients[i]} for i in ids
        }


class Cidre:
    def __init__(
        self,
        min_edge_weight=0,
        min_expected_weight=1,
        group_membership=None,
        alpha=0.01,
    ):
        if isinstance(group_membership, list):
            group_membership = np.array(group_membership)

        self.group_membership = group_membership
        self.edge_filter = filters.EdgeFilter(
            alpha=alpha,
            min_edge_weight=min_edge_weight,
            min_expected_weight=min_expected_weight,
            remove_selfloop=True,
        )

    def detect(self, G, threshold):
        """Detecting anomalous groups with excessive donors and recipients.

        :param G: Network. If G is a sparse matrix, entry G[i,j] should be the weight of the edge from node i to j.
        :type G: scipy sparse matrix or networkx.Graph.
        :param threshold: Threshold for donor and recipient scores. A higher thhreshold yields smaller and tighter groups
        :type threshold: float
        """
        #
        # Input parsing
        #
        A, node_labels = utils.to_adjacency_matrix(G)
        #
        # Edge filtering
        #
        self.edge_filter.fit(A, group_membership=self.group_membership)
        src, dst, w = sparse.find(A)
        src, dst, w = self.edge_filter.transform(src, dst, w)

        Abar = utils.construct_adjacency_matrix(
            src, dst, w, A.shape[0]
        )  # Edge filtered network

        #
        # Node filtering
        #
        num_nodes = A.shape[0]  # number of nodes
        U = np.ones(num_nodes)  # nodes not truncated
        indeg = np.maximum(
            np.array(A.sum(axis=0)).ravel(), 1.0
        )  # In-degree. Clip to one to avoid zero divide.
        outdeg = np.maximum(
            np.array(A.sum(axis=1)).ravel(), 1.0
        )  # Out-degree. Clip to one to avoid zero divide.
        while True:
            # Compute the donor score, recipient score and cartel score
            donor_score = np.multiply(U, (Abar @ U) / outdeg)
            recipient_score = np.multiply(U, (U @ Abar) / indeg)

            # Drop the nodes with a cartel score < threshold
            drop_from_U = (U > 0) * (
                np.maximum(donor_score, recipient_score) < threshold
            )

            # Break the loop if no node is dropped from the cartel
            if np.any(drop_from_U) == False:
                break

            # Otherwise, drop the nodes from the cartel
            U[drop_from_U] = 0

        # Find the nodes in U
        survived_nodes = np.where(U)[0]
        Abar = Abar[:, survived_nodes][survived_nodes, :].copy()

        # Partition U into disjoint groups, U_l
        anomalous_group_list = []
        Gbar = nx.from_scipy_sparse_matrix(Abar, create_using=nx.DiGraph)
        Gbar.remove_nodes_from(list(nx.isolates(Gbar)))
        for _, _nd in enumerate(nx.weakly_connected_components(Gbar)):
            _nodes = survived_nodes[np.array(list(_nd))]

            # Remove the group U_l if
            # U_l does not contain edges less than or equal to
            # min_group_edge_num
            # A_Ul = A[_nodes, :][:, _nodes]

            group = Group(
                _nodes, donor_score[_nodes], recipient_score[_nodes], threshold
            )
            group.set_within_net(A, _nodes)
            anomalous_group_list += [group]

        return anomalous_group_list
