"""Edge filtering function"""
import numpy as np
from scipy import stats, sparse
import sys
from cidre import utils
from functools import partial


class EdgeFilter:
    def __init__(
        self, min_edge_weight=0, alpha=0.01, remove_selfloop=True, min_expected_weight=1
    ):
        self.alpha = alpha
        self.min_edge_weight = min_edge_weight
        self.remove_selfloop = remove_selfloop
        self.min_expected_weight = min_expected_weight

    def fit(self, A, group_membership, mask=None):
        """Find the excessive edges in the network using the dcSBM as a null model.

        :param A: Network
        :type A: scipy sparse matrix
        :param group_membership: group membership of nodes. group_membership[i] is the group to which node i belongs.
        :type group_membership: numpy.array
        :param mask: mask[i,j] = 1 to set edge (i,j) to be insignificant, defaults to None
        :type mask: scipy sparse matrix, optional
        """
        if group_membership is None:
            group_membership = np.zeros(A.shape[0]).astype(int)
        p_value, src, dst, w = self._calc_p_values_dcsbm(
            A, group_membership, self.min_expected_weight
        )

        if self.remove_selfloop:
            s = src != dst
            p_value, src, dst, w = p_value[s], src[s], dst[s], w[s]

        # Remove edges less than minimum weight
        if self.min_edge_weight > 0:
            s = w >= self.min_edge_weight
            p_value, src, dst, w = p_value[s], src[s], dst[s], w[s]

        # Mask pre-selected edges
        if mask is not None:
            wth = np.array(mask[(src_, trg_)]).reshape(-1)
            s = np.isclose(wth, 0)
            p_value, src, dst, w = p_value[s], src[s], dst[s], w[s]

        # Perform the Benjamini-Hochberg statistical test
        is_significant = self._benjamini_hochberg_test(p_value)

        # Find the excessive edges
        src, dst, w = src[is_significant], dst[is_significant], w[is_significant]

        # Construct the filter
        self.filter = self._make_filter_func(src, dst, None, A.shape[0])

    def transform(self, src, dst, w):
        return self.filter(src, dst, w)

    def _calc_p_values_dcsbm(self, A, group_membership, min_expected_weight=1):
        """Calculate the p_values using the degree-corrected stochastic block model.

        :param A: Adjacency matrix. Adjacency matrix, where A[i,j] indicates the weight of the edge from node i to node j.
        :type A: scipy sparse matrix
        :param group_membership: group_membership[i] indicates the ID of the group to which node i belongs
        :type group_membership: numpy.array
        :return: p-values
        :rtype: float
        """

        N = A.shape[0]
        indeg = np.array(A.sum(axis=0)).reshape(-1)
        outdeg = np.array(A.sum(axis=1)).reshape(-1)
        C_SBM = utils.to_community_matrix(group_membership)

        Lambda = C_SBM.T @ A @ C_SBM
        Din = np.array(Lambda.sum(axis=0)).reshape(-1)
        Dout = np.array(Lambda.sum(axis=1)).reshape(-1)

        theta_in = indeg / np.maximum(C_SBM @ Din, 1.0)
        theta_out = outdeg / np.maximum(C_SBM @ Dout, 1.0)

        src, dst, w = utils.find_non_self_loop_edges(A)
        lam = (
            np.array(Lambda[group_membership[src], group_membership[dst]]).reshape(-1)
            * theta_out[src]
            * theta_in[dst]
        )
        lam = np.maximum(lam, min_expected_weight)
        pvals = 1.0 - stats.poisson.cdf(w - 1, lam)

        return pvals, src, dst, w

    def _benjamini_hochberg_test(self, pvals):
        """Benjamini-Hochberg statistical test

        :param pvals: p-values
        :type pvals: numpy.array
        :return: significant[i] = True if the ith element is significant. Otherwise signfiicant[i] = False.
        :rtype: numpy.array (bool)
        """
        order = np.argsort(pvals)
        M = pvals.size
        is_sig = pvals[order] <= (self.alpha * np.arange(1, M + 1) / M)
        if np.any(is_sig) == False:
            return is_sig

        last_true_id = np.where(is_sig)[0][-1]
        is_sig = np.zeros(M)
        is_sig[order[: (last_true_id + 1)]] = 1
        return is_sig > 0

    def _make_filter_func(self, src, trg, wth, N):
        """Make a filter function

        :param src: Source node
        :type src: np.array
        :param trg: Target node
        :type trg: np.array
        :param wth: Minimum Weight of edges between source and target nodes
        :type wth: numpy.array
        :param N: Number of nodes
        :type N: int
        :return: Filtering function
        :rtype: function
        """
        if wth is None:
            wth = 1e-8 * np.ones_like(src)

        # Convert pairs of integers into integers
        W = sparse.csr_matrix((wth, (src, trg)), shape=(N, N))

        # Define the is_excessive function for CIDRE
        def is_excessive(src_, trg_, w_, W):
            wth = np.array(W[(src_, trg_)]).reshape(-1)
            s = (w_ >= wth) * (wth > 0)
            return src_[s], trg_[s], w_[s]

        return partial(is_excessive, W=W)
