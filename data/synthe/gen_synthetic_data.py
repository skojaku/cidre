# %%
from networkx.algorithms.covering import min_edge_cover
import numpy as np
from scipy import sparse
import pandas as pd

#
# Baseline
#
def gen_deg(alpha, dave, N):
    deg = 1 / np.random.power(alpha, N)
    return np.maximum(deg * dave * N / np.sum(deg), 1)


def gen_noisy_dcSBM(indeg, outdeg, gids, mix_rate, is_bipartite=False):
    N = indeg.size
    K = np.max(gids) + 1

    U = sparse.csr_matrix(
        (np.ones_like(gids), (np.arange(gids.size), gids)), shape=(N, K)
    )
    Din = np.array(indeg.T @ U).reshape(-1)
    Dout = np.array(outdeg.T @ U).reshape(-1)

    Arand = np.outer(outdeg, indeg) / np.sum(indeg)
    if is_bipartite:
        Acom = (
            sparse.diags(outdeg / np.maximum(Dout[gids], 1))
            @ U
            @ np.array([[0, np.sum(Dout)], [np.sum(Din), 0]])
            @ U.T
            @ sparse.diags(indeg / np.maximum(Din[gids], 1))
        )
    else:
        Acom = (
            sparse.diags(outdeg / np.maximum(Dout[gids], 1))
            @ U
            @ sparse.diags((Dout + Din) / 2)
            @ U.T
            @ sparse.diags(indeg / np.maximum(Din[gids], 1))
        )
    A = mix_rate * Arand + (1 - mix_rate) * Acom
    A = A - np.diag(np.diag(A))

    return np.random.poisson(A)


def gen_anomalous_group(
    nodes, N, Nd, indeg_base, outdeg_base, inflation_rate, mix_rate
):
    Nc = len(nodes)
    donors = np.random.choice(Nc, Nd, replace=False)
    recipients = np.array(list(set(np.arange(Nc)).difference(donors)))
    Nr = Nc - Nd

    indeg = np.zeros_like(nodes)
    outdeg = np.zeros_like(nodes)
    gids = np.zeros_like(nodes)
    gids[recipients] = 1

    theta = inflation_rate / (1 - inflation_rate)

    indeg[recipients] = indeg_base[nodes[recipients]] * theta
    outdeg[donors] = outdeg_base[nodes[donors]] * theta

    alpha = np.sum(indeg) / np.sum(outdeg)
    if alpha > 1:
        outdeg = outdeg * alpha
    else:
        indeg = indeg / alpha

    B = gen_noisy_dcSBM(indeg, outdeg, gids, mix_rate, is_bipartite=True)
    U = sparse.csr_matrix(
        (np.ones_like(gids), (nodes, np.arange(gids.size))), shape=(N, Nc)
    )
    return U @ B @ U.T


if __name__ == "__main__":
    output_node_file = "node-table.csv"
    output_edge_file = "edge-table.csv"

    N = 100  # Number of nodes
    dave = 200  # average degree
    K = 2  # Number of communities
    mix_rate = 0.2  # Level of noise
    gamma = 3  # degree exponent

    Kc = 5  # Number of anomalous groups
    Nc = 5  # Number of nodes in an anomalous group
    Nd = 2  # Number of donors
    inflation_rate = 0.15  # fraction of edge weights inflated by anomalous groups
    min_edge_weight = 3

    # %% Generate base graph
    outdeg = gen_deg(gamma, dave, N)
    indeg = gen_deg(gamma, dave, N)
    alpha = np.sum(indeg) / np.sum(outdeg)
    if alpha > 1:
        outdeg = outdeg * alpha
    else:
        indeg = indeg / alpha
    gids = np.random.randint(0, K, N)
    gids = np.sort(gids)

    Abase = gen_noisy_dcSBM(indeg, outdeg, gids, mix_rate)

    indeg = np.array(Abase.sum(axis=0))
    outdeg = np.array(Abase.sum(axis=1))

    # %% Generate anomalous groups
    nodes = np.arange(N)
    group_ids = np.ones(N) * np.nan
    A = Abase.copy()
    for k in range(Kc):
        while True:
            _nodes = np.random.choice(nodes, Nc, replace=False)
            if len(set(gids[_nodes])) == 1:
                break

        Ak = gen_anomalous_group(_nodes, N, Nd, indeg, outdeg, inflation_rate, 0)
        A += Ak
        nodes = nodes[~np.isin(nodes, _nodes)]
        group_ids[_nodes] = k

    pd.DataFrame(
        {"id": np.arange(N), "group_id": gids, "anomalous_group_id": group_ids}
    ).to_csv(output_node_file)

    r, c, v = sparse.find(A)

    s = v >= min_edge_weight
    r, c, v = r[s], c[s], v[s]

    pd.DataFrame({"src": r, "trg": c, "weight": v}).to_csv(output_edge_file)

# %%

# %%
