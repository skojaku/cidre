# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # About this notebook
#
# In this notebook, we apply CIDRE to a network with communities and demonstrate how to use CIDRE and visualize the detected groups.
#
# To start, we'll need some libraries.
#

import sys
import numpy as np
from scipy import sparse
import pandas as pd
import cidre

# Next, we load a network. We first present an example of a small artificial network and then a larger empirical network.

# +
# Data path
edge_file = "../data/synthe/edge-table.csv"
node_file = "../data/synthe/node-table.csv"

# Load
node_table = pd.read_csv(node_file)
A, node_labels = cidre.utils.read_edge_list(edge_file)
# -

# We constructed this synthetic network by generating a network using a stochastic block model (SBM) composed of two blocks and then adding excessive citation edges among uniformly randomly selected pairs of nodes. Each block corresponds to a community, i.e., a group of nodes that are densely connected with each other within it but sparsely connected with those in the opposite group. Such communities overshadow anomalous groups in networks. 

# Let's pretend that we do not know that the network is composed of two communities and additional edges. To run CIDRE, we first need to find the communities. To use `graph-tool` package to this end, install `graph-tool` by 

# !conda install -y -c conda-forge graph-tool

# Now, let's detect communities using `graph-tool` by fitting the degree-corrected stochastic block model (dcSBM) to the network and consider each block as a community.
#
# Our approach hinges on the assumption that the anomalous groups are not detected by the community detection algorithm. If we allow the number of communities to be large in this example, we would find anonalous groups as communities. Therefore, we limit the number of communities to be at most 3 to prevent the SBM to detect small anomalous groups in the network. This assumption looks artificial, but we do not need to impose it in the case of large networks because small anomalous groups usually have little impact on the global community structure, and thus SBM would not find them as communities.

import graph_tool.all as gt
def detect_community(A, K = None, **params):
    """Detect communities using graph-tool package

    :param A: adjacency matrix
    :type A: scipy.csr_sparse_matrix
    :param K: Maximum number of communities
    :type K: int
    :param params: parameters passed to graph_tool.gt.minimize_blockmodel_dl
    """
    def to_graph_tool_format(adj, membership=None):
        g = gt.Graph(directed=True)
        r, c, v = sparse.find(adj)
        nedges = v.size
        edge_weights = g.new_edge_property("double")
        g.edge_properties["weight"] = edge_weights
        g.add_edge_list(
            np.hstack([np.transpose((r, c)), np.reshape(v, (nedges, 1))]),
            eprops=[edge_weights],
        )
        return g
    G = to_graph_tool_format(A)

    states = gt.minimize_blockmodel_dl(
        G,
        state_args=dict(eweight=G.ep.weight),
        multilevel_mcmc_args = {"B_max": K },
        **params
    )
    b = states.get_blocks()
    return np.unique(np.array(b.a), return_inverse = True)[1]


group_membership = detect_community(A, K = 3) 

# Now, we input a network and its community structure to CIDRE. To to this,　we create a `cidre.Cidre` object and input `group_membership` along with some key parameters to `cidre.Cidre`.

alg = cidre.Cidre(group_membership = group_membership, alpha = 0.05, min_edge_weight = 1)

# - `alpha` (default 0.01) is the statistical significance level.
# - `min_edge_weight` is the threshold of the edge weight, i.e., the edges with weight less than this value will be removed.
#
# Then, input the network in form of `scipy.sparse_csr` matrix or `nx.Graph` to `cidre.Cidre.detect`.

groups = alg.detect(A, threshold=0.15)

# `groups` is a list of `Group` instances. We can get the donor nodes of a group, for example `groups[0]`, by 

groups[0].donors

# The keys and values of this dict object are the IDs of the nodes and their donor scores, respectively. The recipients and their recipient scores can be obtained by 

groups[0].recipients

# # Visualization 
#
# `cidre` package provides an API to visualize small groups. To use this API, first of all, we need some additional libraries.

import seaborn as sns
import matplotlib.pyplot as plt

# Then plot the group by

# +
# The following three lines are purely for visual enhancement, i.e., changing the saturation of the colors and font size.
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

# Set the figure size
width, height = 5,5
fig, ax = plt.subplots(figsize=(width, height))

# Plot a citation group
cidre.DrawGroup().draw(groups[0], ax = ax)
