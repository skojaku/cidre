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

# + [markdown] id="KICTR2HtEcU9"
# # About this notebook
#
# In this notebook, we apply CIDRE to a network with communities and demonstrate how to use CIDRE and visualize the detected groups.

# + [markdown] id="aLxSCdt2iRVl"
# ## Preparation

# + [markdown] id="ILo2xh5yico6"
# ### Install CIDRE package
#
# First, we install `cidre` package with `pip`:

# + id="MJtwwRBcF8iL"
# !pip install cidre

# + [markdown] id="1URuALNHF6f8"
#
# ### Loading libraries
#
# Next, we load some libraries

# + id="aKnjeGnnEcU_"
import sys
import numpy as np
from scipy import sparse
import pandas as pd
import cidre
import networkx as nx

# + [markdown] id="ifRxK9fbEcVA"
# # Example 1
#
# We first present an example of a small artificial network, which can be loaded by

# + colab={"base_uri": "https://localhost:8080/", "height": 319} id="dsLhebpREcVA" outputId="9daf09e9-5303-4f1d-8a00-4da392d40cf1"
# Data path
edge_file = "https://raw.githubusercontent.com/skojaku/cidre/main/data/synthe/edge-table.csv"
node_file = "https://raw.githubusercontent.com/skojaku/cidre/main/data/synthe/node-table.csv"

# Load
node_table = pd.read_csv(node_file)
A, node_labels = cidre.utils.read_edge_list(edge_file)

# Visualization
nx.draw(nx.from_scipy_sparse_matrix(A), linewidths = 1, edge_color="#8d8d8d", edgecolors="b")

# + [markdown] id="2PDobGDcEcVB"
# ## About this network
#
# We constructed this synthetic network by generating a network using a stochastic block model (SBM) composed of two blocks and then adding excessive citation edges among uniformly randomly selected pairs of nodes. Each block corresponds to a community, i.e., a group of nodes that are densely connected with each other within it but sparsely connected with those in the opposite group. Such communities overshadow anomalous groups in networks. 

# + [markdown] id="M7Nnfwk1bWgw"
# ## Community detection with graph-tool

# + [markdown] id="5PsfTNsQEcVB"
# Let's pretend that we do not know that the network is composed of two communities plus additional edges. To run CIDRE, we first need to find the communities. We use `graph-tool` package to do this, which can be installed `graph-tool` by 
#
# ```python
# conda install -c conda-forge graph-tool
# ```
#
# or in `Colaboratory` platform:

# + id="iNUpxKwTEcVB"
# %%capture
# !echo "deb http://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list
# !apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
# !apt-get update
# !apt-get install python3-graph-tool python3-cairo python3-matplotlib

# + [markdown] id="w8HYpWMhEcVC"
# Now, let's detect communities by fitting the degree-corrected stochastic block model (dcSBM) to the network and consider each detected block as a community.

# + colab={"base_uri": "https://localhost:8080/"} id="h3qtB94jEcVC" outputId="85a56609-c84b-42a1-f7b8-f7a030db39a9"
import graph_tool.all as gt

def detect_community(A, K = None, **params):
    """Detect communities using the graph-tool package

    :param A: adjacency matrix
    :type A: scipy.csr_sparse_matrix
    :param K: Maximum number of communities. If K = None, the number of communities is automatically determined by graph-tool.
    :type K: int or None
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
        multilevel_mcmc_args = {"B_max": A.shape[0] if K is None else K },
        **params
    )
    b = states.get_blocks()
    return np.unique(np.array(b.a), return_inverse = True)[1]


# + id="mZnRHmCHEcVD"
group_membership = detect_community(A)

# + [markdown] id="z3fE7Qq9EcVD"
# ## Detecting anomalous groups in the network
#
# Now, we feed the network and its community structure to CIDRE. To to this, we create a `cidre.Cidre` object and input `group_membership` along with some key parameters to `cidre.Cidre`.

# + id="NBCEKVBZEcVD"
alg = cidre.Cidre(group_membership = group_membership, alpha = 0.05, min_edge_weight = 1)

# + [markdown] id="EvvHYL_lEcVD"
# - `alpha` (default 0.01) is the statistical significance level.
# - `min_edge_weight` is the threshold of the edge weight, i.e., the edges with weight less than this value will be removed.
#
# Then, we input the network to `cidre.Cidre.detect`.

# + id="UGmFBaLbEcVE"
groups = alg.detect(A, threshold=0.15)

# + [markdown] id="60Fp9LA9EcVE"
# `groups` is a list of `Group` instances. A `Group` instance represents a group of nodes detected by CIDRE, and contains information about the type of each member node (i.e., donor and recipient). We can get the donor nodes of a group, for example `groups[0]`, by

# + colab={"base_uri": "https://localhost:8080/"} id="9wU-fvA2EcVE" outputId="ff6fc117-4e4f-498f-fb04-4bd585743b35"
groups[0].donors

# + [markdown] id="CEPElXKkEcVE"
# The keys and values of this dict object are the IDs of the nodes and their donor scores, respectively. The recipients and their recipient scores can be obtained by

# + colab={"base_uri": "https://localhost:8080/"} id="aTFc_o5BEcVE" outputId="848ea1fc-3bd8-4eaa-b241-351c6846238a"
groups[0].recipients


# + [markdown] id="vo19Md36EcVF"
# ## Visualization
#
# `cidre` package provides an API to visualize small groups. To use this API, we first need to import some additional libraries.

# + id="lVQbfWDVEcVF"
import seaborn as sns
import matplotlib.pyplot as plt

# + [markdown] id="W0vYp-lIEcVF"
# Then, plot the group by

# + colab={"base_uri": "https://localhost:8080/", "height": 332} id="au-3a1LHEcVF" outputId="c8000c8e-e541-415e-b334-c9de9302028d"
# The following three lines are purely for visual enhancement, i.e., changing the saturation of the colors and font size.
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

# Set the figure size
width, height = 5,5
fig, ax = plt.subplots(figsize=(width, height))

# Plot a citation group
cidre.DrawGroup().draw(groups[0], ax = ax)

# + [markdown] id="WaAFxBmxZ1d6"
#
# # Example 2
#
# Let's apply CIDRE to a much large empirical citation network, i.e., the citation network of journals in 2013.

# + id="SWxvzmnTEcVG"
# Data path
edge_file = "https://raw.githubusercontent.com/skojaku/cidre/main/data/journal-citation/edge-table-2013.csv"
node_file = "https://raw.githubusercontent.com/skojaku/cidre/main/data/journal-citation/community-label.csv"

# Load
node_table = pd.read_csv(node_file)
A, node_labels = cidre.utils.read_edge_list(edge_file)

# + [markdown] id="RORUAoB4EcVG"
# ## About this network
#
# This network is a citation network of journals in 2013 constructed from Microsoft Academic Graph.
# Each edge is weighted by the number of citations made to papers in the prior two years.
# The following are the basic statistics of this network.

# + colab={"base_uri": "https://localhost:8080/"} id="dnY8niyoEcVG" outputId="4613a203-2fb9-43ac-8bee-df085a6d8594"
print("Number of nodes: %d" % A.shape[0])
print("Number of edges: %d" % A.sum())
print("Average degree: %.2f" % (A.sum()/A.shape[0]))
print("Max in-degree: %d" % np.max(A.sum(axis = 0)))
print("Max out-degree: %d" % np.max(A.sum(axis = 1)))
print("Maximum edge weight: %d" % A.max())
print("Minimum edge weight: %d" % np.min(A.data))

# + [markdown] id="x_IAlmL-EcVH"
# ## Communities
#
# [In our paper](https://www.nature.com/articles/s41598-021-93572-3), we identified the communities of journals using the graph-tool. `node_table` contains the community membership of each journal, from which we prepare `group_membership` array as follows.

# + id="xlaLIDjwEcVH"
# Get the group membership
node2com = dict(zip(node_table["journal_id"], node_table["community_id"]))
group_membership = [node2com[node_labels[i]] for i in range(A.shape[0])]

# + [markdown] id="_F1gHjpLlnyY"
# ## Detecting anomalous groups in the network

# + [markdown] id="vKzcAPkQEcVI"
# As is demonstrated in the first example, we detect the anomalous groups in the network by

# + id="kQEqAIJoEcVI"
alg = cidre.Cidre(group_membership = group_membership, alpha = 0.01, min_edge_weight = 10)
groups = alg.detect(A, threshold=0.15)

# + colab={"base_uri": "https://localhost:8080/"} id="HHqk6MOEEcVI" outputId="19c23bde-34d6-4118-b110-494d812095c7"
print("The number of journals in the largest group: %d" % np.max([group.size() for group in groups]))
print("Number of groups detected: %d" % len(groups))

# + [markdown] id="AJjTNeJecyos"
# [In our paper](https://www.nature.com/articles/s41598-021-93572-3), we omitted the groups that have within-group citations less than 50 because we expect that anomalous citation groups contain sufficiently many within-group citations, i.e., 

# + id="wHvlyy6EdYQm"
groups = [group for group in groups if group.get_num_edges()>=50]

# + [markdown] id="cqg3nU-deGfh"
# where `group.get.num_edges()` gives the sum of the weights of the non-self-loop edges within the group.

# + [markdown] id="SMylhKMNEcVI"
# ## Visualization

# + [markdown] id="sFWZTKxuEcVJ"
# Let us visualize the groups detected by CIDRE. We random sample three groups to visualize.

# + id="MvNno-z9EcVJ"
groups_sampled = [groups[i] for i in np.random.choice(len(groups), 3, replace = False)]

# + colab={"base_uri": "https://localhost:8080/", "height": 327} id="tp5hJeIiEcVJ" outputId="e1c9b1ed-d11b-4cff-c374-0873fe90e107"
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, axes = plt.subplots(ncols = 3, figsize=(6 * 3, 5))

for i in range(3):
    cidre.DrawGroup().draw(groups_sampled[i], ax = axes.flat[i])

# + [markdown] id="c-OTsIFdEcVJ"
# The numbers beside the nodes are the IDs of the journals in the network. You may want to see the journal names and here is how to display them. 
#
#
# First, we load node lables and make a dictionary from the ID of each node to the label:

# + id="Q67C_AJwEcVJ"
df = pd.read_csv("https://raw.githubusercontent.com/skojaku/cidre/main/data/journal-citation/journal_names.csv")
journalid2label = dict(zip(df.journal_id.values, df.name.values)) # Dictionary from MAG journal ID to the journal name

id2label = {k:journalid2label[v] for k, v in node_labels.items()} # This is a dictionary from ID to label, i.e., {ID:journal_name}


# + [markdown] id="4JxPElAoghPl"
# Then, give `id2label` to `cidre.DrawGroup.draw`, i.e.,  

# + colab={"base_uri": "https://localhost:8080/", "height": 390} id="uIQ7XcirEcVJ" outputId="513e0ea1-3729-429b-b1e8-40bc4687f759"
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

fig, axes = plt.subplots(ncols = 3, figsize=(8 * 3, 5))

for i in range(3):
    plotter = cidre.DrawGroup()
    plotter.font_size = 12 # Font size
    plotter.label_node_margin = 0.5 # Margin between labels and node
    plotter.draw(groups_sampled[i], node_labels = id2label, ax = axes.flat[i])
