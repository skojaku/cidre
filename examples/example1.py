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
# In this notebook, we will apply CIDRE to a network with communities and demonstrate how to use CIDRE and visualize the detected groups.
#
# To start, we'll need some basic libraries.
#

import sys
import numpy as np
from scipy import sparse
import pandas as pd
import cidre

# Next, we will need a network to test CIDRE. 

# +
# Data path
edge_file = "../data/synthe/edge-table.csv"
node_file = "../data/synthe/node-table.csv"

# Load
node_table = pd.read_csv(node_file)
A, node_labels = cidre.utils.read_edge_list(edge_file)
# -

# This network consists of several communities. The group membership can be obtained by

# Get group membership
group_membership = node_table["group_id"]

# Now, we have all data to run CIDRE. Create `cidre.Cidre` object and pass `group_membership` along with some key parameters.

alg = cidre.Cidre(group_membership = group_membership, alpha = 0.01, min_edge_weight = 1)

# - `alpha` (defaults to 0.01) is the statistical significance level for anomalous edges.
# - `min_edge_weight` is the threshold of the edge weights, i.e., the edges with weight less than this value will be removed prior to detecting groups.
#
# Then, pass the network in form of `scipy.sparse_csr` matrix or `nx.Graph`.

groups = alg.detect(A, threshold=0.15)

# `groups` is a list of `Group` instance. We can get the donor nodes of a group, for example `groups[0]`, by 

groups[0].donors

# The keys and values of this dict object are the IDs of the node and their donor scores, respectively. The recipients and their recipient scores can be obtained by 

groups[0].recipients

# # Visualization 
#
# `cidre` package provides an API to visualize small groups. To use this API, first of all, we need some additional libraries

import seaborn as sns
import matplotlib.pyplot as plt

# Then plot the group by

# +
# The following three lines are purely for visual enhancement, i.e., changing the saturation of the colors and font size.
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

# Set up the canvs
width, height = 5,5
fig, ax = plt.subplots(figsize=(width, height))

# Plot with cidre package
cidre.DrawGroup().draw(groups[0], ax = ax)
