[![Unit Test & Deploy](https://github.com/skojaku/cidre/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/cidre/actions/workflows/main.yml)
# Python package for the CItation-Donor-REcipient (CIDRE) algorithm.

Please cite:

Kojaku, S., Livan, G. & Masuda, N. Detecting anomalous citation groups in journal networks. Sci Rep 11, 14524 (2021). https://doi.org/10.1038/s41598-021-93572-3. 

```latex


@ARTICLE{Kojaku2021,
  title     = "Detecting anomalous citation groups in journal networks",
  author    = "Kojaku, Sadamori and Livan, Giacomo and Masuda, Naoki",
  journal   = "Sci. Rep.",
  publisher = "Nature Publishing Group",
  volume    =  11,
  number    =  1,
  pages     = "1--11",
  month     =  jul,
  year      =  2021,
}


```

## Requirements
- Python 3.7 or later

## Install

```
pip install cidre
```

## Examples
- [Toy network with communities](examples/example.ipynb)

## A minimal example

```python
import cidre

alg = cidre.Cidre(group_membership)
groups = alg.detect(A, threshold = 0.15)
```
- `group_membership`: If the network has communities, and the communities are not anomalous, tell the community membership to CIDRE through this argument, where `group_membership[i]` indicates the group to which node i belongs. Otherwise, set `group_membership=None`.
- `A`: Adjacency matrix of the input network (can be weighted or directed). Should be either an nx.Graph or scipy.sparse_matrix. If it is in the scipy.sparse_matrix format, A[i,j] indicates the weight of edge from node i to j.
- `threshold`: Threshold for the donor and recipient nodes. A larger threshold will yield tighter and smaller groups.
- `groups`: Detected groups. This is a list of special class, `Group`.

`groups` is a list of `group`s. Each element in the list, `group`, contains the IDs of the member nodes with their roles. 

The donors of a group are given by
```python
group.donors # {node_id: donor_score}
```
- `group.donors` is a dict object, with keys and values corresponding to the node ID and the donor score.

Similarly, the recipients of a group together with their recipient scores are given by
```python
group.recipients # {node_id: recipient_score}
```

## Visualization

```
ax = plt.gca()
dc = cidre.DrawCartel()
dc.draw(group, ax = ax)
```

The labels beside the nodes are the ID of the nodes, or equivalently row ids of the adjacency matrix `A`.

To put the node labels, make a dictionary from the ID to label, like `node_labels = {0:"name", 1:"name 2"}`, and pass it by `node_labels = node_labels`.

