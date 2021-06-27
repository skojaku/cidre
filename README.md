[![Unit Test & Deploy](https://github.com/skojaku/cidre/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/cidre/actions/workflows/main.yml)
# Python package for the CItation-Donor-REcipient (CIDRE) algorithm.

Please cite:

```latex
@misc{kojaku2021detecting,
      title={Detecting citation cartels in journal networks}, 
      author={Sadamori Kojaku and Giacomo Livan and Naoki Masuda},
      year={2021},
      eprint={2009.09097},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph}
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
- [Citation network of journals in 2013](examples/example2.ipynb)

## Minimal example

```python
import cidre

alg = cidre.Cidre(group_membership)
groups = alg.detect(A, threshold = 0.15)
```
- `group_membership`: If the network has communities, and the communities are not anomalous, tell the community membership to CIDRE through this argument, where `group_membership[i]` indicates the group to which node i belongs. Otherwise, set `group_membership=None`.
- `A`: Adjacency matrix of the input network (can be weighted and directed). nx.Graph or scipy.sparse_matrix. if scipy.sparse_matrix format, A[i,j] indicates the weight of edge from node i to j.
- `threshold`: Threshold for the donor and recipient nodes. A larger threshold will yield tighter and smaller groups
- `groups`: Detected groups. This is a list of special class, `Group`.

`groups` are a list of `group`s. Each element in the list, `group`, contains the IDs of member nodes with their roles. 

The donors of  group can be obtained by
```python
group.donors # {node_id: donor_score}
```
- `group.donors` is a dict object, with keys and values corresponding to the node ID and the donor score.

Similarly, the recipients of a group together with their recipient scores are given by
```python
group.recipient # {node_id: recipient_score}
```

## Visualization

```
ax = plt.gca()
dc = cidre.DrawCartel()
dc.draw(group, ax = ax)
```

The labels beside the nodes are the ID of the nodes, or equivalently row ids of the adjacency matrix `A`.

To put the node labels, make a dictionary from the ID to label, like `node_labels = {0:"name", 1:"name 2"}`, and pass it by `node_labels = node_labels`.

