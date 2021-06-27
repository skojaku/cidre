# Python package for the CItation-Donor-REcipient (CIDRE) algorithm.

Please cite:
```latex
Sadamori Kojaku, Giacomo Livan, Naoki Masuda. Detecting citation cartels in journal networks. arXiv:2009.09097 (2020)
```

## Install

```
pip install cidre
```

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

Let us

The donors of  group can be obtained by
```python
groups[0].donors # {node_id: donor_score}
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

## Example script
- [Toy network with communities](examples/example.ipynb)
- [Citation network of journals in 2013](examples/example2.ipynb)
