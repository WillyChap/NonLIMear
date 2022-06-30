# Our modified & adated GCN implementation builds upon the one from below.

-- This is adapated from cachay et al 2020 - Graphino Paper - Cited below. We retain their functionality in our toolbox

Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](../pygcn/figure.png)

Note: There are subtle differences between the TensorFlow implementation in https://github.com/tkipf/gcn and this PyTorch re-implementation. This re-implementation serves as a proof of concept and is not intended for reproduction of the results reported in [1].

This implementation makes use of the Cora dataset from [2].

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```

@article{cachay2021world,
          title={The World as a Graph: Improving El Ni\~no Forecasts with Graph Neural Networks},
          author={Salva Rühling Cachay and Emma Erickson and Arthur Fender C. Bucker and Ernest Pokropek and Willa Potosnak and Suyash Bire and Salomey Osei and Björn Lütjens},
          year={2021},
          eprint={2104.05089},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }