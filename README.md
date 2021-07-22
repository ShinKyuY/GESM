# Graphs, Entities, and Step Mixture for Enriching Graph Representation

This is a TensorFlow implementation of GESM for the task of classification of nodes in a graph, as described in our paper:

Kyuyong Shin, Wonyoung Shin, Jung-Woo Ha, Sunyoung Kwon, [Graphs, Entities, and Step Mixture](https://arxiv.org/abs/2005.08485)


## Introduction

Existing approaches for graph neural networks typically suffer from the oversmoothing issue that results in indistinguishable node representation, as iterative and simultaneous neighborhood aggregation deepens. Also, most methods focus on transductive scenarios that are limited to fixed graphs, they do not generalize properly to unseen graphs. To address these issues, we propose a new graph neural network that considers both edge-based neighborhood relationships and node-based entity features with multiple steps, i.e. Graphs, Entities, and Step Mixture for Enriching Graph Representation (GESM).
