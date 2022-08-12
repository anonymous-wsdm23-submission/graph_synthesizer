Graph Synthesizer: Rethinking Graph Attention for Heterogeneous Networks
==============================================================
# 1. Overview

Our research focuses on improving attenion mechanims in graph attention models to overcome their limitations in heterogeneous domain settings. The main challenge of attention-based GNN models on a heterogeneous graph includes node-node interaction problem where self-attention is computed between different types of nodes, which leads to unfaithful attention weights and poor representations. While a common approach to solve this problem is to build a hierarchical framework that applies attention at different granularities, we aim to tackle the challenge by modifying the attention mechanism itself. Graph Synthesizer (GraphSyn) is a counter-intuitive approach that leverages simplified synthetic attention to reduce unwanted node-node interactions. We have explored multiple adaptations of GraphSyn and assessed their effectiveness and faithfulness as alternatives to self-attention mechanism on heterogeneous graphs. We have also worked with two extensions of GraphSyn models and shown that our synthetic attention is extensible to conventional approaches that focuses on complex modeling.

# 2. Source Code

For each experiment in our research, we have created separate directories to manage source codes. Following the path will guide you to necessary source codes for any benchmark experiment except KD-GraphSyn.

Our experiments are conducted on datasets used in [Wang et al. Heterogeneous Graph Attention Network, 2019][gtn_paper]. The datasets are provided in [their Github repository][gtn_repo].


[gtn_paper] : https://arxiv.org/abs/1903.07293
[gtn_repo] : https://github.com/seongjunyun/Graph_Transformer_Networks
