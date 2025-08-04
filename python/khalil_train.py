"""
Example of pipeline of the BoundML library

It replicates the experiment of "Exact Combinatorial Optimization  with Graph Convolutional Neural Networks"
(http://arxiv.org/abs/1906.01629)
"""

from boundml.model import train

model = train(sample_folder="samples/cfl", learning_rate=0.001, n_epochs=64, output="gnn_agents/agent_cfl.pkl")

