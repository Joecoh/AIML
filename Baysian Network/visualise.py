# visualize.py
import pickle
import matplotlib.pyplot as plt
import networkx as nx

# Load the Bayesian Network model
with open(r"d:\Codes\AIML\Baysian Network\bayesian_model.pkl", "rb") as f:
    model = pickle.load(f)

# Convert Bayesian Network to a directed graph
graph = nx.DiGraph()
graph.add_edges_from(model.edges())

# Draw the graph
plt.figure(figsize=(5, 5))
nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=12)

# Save and show the figure
plt.savefig('model.png')
plt.show()
