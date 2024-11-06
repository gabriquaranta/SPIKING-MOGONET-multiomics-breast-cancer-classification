# GCN

A Graph Convolutional Network (GCN) is a type of neural network designed to work
with graph-structured data. Unlike traditional neural networks that operate on
grid-like data (e.g., images or sequences), GCNs can directly process data
represented as graphs, which are collections of nodes (vertices) and edges
(connections between nodes).

In the context of MOGONET, GCNs are used for omics-specific learning, where each
type of omics data (e.g., gene expression, methylation) is represented as a
graph. The nodes in the graph correspond to samples (e.g., patients), and edges
between nodes represent similarities or relationships between those samples
based on the omics data.

Here's a step-by-step explanation of how a GCN works in MOGONET:

1. **Graph Construction**: For each type of omics data, a weighted similarity graph is constructed. The nodes in this graph represent the samples, and the edges between nodes are weighted based on the similarity (e.g., cosine similarity) between the corresponding omics feature vectors of the samples. Samples with similar omics profiles will have higher edge weights between their nodes.

2. **Feature Representation**: Each node in the graph is associated with a feature vector representing the corresponding sample's omics data (e.g., gene expression levels, methylation values).

3. **Neighborhood Aggregation**: The key idea behind GCNs is to learn a node's representation by aggregating and transforming the representations of its neighboring nodes in the graph. This is done through a convolutional operation, similar to traditional convolutional neural networks (CNNs) used for image processing, but adapted to work on graph-structured data.

4. **Convolutional Layer**: In a GCN convolutional layer, each node's new representation is computed by aggregating the representations of its neighbors using a weighted sum or other aggregation function. The weights used in this aggregation are learned during the training process, allowing the GCN to capture patterns and relationships in the graph structure.

5. **Activation and Transformation**: After the neighborhood aggregation, the resulting node representations can be passed through an activation function (e.g., ReLU) and optionally transformed using a fully connected layer, similar to traditional neural networks.

6. **Multiple Layers**: GCNs can have multiple convolutional layers, allowing them to capture increasingly complex patterns and relationships in the graph data by recursively aggregating information from larger neighborhoods.

7. **Supervised Learning**: In the case of MOGONET, the GCN is trained in a supervised manner to predict class labels (e.g., disease status, tumor grade) based on the omics data represented as a graph. The final node representations from the GCN are used as input to a classification layer (e.g., softmax) to make predictions.

The key advantage of using GCNs in MOGONET is that they can effectively capture and leverage both the omics feature information (through the node features) and the similarities or relationships between samples (through the graph structure). By aggregating information from neighboring nodes, GCNs can learn representations that encode not only the individual sample's omics data but also its context within the broader sample population.

This ability to integrate both feature-level and structure-level information makes GCNs particularly well-suited for omics data analysis, where understanding the relationships between samples can provide valuable insights into biological processes and disease mechanisms.