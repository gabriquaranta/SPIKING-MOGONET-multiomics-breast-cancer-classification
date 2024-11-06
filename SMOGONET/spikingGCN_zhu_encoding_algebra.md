### Explanation of the Formulation

The key idea behind SpikingGCN is to incorporate topological information of the graph into the node representations. This is achieved by normalizing the weights using the adjacency relationship, allowing nodes to selectively aggregate attributes from their neighbors.

#### Node Representation Update

The update rule for the node representation $\mathbf{h}_i$ is given by:

$$\mathbf{h}_i\leftarrow\frac{1}{d_i + 1}\mathbf{x}_i +\sum_{j=1}^{N}\frac{a_{ij}}{\sqrt{(d_i + 1)(d_j + 1)}}\mathbf{x}_j$$

where:

- $\mathbf{h}_i$ is the new representation of node $v_i$.
- $\mathbf{x}_i$ is the original attribute (feature) of node $v_i$.
- $d_i$ is the degree of node $v_i$.
- $a_{ij}$ is the element of the adjacency matrix $A$, indicating whether there is an edge between node $ i $ and node $j$.
- $N$ is the total number of nodes in the graph.

This update rule means that each node's new representation is a weighted combination of its own attribute and the attributes of its neighbors. The weights are normalized by the degrees of the nodes to ensure that the representation is balanced across the graph.

#### Matrix Formulation

The attribute transformation over the entire graph can be expressed in a matrix form as:

$$S =\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$$

where:

- $\tilde{A}= A + I$ is the adjacency matrix with added self-connections (identity matrix $I$).
- $\tilde{D}$ is the degree matrix of $\tilde{A}$, where $\tilde{D}_{ii}= d_i + 1$.

The new node representations equivalent to after $K$ graph convolution layers are given by:

$$H = S^K X$$

where:

- $H$ is the matrix of new node representations.
- $X$ is the matrix of original node attributes.
- $S$ is the normalized adjacency matrix.

### Detailed Breakdown

1. **Self-Connection Addition**:
   - By adding the identity matrix $I$ to the adjacency matrix $A$, we include self-loops. This ensures that each node considers its own attribute in the aggregation process.
2. **Normalization**:

   - The degree matrix $\tilde{D}$ is constructed from $\tilde{A}$, where $\tilde{D}_{ii}= d_i + 1$. This includes the self-loop in the degree calculation.
   - The normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ ensures that the attributes are scaled properly, preventing large degree nodes from dominating the representation.

3. **Propagation**:
   - The propagation mechanism involves multiplying the normalized adjacency matrix $S$ with the attribute matrix $X$. This operation aggregates the attributes from neighboring nodes.
   - Repeating this process $K$ times (for $K$ layers) allows the network to capture information from nodes that are up to $K$ hops away in the graph.
