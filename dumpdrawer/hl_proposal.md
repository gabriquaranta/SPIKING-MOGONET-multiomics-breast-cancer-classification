## Project Proposal

### Relevant Repositories:

- https://github.com/ZulunZhu/SpikingGCN
- https://github.com/txWang/MOGONET

### Hardware/Software Requirements:

The project should be executable on local machines or, if needed, on Google
Colab. We do not anticipate requiring specialized hardware or cloud services
beyond what is accessible through Google Colab.

### Implementation Plan:

The core idea is to explore the integration of spiking neural networks,
particularly the SpikingGCN model, into the MOGONET architecture for multimodal
graph representation learning. Specifically, we plan to investigate the
following:

1.  **Spiking Encoding Schemes**:

    - Evaluate different spiking encoding methods, such as rate-based and
      latency-based encoding, to represent graph data as spike trains.

2.  **Substitute GCN with SpikingGCN in MOGONET**:

    - "To avoid the performance degradation of attributes-only encoding,
      SpikingGCN utilizes the graph convolution method inspired by GCNs to incorporate
      the topological information. The idea is to use the adjacency relationship to
      normalize the weights, thus nodes can selectively aggregate neighbor attributes.
      The convolution result, i.e., node representations, will serve as input to the
      subsequent spike encoder." (Zhu et al., 2021)
    - Either keep the output of SpikingGCN as spike trains and modify VCDN or map it to class confidence scores and keep VCDN.

3.  **Modify VCDN (maybe?)**
    Either:

    - Modify the VCDN component of MOGONET to directly accept and process the spiking output from SpikingGCN.

    Or:

    - Map the spiking output to class confidence scores while keeping the overall VCDN architecture intact.

4.  **Performance Evaluation**:

    - Compare the performance of the modified MOGONET architecture with the
      original MOGONET and other state-of-the-art models on both encoded datasets
      (e.g., rate-based and latency-based).
    - Analyze the trade-offs between accuracy and computational efficiency/energy
      consumption.

5.  **Comparative Analysis**:
    - Conduct a comparative analysis of the best-performing spiking model against
      the original MOGONET and other relevant models in terms of accuracy,
      computational complexity, and potential energy savings.

### Tentative Timeline:

It should be done by the first call (4 July).
