# From atom to space: A region-based readout function for spatial properties of materials

- Decision: Accept (Poster)
- Scores: 4, 8, 2, 6

## Abstract
The message passing–readout framework has become the de facto standard of graph neural networks (GNNs) for material property prediction. However, most existing readout functions are built on an atom-decomposable inductive bias, i.e. the material-level property or feature can be reasonably assigned to contributions of individual atoms. This is a strong bias and may not hold for all properties, limiting the application scenarios (e.g. gas adsorption or separation of Metal Organic Frameworks, MOFs). In this work, we propose a region-based decomposition perspective, reformulating material properties as integrals over space and pooling contributions from spatial regions rather than atoms. Specifically, we propose a novel readout function named SpatialRead. SpatialRead introduces additional spatial nodes to represent a voxelized space, transforming the atomic isomorphic graph into a heterogeneous atom–space graph with unidirectional message flow from atoms to spatial nodes. To combine the two types of inductive bias, multimodal methods can be used to fuse the features of atoms the spatial nodes. Such a region-based readout function is especially suited for spatial properties such as gas adsorption capacity, separation ratio. Extensive experiments demonstrate that a simple PaiNN–Transformer-based SpatialRead trained from scratch outperforms state-of-the-art pre-trained foundation models on these special tasks. Our results highlight the importance of designing physically grounded readout functions tailored to the target property. The code and dataset can be found in github https://github.com/nankusa/SpatialRead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Introduces a new readout function for materials property prediction tasks using GNNs/MPNNs. The idea of the paper is to introduce a grid of additional spatial nodes and readout functions that only use learned node features of these spatial nodes or readout functions that take into account atom and spatial nodes. Experiments are performed on different datasets, and very good performance is demonstrated across different tasks and datasets, compared to fine-tuned models and models trained from scratch.

### Strengths
The experiments show very strong performance of the newly introduced readout functions based on spatial nodes.

The analysis in Figure 2 and the additional experiments in Figure 3 are very interesting and insightful.

### Weaknesses
The paper claims that current read-out functions introduce an inductive bias that is not suitable for certain tasks and materials, e.g. the prediction of gas adsorption capacity in metal organic frameworks. This claim is not completely unreasonable, but it is also not supported by any quantifiable evidence. An equivariant GNN can learn local geometric atom environments (also around a pore), up to a cutoff radius which is determined by cutoff-hyperparameters and model depth. Thus, in principle, there is no direct reason to claim that it is impossible to learn pore volumes as a function of mean- or sum-pooled atom embeddings. Every atom can learn from its local geometric environment if it is adjacent to a pore or not, and what pore volume is "attributed" to this atom, so sum pooling can yield the pore volume.

The baseline models CGCNN, GemNet, and PaiNN are more than 5 years old. The performance of newer GNN architectures is not compared. What is the performance of the models that are currently leading the MatBench benchmark?
The pretrained models are newer, but the authors do not compare the performance of the fine-tuned models to the performance if those models were trained from scratch.

Generally, a more differentiated comparison to other, more recent readout functions is missing.

### Questions
In Section 2, you discuss readout functions used in the GNN/MPNN literature. All references are older than 2019. Please report about more recent developments of readout functions, e.g. the ones discussed in [1] or even completely different approaches such as [2].
[1] Liu, C., Zhan, Y., Wu, J., Li, C., Du, B., Hu, W., Liu, T. and Tao, D., 2022. Graph pooling for graph neural networks: Progress, challenges, and opportunities. arXiv preprint arXiv:2204.07321.
[2] Wu, Z., Jain, P., Wright, M., Mirhoseini, A., Gonzalez, J.E. and Stoica, I., 2021. Representing long-range context for graph neural networks with global attention. Advances in neural information processing systems, 34, pp.13266-13279.

If Theorem 3.1 is true and Eq. 6 and 8 have the same expressivity, what is the advantage of Eq. 8 over Eq. 6?

What is the difference between extensive quantities of materials and your definition of spatial properties according to definition 3.1? Are all extensive quantities also spatial quantities?

What are the exact definitions of N(s_j) and N(v_j) in Eq. 9 and 10? How many atomic node connections does every spatial node have, and how many spatial node connections? Is this based on cutoff radii as indicated by the phrase "nearby atoms"? How are "adjacent spatial nodes" defined? The 6 nearest neighbor voxels? Or 26? Or also based on a cutoff radius? Figure 1 does not indicate any s_i to s_j edges. How are the positions of the spatial nodes selected? Figure 1 does not indicate a regular lattice. How is the spatial voxel grid defined for more complex space groups than the one shown in Figure 1? Are distances defined in relative coordinates (within the unit cell) or in absolute coordinates (in real space)? 

Multi-Modal Transformer: "we impose an explicit ordering" - How is this done, and how does it compare for different types of symmetry groups with different unit cell shapes?

Equation 11: Why does every spatial node only contribute a scalar contribution to the final property? Why do you not use p = MLP_final(Sum(MLP(h))), where MLP(h) is outputting a vector rather than a scalar?

Minor comments: Mistake in Theorem 3.1: "then is must can be".

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper identifies a key inductive bias limitation in graph neural networks (GNNs) for material property prediction — namely, that existing message passing–readout frameworks assume atom-decomposable properties, which is not valid for spatially defined material properties such as adsorption, separation, or accessible surface area.

To overcome this, the authors propose SpatialRead, a novel readout function that reformulates the graph-level representation as an integral over space rather than a sum over atoms. The method introduces spatial nodes that voxelize the material domain, forming a heterogeneous atom–space graph with one-way message flow from atoms to spatial nodes. A Transformer-based multimodal readout fuses atomic and spatial representations.

Theoretically, the authors prove that the space-integral and node-summation formulations are equivalent in expressivity when the receptive field is finite. Empirically, SpatialRead achieves state-of-the-art results on 44,157 porous materials and 27 spatial-property prediction tasks, outperforming even large-scale pre-trained foundation models such as JMP (120M samples). The approach also maintains strong performance on conventional (non-spatial) material properties.

### Strengths
- The first formal unification of node-decomposable and region-decomposable graph readouts.
- The directionality (atoms → space) aligns with actual field theory and adsorption physics.
- Equivalence theorem ensures expressivity preservation.
- 44k+ samples and 27 tasks, across multiple material systems.
- ~2.9 MB model surpasses 38 MB foundation models.
- Bridges discrete atomic graphs and continuous spatial modeling paradigms.

### Weaknesses
- The voxelization scheme (e.g., 8×8×8) is fixed; adaptive resolution or hierarchical partitioning could further improve scalability and accuracy?
- Although SpatialRead outperforms pretrained models from scratch, it would be interesting to see whether adding pretraining further enhances its capability.
-  The paper focuses on porous materials and spatial properties; it remains unclear whether the proposed readout function generalizes to conventional datasets like Materials Project or JARVIS, especially since the baseline models (e.g., CGCNN, ALIGNN, Matformer) were originally designed and benchmarked on such conventional crystal datasets.

### Questions
- Does the voxelization scheme preserve lattice periodicity? If not, predictions might vary under cell replication.
- While efficient for moderate datasets, its scalability to extremely large systems (>10⁴ atoms) isn’t thoroughly discussed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel spatial readout mechanism for Graph Neural Networks, with applications to materials science tasks, particularly Metal-Organic Frameworks (MOFs). The authors argue that pooling/readout mechanisms in GNNs still have room for innovation and introduce a method that incorporates additional “spatial” nodes (mostly in void/vacuum) space to improve property predictions.

### Strengths
- Addresses a relevant problem: readout mechanisms in GNNs remain an area where improvements are possible
- Applies methodology to practical materials science applications

### Weaknesses
### 1. Weak Empirical Validation

The experimental evaluation is insufficient to support the claims:

- **Trivial target properties**: The main benchmarks focus on predicting surface area and pore volume for MOFs. From a materials science perspective, these properties are almost directly derivable from the structure itself (via van der Waals sphere calculations), making them poor choices for demonstrating the value of a novel readout mechanism.

- **Selective MatBench reporting**: The paper only reports results on a subset of MatBench properties. This selective reporting raises concerns about the method's general applicability. What happened to the other MatBench tasks?

- **Inconsistent and marginal improvements**: Across benchmarks, the improvements are neither consistent nor substantial enough to justify the added complexity.

### 2. Missing Ablation Studies

The paper lacks essential ablation studies to disentangle the contributions of different components:

- **Effect of additional void nodes**: The method adds nodes in empty/void space, which allows the model to trivially compute the ratio between occupied and unoccupied space—essentially providing a direct signal for pore accessible volume fraction. This could be the primary driver of any improvements, independent of the readout mechanism.

- **Required ablation**: Compare additional void nodes with standard pooling (max/mean) versus the proposed spatial readout. This would demonstrate whether the gains come from the readout innovation or simply from the additional structural information.

### 3. Added Complexity Without Clear Justification

- The method introduces an additional hyperparameter (sampling degree for void space), increasing model complexity and optimization difficulty
- Given the weak empirical gains, it's unclear whether this added complexity is warranted

### 4. Questionable Benchmark Choice

The authors introduce a new benchmark, but the field already has established benchmarks (e.g., mofdscribe for MOFs). The rationale for a new benchmark is not clearly articulated.

### Questions
1. Can you provide complete MatBench results for all properties rather than a selective subset?
2. Can you include an ablation study comparing: (a) base model, (b) base model + void nodes + simple pooling, (c) base model + void nodes + proposed readout?
3. Why is a new benchmark needed when established ones like mofdscribe exist?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present SpatialRead, a new region-based readout function which introduces additional spacial nodes to represent voxelized space that is an important characteristic of some materials. SpatialRead also fuses both atom and spatial nodes passed to Transformer-based readout.

The empirical evaluation of SpatialRead is performed with several tasks including spatial properties of materials such as gas absorption capacity, showing the superiority of SpatialRead against several existing methods.

The paper addresses an important problem in AI and materials science. Their approach adds important features on spacial properties which have not been effectively considered by the existing GNN-based approaches. 
In addition to the evaluation with standard machine learning metrics such as $\mbox{R}^2$ scores and MAE, the authors also attempt to interpret what happens with the spatial nodes with good illustrations regarding absorption of materials (i.e., Figure 2B).
I believe this is also an important contribution which reveals the behavior of SpatialRead, especially if/when SpatialRead is used in practical situations to work on materials discovery . 

One comment is that a better strategy which is commonly used in graph embedding for organic molecules can be used easily to further improve the performance. See Equation 4.2 in the following paper: 

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. "How Powerful Are Graph Neural Networks?", ICLR 2019. 

I wonder why the existing approaches as well as SpatialRead do not incorporate this simple, well-known improvement. 

The other minor comment is that Definition 3.1 should be introduced before Equation (8). 

All in all, this is an interesting paper which includes important contributions to the AI and materials informatics research communities.

### Strengths
Introducing an important readout function that can additionally consider spacial capacities of materials

Performance evaluation of SpacialRead that shows the superiority to other existing methods

Interpretation of SpacialRead regarding the contributions of spacial nodes, which clearly shows strong evidence on why SpacialRead works effectively

### Weaknesses
Comparison/discussion against an approach that enumerates the embedding from iterations 0, 1, .. to T as is done by so-called Graph Isomorphism Network (see equation 4.2 in the paper mentioned above)

### Questions
Considering that it is a simple enhancement known to improve the performance for organic molecules, how would the performance fare if all methods presented in the paper incorporate  equation 4.2 in the paper pointed out above?

### Soundness
3

### Presentation
3

### Contribution
3
