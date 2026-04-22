# DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
Large-scale atomistic simulations are essential to bridge computational materials and chemistry to realistic materials and drug discovery applications. In the past few years, rapid developments of machine learning interatomic potentials (MLIPs) have offered a solution to scale up quantum mechanical calculations. Parallelizing these interatomic potentials across multiple devices poses a challenging, but promising approach to further extending simulation scales to real-world applications. In this work, we present \textbf{DistMLIP}, an efficient distributed inference platform for MLIPs based on zero-redundancy, graph-level parallelization. In contrast to conventional spatial partitioning parallelization, DistMLIP enables efficient MLIP parallelization through graph partitioning, allowing multi-device inference on flexible MLIP model architectures like multi-layer graph neural networks. DistMLIP presents an easy-to-use, flexible, plug-in interface that enables distributed inference of pre-existing MLIPs. We demonstrate DistMLIP on four widely used and state-of-the-art MLIPs: CHGNet, MACE, TensorNet, and eSEN. We show that DistMLIP can simulate atomic systems 3.4x larger and up to 8x faster compared to previous multi-GPU methods. We show that existing foundation potentials can perform near-million-atom calculations at the scale of a few seconds on 8 GPUs with DistMLIP.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a novel distributed inference platform, DistMLIP, for machine learning force fields.  DistMLIP enables efficient MLIP parallelization through graph partitioning, overcoming the requirement for LAMMPS. The results show that this platform 1)  significantly outperforms previous SOTA regarding the inference efficiency, 2) supports large-molecule inference, and 3) supports multiple network architectures.

### Strengths
1. DistMLIP significantly improves the inference speed for several models.
2. DistMLIP enables large molecular inference.
3. DistMLIP is model-agnostic.
4. DistMLIP achieves above features via full-atom level graph partition instead of spatial partition, omitting ghost atoms utilized by SevenNet, which are methodologically novel.

### Weaknesses
1. This platform does not support training.
2. This platform does not support multi-node inference.
3. The vertical wall is ad hoc. Although it performs well for the selected molecular system, it could fail for anisotropic molecules.

### Questions
Weakness 3 seems like an inherent limitation of DistMLIP by nature, but do you have any plan to address it, or to show that the method could also perform well for anisotropic molecules?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
DistMLIP is a distributed inference platform for ML interatomic potentials (MLIPs) that parallelizes graph-based models across GPUs with zero redundant compute. It partitions the atom graph (and the three-body “bond” line graph) and exchanges border features every GNN layer, so existing models (MACE, CHGNet, TensorNet, eSEN) can run multi-GPU without architectural changes. It can use up to 3.4× larger systems and is up to 8× faster than prior multi-GPU approaches.

### Strengths
1. Works with popular MLIPs (MACE, TensorNet, CHGNet, eSEN) with minimal adaptation, so we don’t need model-specific rewrites.
2. The “vertical” partition rule is reported up to 8x faster than standard graph partitioners (e.g., METIS/RCMK). And against SevenNet’s distributed inference, DistMLIP has up to 10x higher max capacity and is 4x faster.

### Weaknesses
Majors:
1. The authors say the design keeps backprop intermediates in their contribution claims, but they only benchmark inference; there’s no distributed-training result or accuracy/stability study over long MD runs.
2. All inference timing is on one cluster of 8x A100-80GB; there’s no multi-node or NVLink study to justify capability of large scale simulation.

Minors:
1. Line 39: "CHARM" -> "CHARMM"
2. Line 44: "coupled clustering" -> "coupled cluster"
3. Line 132: Citation format
4. Line 201: "G_n" -> "G_p"
5. Line 271: "Pytorch, Jax" -> "PyTorch, JAX"
6. Line 346: "partitoining" -> "partitioning"

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DistMLIP, a platform designed for efficient distributed inference of Machine Learning Interatomic Potentials (MLIPs), particularly targeting large-scale atomistic simulations (up to millions of atoms). The authors argue that while MLIPs offer quantum accuracy at lower cost, scaling them further requires multi-device parallelization . Existing methods, primarily based on spatial partitioning (like in LAMMPS), suffer from computational redundancy (ghost atoms) and are often ill-suited for modern, long-range GNN-based MLIPs . DistMLIP proposes a graph-level parallelization strategy based on graph partitioning, aiming for zero redundancy. It partitions the atom graph (and optionally higher-order graphs like bond graphs) across multiple GPUs and manages the communication of necessary node/edge features between partitions at each GNN layer . A key feature is its design as a flexible, easy-to-use, standalone platform that can wrap existing MLIPs without requiring model modification or reliance on specific simulation packages . The effectiveness is demonstrated on four diverse MLIPs (CHGNet, MACE, TensorNet, eSEN), showing significant improvements in maximum simulatable system size (up to 3.4x larger) and speed (up to 8x faster) compared to previous methods, enabling near-million-atom calculations in seconds on 8 GPUs.

### Strengths
- Scaling MLIP simulations to biologically and materially relevant sizes (millions of atoms) is a major challenge. DistMLIP provides a much-needed solution specifically tailored for efficient distributed inference of modern GNN-based MLIPs.
- DistMLIP is designed as a model-agnostic, plug-in platform . This allows researchers to apply it to their existing, pre-trained MLIPs with minimal adaptation (as demonstrated with four different models), significantly lowering the barrier to large-scale simulations. Its independence from specific simulation software like LAMMPS increases flexibility.
- The reported results are substantial: linear scaling of capacity, significant speedups in strong scaling tests (up to 8x faster), and the ability to perform near-million atom calculations in seconds on modest hardware (8 GPUs). Comparisons vs LAMMPS MACE and SevenNet further highlight the advantages.

### Weaknesses
- Graph partitioning inherently requires communication between GPUs after each message-passing layer to exchange border node information. The paper acknowledges scaling isn't always ideal (Fig 2b, 2c) partly due to overheads, but a more detailed analysis of communication cost vs. computation cost, and how it scales with the number of GPUs, graph density, and partition quality, would be valuable.
- The paper appears to be lacking comparisons against several relevant baselines. For instance, a critical benchmark is missing: how does the speed (e.g., throughput or latency) of the proposed system compare to other established distributed systems, such as Allegro?

### Questions
- Can the authors provide a breakdown of inference time into computation, communication, and graph construction/partitioning for different scenarios (varying GPU counts, system sizes)?
- What are the main challenges and potential strategies for extending DistMLIP to multi-node environments?
- How much effort is typically required to integrate a new MLIP model into the DistMLIP framework? Does it require modifications to the model's forward pass implementation? (Code 2 gives hints, but more context would be useful).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces DistMLIP, a distributed inference platform designed to scale Machine Learning Interatomic Potentials (MLIPs) to large-scale atomistic simulations. The core problem it addresses is the computational bottleneck of running modern, high-accuracy MLIPs—many of which are based on Graph Neural Networks (GNNs) with long-range interactions—on systems with millions of atoms.

The key contribution is a "zero-redundancy, graph-level parallelization" strategy. This method contrasts with conventional spatial partitioning (e.g., in LAMMPS), which suffers from high computational redundancy due to the need to calculate "ghost atoms" at partition boundaries. DistMLIP partitions the atomic graph itself and distributes subgraphs to different GPUs, enabling efficient parallel inference. The platform is presented as a flexible, "plug-in" library that does not depend on third-party simulation packages like LAMMPS.

The authors demonstrate DistMLIP's effectiveness by benchmarking four popular MLIPs: CHGNet, MACE, TensorNet, and eSEN. The results show that DistMLIP can simulate systems 3.4x larger and achieve up to 8x faster performance compared to previous multi-GPU methods , enabling near-million-atom simulations on a single 8-GPU node.

### Strengths
### High-Impact Problem: 

The paper tackles a critical and timely bottleneck in computational science. Scaling MLIPs to the meso-scale (millions of atoms) is essential for bridging quantum-accurate simulations with real-world applications in materials science, chemistry, and biology.

### Sound and Novel Method: 

The graph-level parallelization approach is fundamentally better suited for GNN-based MLIPs than traditional spatial partitioning. The paper clearly articulates the "zero-redundancy" advantage , which correctly avoids the cubic scaling of redundant computations (ghost nodes) that spatial partitioning faces as the MLIP interaction range increases. The method's native support for both atom graphs and higher-order bond graphs (used in models like CHGNet) is a significant advantage.

- Comprehensive and Rigorous Empirical Validation: This is the paper's strongest aspect.

- Diverse Models: The method is validated on four distinct and widely-used MLIPs, demonstrating its generality.

- Strong Baselines: The authors provide direct comparisons against two crucial baselines: (1) The industry-standard spatial partitioning (LAMMPS-MACE) and (2) Another graph-parallel method (SevenNet). DistMLIP shows superior performance in maximum capacity and speed against both.

- Excellent Scaling Analysis: The paper provides clear strong and weak scaling plots (Fig. 2) , as well as detailed analyses of how performance scales with model parameters and, most importantly, interaction range (Fig. 3). The linear scaling with interaction range (vs. cubic for spatial partitioning) is a key result.

### Pragmatic Design Insight:

A standout finding is the justification for using a simple "vertical wall" partitioning scheme. The authors correctly identify that for MD, the partitioning latency (which must be paid at every time step) is a critical bottleneck . Their simple heuristic is shown to be up to 8x faster than more complex graph partitioning algorithms like METIS or RCMK (Table 4), demonstrating a deep, practical understanding of the problem domain.

### Practicality and Usability: 

By designing DistMLIP as a standalone, "plug-and-play" Python-based library , the authors have significantly lowered the barrier to adoption for the broad community of researchers who use these models but are not experts in distributed computing.

### Weaknesses
### Single-Node Limitation:

 The paper states the current implementation only supports "single-node multi-GPU inference". This is a significant limitation for scaling to truly massive systems (tens of millions+ atoms), which would require a multi-node, multi-GPU setup. The paper would be stronger if it discussed the roadmap and key challenges (e.g., managing communication overhead of border node features across a network interconnect) for a multi-node implementation.

### Clarification of "8x Faster" Claim: 

The abstract and introduction claim "up to 8x faster". However, the direct end-to-end inference comparison with SevenNet shows a ~4x speedup , and the LAMMPS-MACE comparison shows similar (not 8x faster) speeds, albeit with a non-compiled model. The 8x speedup figure appears to be sourced from the partitioning algorithm comparison in Table 4. The authors should clarify this in the main text to avoid overstating the end-to-end simulation speedup.

### Scaling of High-Order Graphs: 

The paper honestly reports "suboptimal" weak scaling for CHGNet, attributing it to the three-body graph construction cost. This is an important detail, as it suggests that the performance benefits of DistMLIP may be partially bottlenecked by models with complex, high-order interactions. A brief discussion of whether this construction is (or can be) parallelized within DistMLIP would be beneficial.

### eSEN Performance: 

The eSEN model consistently performs poorly, with high memory usage and frequent OOM errors. While this is likely due to the model's architecture rather than DistMLIP, its poor performance slightly detracts from the platform's "general and versatile" claim.

### Questions
## 1. Multi-Node Scalability: 

The current work is an excellent demonstration of single-node, multi-GPU scaling. Could you elaborate on the primary challenges for extending DistMLIP to a multi-node environment? Specifically, how do you envision managing the atom_transfer step across a network interconnect, and what do you anticipate will be the new performance bottleneck (e.g., network latency vs. bandwidth)?

## 2. High-Order Graph Construction Bottleneck: 

The suboptimal weak scaling of CHGNet is due to the three-body graph construction. Is this construction step (described in Algorithm 2) fully parallelized within DistMLIP, or is it a separate, serial (or partially parallel) step that acts as a bottleneck before the GNN forward pass? Does this imply a fundamental limitation for DistMLIP's performance on future models that might incorporate even higher-order (e.g., four-body) interactions?

### Soundness
3

### Presentation
3

### Contribution
4
