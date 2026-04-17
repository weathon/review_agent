# Graph-Based Operator Learning from Limited Data on Irregular Domains

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 2

## Abstract
Operator learning seeks to approximate mappings from input functions to output solutions, particularly in the context of partial differential equations (PDEs). While recent advances such as DeepONet and Fourier Neural Operator (FNO) have demonstrated strong performance, they often rely on regular grid discretizations, limiting their applicability to complex or irregular domains. In this work, we propose a Graph-based Operator Learning with Attention (GOLA) framework that addresses this limitation by constructing graphs from irregularly sampled spatial points and leveraging attention-enhanced Graph Neural Netwoks (GNNs) to model spatial dependencies with global information. To improve the expressive capacity, we introduce a Fourier-based encoder that projects input functions into a frequency space using learnable complex coefficients, allowing for flexible embeddings even with sparse or nonuniform samples. We evaluated our approach across a range of 2D PDEs, including Darcy Flow, Advection, Eikonal, and Nonlinear Diffusion, under varying sampling densities. Our method consistently outperforms baselines, particularly in data-scarce regimes, demonstrating strong generalization and efficiency on irregular domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GOLA (Graph-based Operator Learning with Attention) — a framework for operator learning on irregular spatial domains. The method combines (1) a learnable Fourier encoder for spectral embedding of irregular samples and (2) attention-enhanced graph neural networks (GNNs) for message passing. The authors claim GOLA generalizes across PDE types (Darcy Flow, Advection, Eikonal, Nonlinear Diffusion) and sampling densities, outperforming DeepONet, AFNO, and Graph Kernel Network (GKN) in low-data regimes.

### Strengths
1. The paper targets an important challenge: operator learning on irregular and sparse domains.
2. The writing is overall structured and readable.

### Weaknesses
1. Limited Novelty and Overlap with Existing Work
- Both Fourier encoding and attention mechanisms have been extensively explored in the context of graph neural networks, including Fourier Feature Networks, Graph Attention Networks (GATs), and Graph Transformer networks (GTN).
- The proposed combination in GOLA therefore appears incremental, as similar designs integrating spectral embeddings with attention-based message passing already exist in prior operator learning and geometric deep learning literature.
- The paper fails to clearly articulate what novel insight, mechanism, or theoretical contribution differentiates GOLA from existing graph-based operator learning frameworks.

2. Weak Baselines and Benchmarks
- The baseline comparisons are limited to older and weaker models (DeepONet, AFNO, GKN), which no longer represent the state of the art. The authors should include more recent and competitive baselines that address irregular or non-Euclidean domains, such as Transolver, Position-induced Transformer (PiT), Geo-FNO, PINO, Geo-FNO, etc.
- The chosen benchmarks are small-scale, synthetic PDEs (Darcy, Advection, Eikonal, Diffusion) simulated in-house. These toy problems do not adequately demonstrate the model’s generalization capability. Evaluation on standard community datasets—such as Airfoil, ShapeNet-Car, or AirFRANS—would significantly strengthen the empirical validation.

3. Lack of Depth in Theoretical and Conceptual Analysis
- The “theoretical analysis” section is superficial, restating standard neural operator approximation theorems without offering any model-specific insight or proof.
- The paper provides no theoretical reasoning or ablation-based evidence explaining why combining Fourier encoding with attention-enhanced GNNs should lead to improved expressivity, generalization, or sample efficiency.

### Questions
1. What is the key novelty beyond existing graph-based operator networks with attention?
2. What if the authors remove the Fourier encoder or attention modules? Are there any ablation studies?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes GOLA (Graph-based Operator Learning with Attention), an operator-learning framework for PDEs that works on irregular domains and sparse, nonuniform samples. GOLA first embeds inputs with a learnable Fourier encoder that projects function values at arbitrary coordinates into a spectral basis with complex-valued modes, then performs attention-enhanced message passing on a proximity graph built from spatial points to capture both local and global dependencies. Experiments on four 2D PDE families—Darcy Flow, Advection, Eikonal, and Nonlinear Diffusion—show consistent gains over baselines (DeepONet, AFNO, GKN).

### Strengths
1. **Learnable Fourier encoder on spatial graphs.**
   Turning inputs into a complex Fourier basis lets the model capture long-range/global interactions with few coefficients, while graph message passing refines local structure. Because of the learned frequencies, the model can generalize across resolutions (the spectral representation is not tied to a fixed grid). This also reduces aliasing artifacts compared with naïve coordinate MLPs and gives a compact, physics-plausible feature space for operator learning.

2. **Clear and sufficiently detailed method presentation.**
   The paper lays out the pipeline cleanly—Fourier encoding → graph construction → attention-augmented message passing → decoding—with consistent notation and design choices explained (edge features, attention rationale, training objective). The ablations and component descriptions make it straightforward to reproduce and to understand which parts drive gains.

3. **Helpful visualizations.**
   The figures (architecture block diagram, graph construction sketches, and qualitative field reconstructions) make the workflow and design choices easier to follow and provide intuitive evidence for how attention and spectral features influence predictions. These plots also highlight resolution/generalization behavior and error patterns, aiding interpretability.

### Weaknesses
1. **Outdated or incomplete comparisons on irregular-domain PDE learning.**
   The related work and experimental baselines underrepresent recent approaches across **GNN** [1], **implicit neural representations (INR/SIREN/coordinate networks)** [2], and **Transformer-style neural operators** [3] tailored to unstructured meshes or scattered points. Without head-to-head evaluations against stronger and more recent models (e.g., graph/mesh operators with positional encodings, attention-based operators on point clouds), it’s hard to substantiate GOLA’s claimed advantage. This limits the paper’s positioning and makes the empirical novelty less compelling.

2. **“Irregular domains” are synthetically derived from uniform grids, weakening the motivation.**
   The paper’s “irregular sampling” is obtained by subsampling a uniform lattice, which **does not reflect** the practical challenges that motivate irregular discretizations: complex or curved boundaries, anisotropic resolution to capture fine features, mesh adaptivity, or true unstructured meshes common in CFD and geometry-rich PDEs (e.g., airfoils, cylinder flows, ShapeNet-like CAD geometries). As a result, the experiments do not test boundary handling, mesh heterogeneity, or topology changes—key factors for demonstrating real-world utility. The focus on **only 2D** further narrows external validity; many operator-learning applications (fluid/solid mechanics, climate/ocean) require robust performance on 3D unstructured meshes.


[1] Li, Zhihao, et al. "Harnessing scale and physics: A multi-graph neural operator framework for pdes on arbitrary geometries." Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2025.

[2] Wang, Honghui, Shiji Song, and Gao Huang. "GridMix: Exploring Spatial Modulation for Neural Fields in PDE Modeling." The Thirteenth International Conference on Learning Representations. 2025.

[3] Wu, Haixu, et al. "Transolver: A Fast Transformer Solver for PDEs on General Geometries." International Conference on Machine Learning. PMLR, 2024.

### Questions
1. **Stronger, more representative experiments.**
   Could you add evaluations on *truly* irregular domains (unstructured/anisotropic meshes and complex boundaries), e.g., airfoil/cylinder flows or CAD-like geometries, and compare against more recent GNN/INR/Transformer-based operators designed for such settings? This would directly address external validity and positioning (see Weaknesses).

2. **Fourier encoding on irregular meshes.**
   Please clarify why the proposed Fourier encoder is applicable on nonuniform point sets:

   * Is the encoding purely a function of coordinates (i.e., independent of grid regularity), or does its validity rely on the fact that your samples come from an underlying uniform lattice?
   * How does aliasing/spectral leakage behave on truly unstructured meshes where Nyquist-style guarantees don’t hold?
   * Can the approach extend to 3D unstructured meshes (e.g., tetrahedral/hexahedral) and curved boundaries, and what modifications (basis choice, graph construction, positional encodings) would be required?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposed a novel architecture of neural operators based on graph neural networks and attention, GOLA, for data-scarce scenarios. On 4 2D benchmarks, GOLA shows superior performance compared with 3 baselines.

### Strengths
The paper is presented in high clarity.

### Weaknesses
Regarding the experiments:
1. The baseline models are not SOTA, where the latest is AFNO, a work in 2021. The baseline should include some SOTA neural operators, e.g., Transolver [1].
2. The author claimed generalization across sample densities and resolutions. However, the experiments are conducted only for GOLA. How about the baseline models? Is GOLA better at generalization across sample densities and resolutions, or worse?
3. On data efficiency, again, comparison should be made among baseline models. At what amount of data would baseline models catch up with GOLA?
4. Are models compared in the setting of similar number of parameters?

[1] Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024). Transolver: A fast transformer solver for pdes on general geometries. arXiv preprint arXiv:2402.02366.

### Questions
N.A.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a new architecture for operator learning over irregular point clouds.  It encodes sampled inputs along trainable Fourier features, embeds these features on a spatial graph according to the sampling scheme, and uses a message passing algorithm and an attention mechanism to produce an output over these spatial nodes.  A proof of universality of this architecture is presented and experiments on Darcy flow, nonlinear diffusion, Eikonal, and advection PDE operators are conducted to compare against DeepONet, AFNO, and GKN architectures.  Compared to these three other architectures, the proposed method shows improved results.

### Strengths
Exploring additional architectures for neural operators, especially on irregular domains or with irregularly sampled discretizations, is an important area of research to further improve our ability to make surrogate models for PDE solution operators.

### Weaknesses
The paper claims that the method can be used for learning operators on irregular domains, however all experimental results are for rectangular domains.  The sampling within these domains is not necessarily on a grid, but throughout the paper there is no evidence support any claims for irregular domains themselves.  

Not addressing irregular domains properly is also the cause for some incorrect theoretical arguments.  In particular, the paper references the existence of the complex exponentials as a basis for $L^2(\Omega)$, with $\Omega$ a compact domain.  This is false:

Iosevich, Alex. "The Fuglede spectral conjecture holds for convex planar domains." Mathematical Research Letters 10.5 (2003): 559-570.

The proof of universality of the architecture is incomplete.  The proof proceeds by showing first that input functions can be represented arbitrarily well by a finite number of complex exponentials (not true for general compact $\Omega$).  It then uses the universarlity of GNNs to claim arbitrarily close approximation (implicitly in L2 but not specified) of the mapping on the sampled points.  The GNN decoder $D_\theta$ is invoked without being defined and its error is not accounted for.  

The only motivation provided for the architecture choices is that a graph is used to handle non-uniformly spaced sampling points.  All other components are presented largely without motivation.

Previous work has investigated methods for operator learning on irregular point clouds which has not been discussed or compared against in this paper, such as 

Lingsch, Levi E., et al. "Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains." International Conference on Machine Learning. PMLR, 2024..

Li, Zongyi, et al. "Fourier neural operator with learned deformations for pdes on general geometries." Journal of Machine Learning Research 24.388 (2023): 1-26.

Liu, Ning, Siavash Jafarzadeh, and Yue Yu. "Domain agnostic fourier neural operators." Advances in Neural Information Processing Systems 36 (2023): 47438-47450.

Due to lack of relative baselines for other neural operator methods on irregular domains or sampling schemes the benefits of this particular construction are not clear.

### Questions
1.  How does this method compare to other methods for operator learning with irregularly sampled points?

2.  What is the motivation for the message aggregation as presented in (8)?

3.  Is this method competitive with other operator learning methods specifically designed for irregular sampling on the presented benchmarks?

4.  Is this architecture universal if we cannot appeal to the density of spans of complex exponentials in $L^2(\Omega)$?

### Soundness
1

### Presentation
1

### Contribution
2
