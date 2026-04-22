# Lagrangian Meets Diffusion: Feasibility-aware Generative Modeling for Mixed Integer Linear Programming

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 4, 6

## Abstract
End-to-end Predict-and-Search (PaS) methods show promise for Mixed Integer Linear Programming (MILP), but they typically assume variables independence and provide only deterministic single-point predictions, limiting solution diversity and demanding extensive search for high-quality solutions. We propose \textbf{VRG}, a feasibility-aware generative framework that operates in visual space. It transforms MILP solution vectors into image representations, which are in turn processed by a U-Net-based score network with Lagrangian relaxation guidance. The visual encoding enables convolutional kernels to capture interdependencies among variables while Lagrangian relaxation guides sampling toward feasible, near-optimal regions. The guided generator produces diverse, high-quality candidates rather than a single point estimate. The resulting candidates define compact and effective trust-region subproblems for standard MILP solvers. Across various public benchmarks, VRG consistently outperforms PaS baselines in solution quality and, while maintaining competitive optimality with state-of-the-art solvers such as SCIP and Gurobi, achieves markedly lower computational effort (reduced search time and explored nodes). Our source code is available at https://anonymous.4open.science/r/VRG-E09E/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes VRG, a generative framework that integrates diffusion models with a Lagrangian-guided optimization mechanism for solving MILPs. The authors systematically revisit the limitations of existing Predict-and-Search frameworks and introduce a novel representation method for solution vectors. Specifically, they employ a bijective visual transformation that reshapes each solution vector into an image, where each variable corresponds to a pixel, enabling convolutional diffusion networks to capture interdependencies among variables. Furthermore, to improve the quality and feasibility of generated solutions, the framework incorporates objective optimality and constraint violation penalties as regularization terms within the diffusion process.

### Strengths
1. This paper introduces a Lagrangian-guided diffusion process where both the objective optimality and constraint violation are used as guidance terms, effectively steering the generation toward feasible and high-quality solutions.
2. Across four standard MILP benchmarks, VRG consistently outperforms PaS and CoPaS in both solution quality and solver efficiency, while completely avoiding infeasible subproblems.

### Weaknesses
1. To capture interdependencies among variables, the authors reshape the solution vector into an image via a bijective transformation. However, this transformation is essentially a linear index mapping that may distort the inherent structural relationships among variables. Variables that are tightly coupled within the same constraint may end up being placed far apart in the image space, while unrelated variables may become adjacent. In contrast, the bipartite graph representation naturally preserves the structural relations between variables and constraints, making it a more faithful representation for MILP problems.
2. Although the authors emphasize Lagrangian relaxation, its core advantage of providing dual bounds that can accelerate convergence is not fully exploited here. The proposed approach only incorporates the optimality gap and constraint violations as guidance terms in the diffusion process, which captures the conceptual idea but not the computational strength of classical Lagrangian dual methods.
3. The experimental evaluation lacks comprehensive ablation studies to isolate the contribution of each component. Without such analysis, it is difficult to assess the true source of the performance gains and the reliability of the reported results.

### Questions
1. Could you elaborate on the design rationale of the visual–vector transformation and explain, with evidence, why this image-based representation can capture interdependencies among variables better than the bipartite graph representation?
2. During the visual–vector transformation, the choices of $h$ and $w$ affect how closely related variables are placed in the image space. Could you provide a sensitivity analysis showing how different ($h$, $w$) settings impact performance and the preservation of variable dependencies?
3. Since $\gamma_o$ and $\gamma_c$ control the strength of optimality and feasibility guidance in the diffusion process, please include an extensive analysis of these hyperparameters. In particular, report their effect on solution quality and feasibility across scales and benchmarks, and provide practical guidelines for setting them.
4. To show the benefit of the guidance mechanism, please report results for the diffusion model without the guidance terms. In addition, diffusion models often require deeper architectures than GNN-based solvers. Could you include a study of performance as a function of network depth?
5. Could you add some new baselins? such as [1] and [2].

[1] APOLLO-MILP: An Alternating Prediction–Correction Neural Solving Framework for MILP. ICLR 2025.

[2] Effective Generation of Feasible Solutions for Integer Programming via Guided Diffusion. KDD 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes VRG, a novel generative framework for solving Mixed Integer Linear Programming (MILP) problems. The authors identify key limitations in existing "Predict-and-Search" (PaS) methods: (1) they typically assume variable independence, and (2) they provide only a single, deterministic solution prediction, which limits diversity and requires extensive search.

VRG addresses this by re-framing the MILP problem as a visual generation task. The core ideas are:

Visual-Vector Transformation, Generative Modeling, Lagrangian-Guided Diffusion, Multi-Modal Solution Generation.

### Strengths
High Novelty: The central concept of transforming a discrete optimization problem's solution vector into an image and applying generative computer vision techniques (U-Net, guided diffusion) is exceptionally creative and marks a significant cross-disciplinary contribution.

Directly Addresses PaS Limitations: The method cleverly tackles the two main weaknesses of PaS. First, by using a U-Net, it inherently models local dependencies between variables (pixels), moving beyond the variable independence assumption. Second, by using a generative model, it naturally produces a multi-modal distribution of diverse solutions, rather than a single deterministic guess.

Elegant Integration of Theory: The paper does an excellent job of integrating a classic technique from optimization theory (Lagrangian relaxation) into a modern deep learning framework (score-based SDEs). The theoretical justification (Theorems 2, 3, 4) provides a principled foundation for why this guidance mechanism works, linking it to KL divergence minimization and probability concentration.

Strong and Practical Empirical Results: The experiments are very compelling. VRG doesn't just outperform other ML baselines; it demonstrates practical utility by accelerating state-of-the-art commercial solvers (Table 5). Achieving comparable solution quality to Gurobi or SCIP with significantly fewer search nodes (e.g., ~65% reduction in nodes for CA-Large on SCIP) is a very strong result.

### Weaknesses
Scalability of the Visual Representation: The experiments are conducted on problems where the number of variables ($n$) is relatively small (e.g., $\le 1500$), resulting in small images (e.g., $30 \times 50$). How does this approach scale to real-world MILPs with tens of thousands or millions of variables? A 1,000,000-variable problem would require a $1000 \times 1000$ image, making the U-Net/diffusion model computationally massive and potentially negating any time saved in the solver.

Generation Overhead: While the generation time is reported as minimal (Table 4), this is for a very small number of diffusion steps (5-10). For larger, more complex problems (i.e., higher-resolution images), achieving high-fidelity generation may require significantly more steps, which could increase this overhead non-trivially.

### Questions
Scalability to Large-Scale MILPs: How do the authors envision VRG scaling to problems with $n \gg 10,000$ variables? The U-Net's computational cost scales at least quadratically with image resolution. At what point does the generation overhead (time and memory) for the high-resolution "solution image" become the new bottleneck, overwhelming the gains in solver search time?

Hyperparameter Sensitivity (Guidance): How are the guidance weights $\gamma_o$ and $\gamma_c$ determined? How sensitive is the model's performance (in terms of final solution quality and search efficiency) to these hyperparameters?

Nature of Captured Dependencies: The paper claims the U-Net captures variable interdependencies. Is there any qualitative analysis (e.g., via attention maps or gradient visualization) to demonstrate that the model is learning a representation of the MILP's constraint structure, or is it primarily acting as a powerful, spatially-aware density estimator that benefits from the Lagrangian guidance?

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
This paper proposes VRG, a feasibility-aware diffusion framework for mixed-integer linear programming (MILP). The key idea is to represent an MILP solution vector as a 2D image, allowing a U-Net-based diffusion model to capture variable correlations. The generation process is further guided by Lagrangian relaxation, which introduces penalty terms related to feasibility and optimality. Experiments on standard MILP benchmarks (SC, MIS, FC, CA) demonstrate that VRG achieves comparable or better objective values than Predict-and-Search (PaS) and Contrastive PaS (CoPaS) baselines.

### Strengths
1. The paper explores a promising and creative direction—integrating Lagrangian relaxation guidance with diffusion-based generative modeling for MILP.

1. The formulation of diffusion guidance and the connection to optimization regularization are clearly written and mathematically elegant.

1. The authors provide several theoretical results (e.g., optimization-equivalence, probability concentration) that make the framework appear theoretically grounded, although the insights are largely standard.

### Weaknesses
1. Mapping MILP solutions into 2D images lacks clear motivation and explanations. It does not preserve the structural properties of the original variable–constraint graph. The method is not permutation-invariant to variable ordering, meaning results may depend on the arbitrary order of variables. Moreover, such a representation likely harms scalability and cross-size generalization: all experiments are trained and tested at fixed problem sizes, and the paper gives no discussion on how $h$ and $w$ are chosen or how the model handles variable dimensions.
1. It is unclear whether the model is trained using optimal solutions as supervision or a weighted average target as in PaS. The text suggests that the ground-truth label is the optimal solution, which contradicts the practice in Predict-and-Search frameworks.
1. The proposed “Lagrangian-guided diffusion” is essentially a diffusion model with standard regularization penalties on feasibility and objective distance. The proposed theorems seem to lack novel insights. For example, Theorem 1 is a well-established property of Lagrangian duality. Theorem 2 merely rewrites a penalized objective as a KL-divergence, which is a routine energy-based trick. The claim that the analysis “provides formal guarantees on both feasibility and near-optimality” is over-stated, as no real convergence or approximation bound is proven.
1. The benchmarks are synthetic and relatively small—even the “large-scale” settings contain only around 1–2 k variables. No experiments are conducted on real or challenging datasets such as MIPLIB.
1. Several recent methods closely related to this paper are not discussed or compared, such as: Difusco [1] and T2T [2] for diffusion in combinatorial optimization; [3] for diffusion-based solution generation for integer programming; and DiffILO [4] for constraint-penalty-based solution generation for integer programming.
1. The reported performance of PaS and CoPaS being worse than SCIP/Gurobi contradicts prior literature. Hyperparameter settings for these baselines are not described—were they re-tuned? Some results, such as the small-scale FC benchmark, seem numerically inconsistent or mislabeled.
1. The ablation only tests the presence or absence of the guidance term. The paper does not isolate the effects of the diffusion component or the CNN visual encoder, nor does it analyze key hyperparameters such as $\gamma_o$​, $\gamma_c$, or $h$ and $w$. Without these, the actual source of performance gain remains unclear.

[1] Sun, et al., Difusco: Graph-based diffusion solvers for combinatorial optimization. NeurIPS 2023.

[2] Li, et al., T2t: From distribution learning in training to gradient search in testing for combinatorial optimization. NeurIPS 2023.

[3] Zeng, et al., Effective Generation of Feasible Solutions for Integer Programming via Guided Diffusion. SIGKDD 2024.

[4] Geng, et al., Differentiable integer linear programming. ICLR 2025.

### Questions
1. In Theorem 1, the statement “the sub-gradient method converges to a stable sequence” needs clarification.
1. How are the visual parameters $h$ and $w$ determined? Do different reshape orders affect results?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a VRG, a feasibility-aware generative framework that operates in visual space. The method represents the solution of an MILP problem as an image, and uses an image processing network (U-Net) to learn inter-variable dependencies. The Lagrangian relaxation guides sampling toward feasible, near-optimal regions. VRG generates diverse high-quality candidate solutions that define trust regions for downstream MILP solvers. Experiments demonstrate good performance over several benchmarks.

### Strengths
1.  The idea of reshaping MILP solution vectors into 2D images is interesting. This design leverages the spatial receptive fields of convolutional neural networks (CNNs) to explicitly model variable interdependencies.

2.  The paper is well-organized and presented clearly.

### Weaknesses
1. The paper claims PaS methods "assume decision variables are independent" but overlooks that PaS uses GNNs with message passing over MILP bipartite graphs (variable-constraint nodes) to capture dependencies. For 2-layer GNNs, variables aggregate information from others sharing constraints—this is not "independence" but explicitly considers the constraints and variables in the same constraints. Meanwhile, VRG’s CNN-based capture is limited to spatial receptive fields in the MILP-image, which depends on variable ordering (arbitrary for most MILPs) rather than constraint topology. The paper fails to acknowledge this parity or explain why CNN-based capture is superior to GNN-based capture.

2. The MILP-image is constructed by reshaping 1D solution vectors into 2D grids, so variables adjacent in the grid (and within CNN receptive fields) depend on input order—yet the paper does not analyze how variable ordering impacts performance. For example, random permutations of variables could break critical constraint-induced adjacencies, degrading dependency capture. Without a robustness analysis, the visual encoding’s reliability for general MILPs is uncertain.

3. The time cost of the inference of the diffusion model is unclear. 

4. VRG uses fixed-resolution MILP-images (e.g., (20,25) for small SC, (30,50) for large FC; Table 9) but does not address MILP instances with different sizes in a single model. Retraining or resizing is needed for instances with different sizes. The paper does not explore dynamic adaptations to handle different sizes.

5. The implementation details are missing. 

   1. The U-Net’s layer count, filter sizes, cross-attention head count, and GNN encoder depth for MILP instances are unspecified.

   2. The paper mentions 100 test instances per small-scale benchmark but provides no details on the training dataset size and the number of training instances. 

   3. The hyperparameters of the network and training process are missing. 

6. The paper only ablates Lagrangian guidance but omits ablations for core components: (1) No comparison to a GNN-based encoding (same diffusion/Lagrangian framework) to isolate the visual representation’s value. (2) No validation that diffusion’s multimodality improves robustness over deterministic generative models.

7. The paper compares to PaS and exact solvers but omits diffusion-based MILP competitors.

8. The author should analyze the sensitivity of the hyperparameters, such as the generation steps in the diffusion model. 

9. To demonstrate the performance, I suggest the authors conduct experiments on the real-world datasets, such as MILPLIB. At least, conduct experiments on the dataset from the Neurips competition, including item placement and workload appointment datasets.

### Questions
Please refer to the weakness part above.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper's main contribution is VRG a generative model for MILPs where feasibility
(constraint non-violation) is incorporated in the energy function of the
denoiser of a diffusion process.
This integration takes the form of a penalized version of the objective, hence the connection to Lagrangian relaxation. 

The proposition is backed up by two results:
1. First a so-called 'optimization-equivalence' result states that the guidance provided by the
feasibility awareness leads to same optima. 
2. Second a concentration guarantee which shows that the search space with the feasibility awareness is a restriction of
the general search space.

The generative model is parametrized by a convolutional neural network. The
neural architecture uses both the bi-partite factor graph that is common for
MILP encoding with a novel encoding of MILP variables as (visual) images. This enables the use of
Unet layers can be used to encode variable correlations.

Finally, a strong experimental setup shows that the method exhibits superior
performance to SOTA methods for ML-based heuristics for MILP solving.

However, 
while the title use `Lagrangian` and the model hinges on a scalar penalization of constraint violation akin to Lagrangian relaxation, the penalization $\gamma_c$ is a hyper-parameter that does not evolve nor guarantee optimality.
VRG returns an assignement for the continuous relaxation of the input MILP, and
must be processed further by a solver. In that sense, VRG remains a tool to
warm-start a solver.

### Strengths
1. Feasibility is incorporated seamlessy in the diffusion process guidance, and theoretical guarantees are provided.
2. The encoding of a generic MILP as an image is novel (to my knowledge) although its description could be improved
3. Strong experimental results that demonstrate the validity of the approach

### Weaknesses
1. The main weakness is about the use of a new MILP encoding. It is difficult to
   unravel the experimental setup to know whether the empirical improvements are
   caused by the feasibility mechanism or the neural architecture. There should
   be ablation studies to better understand the interplay between the contributions
2. The impact of binarization of variables is not really addressed and the normalization
   remains unclear. Can this model really handle integer variables?
3. There are numerous typos and presentation sometimes lacks rigor. For instance:
   - eq 7 $\mathcal{X^{*}}$ is not defined (until line 208), and the definition of
     $\cong$ is not given
   - line 190 : date -> data
   - line 220 again $\cong$:  is t different from $=$ used in equation 10?
   - lines 265-268 the dimensions of the resulting tensors look inconsistent  
   - 271 : typo learning -> learn
   - 630 : equation 31 lacks sum-to-1 normalization
   - 678 (eq 40) gradients are not indexed

### Questions
- Why can't the constraint $l \leq x \leq u$ be factored in $Ax \leq b$ in Equation 1? It
  seems like an unnecessary burden, and more importantly an unexplained idiosyncrasy. Are the bounds used in any specific way?
- The transformation in 4.1 (equation 5) looks like it already assumes binary
  variables which contradicts lines 174-175 where a normalization is applied
  afterwards. Can you explain?
- The placement of variables on the grid $\mathcal{X}$ must have an impact. How is this decided?
- The proof of concentration lacks a conclusion: how do you go from equation 37 to the claim?
- I am not sure that Table 4 reports total time (VGR+solver) or simply VGR. Can
  you elaborate and give the time breakdown between the two phases ?

### Soundness
3

### Presentation
2

### Contribution
2
