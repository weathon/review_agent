# Neuro-Symbolic AI for Analytical Solutions of Differential Equations

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Analytical solutions to differential equations offer exact insight but are rarely available because discovering them requires expert intuition or exhaustive search in large combinatorial spaces. We introduce SIGS, a neuro-symbolic framework that automates this process. SIGS uses a formal grammar to generate only syntactically and physically valid building blocks, embeds these expressions into a continuous latent space, and then searches this space to assemble, score, and refine candidate closed-form solutions by minimizing a physics-based residual. This design unifies symbolic reasoning with numerical optimization; the grammar constrains candidate solution blocks to be proper by construction, while the latent search makes exploration tractable and data-free. Across a range of differential equations SIGS recovers exact solutions when they exist and finds highly accurate approximations otherwise, outperforming tree-based symbolic methods, traditional solvers, and neural PDE baselines in accuracy and wall-clock efficiency. These results are a step forward integrating symbolic structure with modern ML to discover interpretable, closed-form solutions at scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes SIGS (Symbolic Iterative Grammar Solver), a neuro-symbolic framework for discovering analytical (closed-form) solutions of PDEs.

The key idea is to construct a grammar-based atom library and use a Topology-Regularized Grammar VAE (TGVAE) to learn a smooth latent space of symbolic expressions. SIGS then performs a two-stage search: 

- Stage 1: cluster latent codes and interpolate within clusters to identify promising candidate structures;

- Stage 2: refine constants through gradient-based optimization. Experiments across elliptic, parabolic, and hyperbolic PDEs show that SIGS can recover exact or near-exact symbolic solutions and outperform recent symbolic discovery baselines (HD-TLGP, SSDE) in accuracy and efficiency.

### Strengths
1. Novel neuro-symbolic design: Integrating grammar-level atoms, latent interpolation, and geometric/topological constraints is a creative and elegant framework.
2. Strong empirical performance: On selected PDE benchmarks, SIGS accurately reconstructs analytical expressions, sometimes achieving machine precision.
3. Well-motivated two-stage pipeline: The coarse-to-fine (structure search → constant refinement) approach is conceptually clean and provides interpretability.

### Weaknesses
1. Dependence on handcrafted Ansatz and grammar: The system assumes access to an appropriate symbolic library. When the correct building blocks are missing, SIGS likely fails; this is not tested quantitatively.
2. Limited fairness and transparency in baselines: It is unclear whether competing methods (e.g., HD-TLGP, SSDE, FEniCS) had access to the same symbols, constants, or data budgets. Table 2 comparisons could be biased.
3. Scalability and robustness not demonstrated: The method is validated on PDEs with known, smooth analytic forms. Its behavior on nonlinear, discontinuous, or noisy settings is untested.
4. Motivational gap vs. numerical solvers: Since SIGS can be slower and less accurate than classical solvers (e.g., FEniCS), the paper should articulate clearer motivation for discovering symbolic forms instead of direct numerical solutions.

### Questions
1. Ansatz sensitivity: How does SIGS perform when the provided Ansatz or atom library omits key operators (e.g., removing tanh for Burgers)? Please quantify degradation.
2. Residual evaluation: Have you computed the residuals of the recovered analytical expressions of Poisson equation in Table 3? Please compare residuals between SIGS and FEniCE under different spatial resolutions. And how about other boundary conditions, such as Neumann and periodic?
3. FEniCS resolution: What spatial resolution was used for FEniCS in Table 2? How does the FEniCS error and runtime change with mesh refinement, and how do these compare to SIGS?
4. Motivation for symbolic discovery: Given that FEniCS achieves higher accuracy faster, what is the practical motivation for SIGS? Is interpretability the main benefit, or does SIGS generalize across PDE families?
5. Failure modes: Can you show examples where SIGS fails or outputs incorrect forms, and analyze why?

Typo: Inconsistent of poisson equation form between 1175 line and 1313 line.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The aim of this paper is to automatically find analytical solutions/closed-form expressions of PDE without relying on observational data, and give a scalable solution to the combinatorial explosion and illegal expression problems of traditional symbolic tree search.
Methods: SIGS based on formal grammar and topological regularization grammar VAE (TGVAE) are proposed: The limited feasible expression is constructed with the hierarchy of "Ansatz+atoms", and the discrete expression is embedded into the smooth latent space for structure search and parameter refinement.
Experiments show that the proposed method is superior to the strong baseline in accuracy and efficiency on multiple classes of PDE, and has the advantages of data independence, interpretability, and scalability.

### Strengths
1. This paper proposes A method to obtain the analytical solution of PDE (although there is more than one such work, for example, Closed-form Solutions: A New Perspective on Solving Differential Equations). Unlike previous methods, the generated expressions are "mathematically valid and physically meaningful".


2. SIGS seems to give good results

### Weaknesses
1. I think the biggest innovation of this paper is that it is possible to obtain an analytical solution of a PDE, but unfortunately, this idea has been previously worked on by many people, such as Closed-form Solutions: A New Perspective on Solving Differential Equations. And this article doesn't even cite that article!

2, it is mentioned that the generated expressions are "mathematically valid and physically meaningful", which is not new in the field of symbolic regression and has also been well solved.

3. It is felt that this method is a combination of existing methods and is not too innovative.

### Questions
Q1: Please analyze how your way of making expressions "mathematically valid and physically meaningful" differs from previous symbolic regression methods, and what are your innovations and advantages?

Q2: As you mentioned in your article, many of the expressions obtained by previous GP and RL-based methods are invalid, but previous symbolic regression methods impose constraints and handle them, and work well. So I don't think your assumption is meaningful.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SIGS (Symbolic Iterative Grammar Solver), which is a neuro-symbolic framework that discovers analytical solutions to differential equations by combining symbolic grammars with deep latent-space optimization. The approach constructs candidate functional forms (Ansatze) using a formal grammar of atoms (e.g., polynomials, trigonometric functions, exponentials), which are then embedded into a continuous manifold using a Topological Grammar Variational Autoencoder (TGVAE). This enables smooth optimization of symbolic expressions through gradient-based search while maintaining symbolic interpretability. The method is validated on several canonical PDE families (Burgers, Poisson, Schrodinger, and wave equations). SIGS achieves exact or near-exact recovery of ground-truth analytical solutions, outperforming symbolic regression and hybrid neural PDE solvers. It also shows robust performance under noise and demonstrates interpretability advantages over black-box neural solvers.

### Strengths
This is an innovative and well-executed paper that pushes the boundary of symbolic and neural hybrid modeling. The central idea of representing symbolic solution structures through a grammar-based latent manifold is both original and conceptually elegant. The method bridges symbolic regression and deep learning by introducing a structured, differentiable search space, allowing neural optimization to operate over interpretable symbolic forms. The theoretical framing is rigorous, with clearly defined grammar rules, topology-preserving constraints in the TGVAE, and a solid justification for why the latent manifold preserves functional equivalence among expressions. The methodological novelty lies in using the grammar-VAE coupling to enable continuous symbolic optimization, which is a meaningful advance over existing neuro-symbolic PDE solvers. Empirically, the results are comprehensive and convincing. SIGS is benchmarked across multiple PDE categories, with both analytical recovery and quantitative accuracy metrics. The figures and tables are clear, and the visual comparison between recovered and ground-truth solutions is compelling. Ablation studies and timing analyses add credibility to the claims. The writing and presentation are polished and well-organized; complex ideas are explained with appropriate examples, and the motivation and related work sections are thorough. Overall, the paper is a strong contribution that demonstrates how neuro-symbolic methods can recover physically meaningful, interpretable solutions to PDEs.

### Weaknesses
While the contribution is conceptually strong, several aspects could be improved for clarity and generalization. First, the scalability of SIGS to higher-dimensional or more chaotic systems remains unproven. The method depends on handcrafted grammars and pre-specified Ansatz templates, requiring substantial domain expertise to design. This semi-manual setup could limit its practical use for complex, real-world systems where the appropriate functional vocabulary is unknown. Second, although the authors claim computational efficiency, the runtime and scaling analysis is only qualitatively discussed. A more detailed comparison of times, gradient steps, and scaling with grammar size would better substantiate the efficiency claim. Additionally, the method is evaluated mostly against symbolic or PINN-type baselines, not against neural operator architectures such as FNO or DeepONet, which are now standard references in PDE learning. Including such comparisons would help contextualize the performance advantage. Finally, the technical presentation, while mathematically correct, is dense in sections describing the TGVAE regularization and grammar construction. These could be made more intuitive with small worked examples illustrating how grammar generation and latent search interact. Despite these issues, the weaknesses are primarily about scope and exposition, not correctness.

### Questions
1) How sensitive is SIGS to the choice of grammar primitives or Ansatz structure? Could an incorrect or incomplete grammar prevent the discovery of correct solutions?

2) Have you tested the method on approximate analytical solutions or PDEs with no closed form, where SIGS might produce interpretable approximations?

3) Can the framework scale to parameterized PDE families or higher-dimensional systems, and if so, how does the latent search complexity grow with grammar size?

4) How would SIGS compare with Neural Operators (FNO, DeepONet) in accuracy and efficiency for continuous solution families?

5) Could the symbolic latent space be combined with physical constraints or PINN losses to enable hybrid symbolic–numeric discovery?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes a formal grammar-based approach for discovering analytic solutions of PDEs. The authors use a pre-trained variational autoencoder trained on expressions that fit a certain ansatz. The latent space of the autoencoder is then used to search for solutions fitting the PDE.

While a potentially interesting methodology for symbolic regression, I am not convinced by the chosen application and experiments.

### Strengths
The authors propose a conceptually interesting approach to incorporating inductive bias, in terms of the chosen ansatz, into the grammar-based search using the latent space of an autoencoder.

### Weaknesses
First, it is not at all clear what the real-world application is for this kind of analytic solution discovery. Outside of linear PDEs, most PDEs do not admit analytic solutions except in very simple geometries/boundary conditions.

Relatedly, most of the PDEs studied in this work are linear PDEs with very well-understood general solutions. There is no need for symbolic discovery for these linear PDEs. The only nonlinear PDE considered here is Burgers equation, and the authors had to include the relevant basis function (tanh) to handle that equation, which makes the solution trivial.

Regarding the claim and experiments with approximate solutions, it is not clear what advantage this approach has over simply looking at the numerical solution. For a fast approximate solution, you could have achieved the same result by just choosing to expand the solution in a particular basis set and fitting the linear combination of basis elements.

### Questions
1. What applications do you see this kind of method being used for?
2. What advantage can you demonstrate for this approach over classical analytic methods?
3. Given the simplicity of the discovered solutions (especially for the one nonlinear PDE), it is not clear what role the main contribution of the paper is playing. Why not naively restrict to the form of the given ansatz during a direct search rather than using a latent space?

### Soundness
2

### Presentation
3

### Contribution
2
