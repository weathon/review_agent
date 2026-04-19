# SPFNO: spectral operator learning for PDEs with Dirichlet and Neumann boundary conditions

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 1, 5, 3

## Abstract
Neural operator has been validated as a promising deep surrogate model for solving partial differential equations (PDEs). Based on the spectral operator learning (SOL) architecture, an enhanced orthogonal polynomial neural operator we have developed significantly improved the method’s accuracy by precisely satisfying the boundary conditions (BCs), but is associated with Gauss-type grids, limiting comparisons on most public datasets. In this paper we introduce SPFNO, a novel SOL method, to learn the target operators on uniform grid datasets for PDEs with non-periodic BCs. Numerical results for various PDEs such as viscous Burgers’ equation, Darcy flow and coupled Allen–Cahn equations demonstrate the computational efficiency, resolution invariant property, and BC-satisfaction behaviour of proposed model. An accuracy improvement of approximately 1.7X–4.7X over non-BC-satisfying baseline is also noted. Furthermore, studies on SOL emphasizes the imporance of respecting BCs as a criterion for deep surrogate models of PDEs.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes SPFNO, a so-called spectral operator learning method to solve PDEs with Dirichlet and Neumann boundary conditions on uniform grids. SPFNO leverages trigonometric polynomials and sine/cosine transforms to precisely satisfy boundary conditions.

### Strengths
1. Explore methods on different Dirichlet/Neumann BCs to further improve the ability of neural operator series work.
2. Provides fast O(NlogN) transforms for spatial differentiation using spectral methods.

### Weaknesses
1. The proposed Semi-Periodic FNO appears to be overly simplistic and lacks in-depth analysis and exploration. For instance, the absence of alternative basis selection limits the thorough examination of its capabilities. As a result, the paper appears more akin to a technical report rather than a scientific research article.

2. The experiments are rough without in-depth analysis. For example, the paper lacks ablation studies to validate design choices like bandwidth of learnable matrices, no analysis provided on how accuracy scales with problem size and dataset size, and it does not discuss performance on other complex boundary conditions like Robin or mixed BCs. The most important thing is that it does not compare with other BC satisfying neural operator methods and claim it is sota which cannot convince me.

3.  There is no evidence that satisfying BCs improves generalization and robustness.

4. Limited hyperparameter tuning details are provided for model architecture and training.

### Questions
1. Can you expand studies to other boundary conditions like Robin, mixed, time-dependent, etc
2. Can you add ablation studies on model architecture choices
3. Can you demonstrate how accuracy changes with dataset size and problem dimensionality
4. Can you compare with other BC satisfying operator learning methods
5. Can you show if satisfying BCs improves robustness through noise tests, outlier evaluation, etc
6. Can you provide more details on model tuning, architecture search, regularization techniques used
7. Can you enhance readability by adding more architectural and algorithmic details to sections

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the problem of learning solution operators associated with partial differential equations (PDEs) with deep learning. The authors propose a new architecture, in a similar spirit of the popular Fourier neural operator, based on the discrete sine and cosine transform to enforce Neumann or Dirichlet boundary conditions strongly in operator learning problems, and perform convolutions in the feature space using Discrete Cosine/Sine transforms (DCT/DST). Then, the authors perform deep learning experiments and compare their approach using FNO and other architectures.

### Strengths
- The FNO enforces periodic boundary conditions strongly but Dirichlet or Neumann are more common in practical applications and are enforced weakly by current neural operator architectures.
- The numerical experiments show that the authors' approach achieve close to machine precision error on the boundary conditions, which is an improvement over the FNO.

### Weaknesses
- The paper is not well-written and contains many typos and spelling mistakes.
- The relation between this work and the previous work by Liu (2022b) is very confusing. The authors mention that their work is an improvement over Liu (2022b): the difference is that Liu et al's approach requires Gaussian grid to evaluate the neural operator while the present paper allows for uniform grid. First, this seems to be a minor contribution. Secondly, Subsection 2.1 and Fig. 2 are copied from Liu 2022b but not references as such. 
- The SPFNO approach is not well introduced.
- The experimental results also contain several drawbacks. First, the experiment setup is not provided, and the number of parameters between models is not comparable. In Example 2, OPNO (Liu, 2022b) outperforms the paper's approach, but is not evaluated in examples 3 and 4. The training time for FNO is much faster than SPFNO (factor 1.83 and 3.58 for experiments 3/4, respectively).

### Questions
- What is the motivation for using uniform grid as it might lead to aliasing errors?
- 2nd paragraph of p.2 in bold: which fast transform?
- There is no description of Fig. 1 in p.2.
- First paragraph of Section 2.1: "respecting boundary conditions exactly" does not make sense, it should be strongly.
- Last sentence of Section 2.1: "probably sparse structure": what does that mean?
- Theorem 2.1: there is no assumption on the function f (i.e. its function space).
- Theorem 2.2: the statement is not clear, what does strictly satisfy the boundary conditions mean?
- Appendix A.1: The proof of Thm. 2.1 is not rigorous: where does Eq.(11) comes from? The authors should state that it's the representation of f in an orthonormal basis of L^2. The coefficients a_m, b_n should be introduced/defined. Corollary A.1 comes without a proof.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose SPFNO, which is a spectral operator leaning (SOL) method that enforces boundary conditions within Neural Operator methods.  By incorporating boundary conditions and techniques from numerical analysis, the proposed method is shown to be more accurate than FNO and other data-driven methods on a range of experiments but is not compared to BOON which also enforces boundary conditions in Neural Operators to obtain higher accuracy using an alternative approach. In particular, the authors propose to use a sine basis function for Dirichlet BC and cosine basis function for Neumann BC, which enforces the constraint on the derivative of the solution by using odd and even extensions, respectively.

### Strengths
- It is good to incorporate ideas from numerical methods such as spectral methods and boundary conditions to improve the accuracy of pure data-driven methods for PDEs.
- Nice mention of stability, numerical convergence and matrix sparsity from numerical methods that can be advantageous to leverage within DL methods.
- It is nice to see that as in BOON (Saad et. al, ICLR 2023) enforcing the boundary conditions improves the accuracy of the model.
- Good overview of numerical methods in the introduction.
- Good that the method does not break the desired resolution invariance property of neural operators.
- Nice test case on challenging shock problem with 1D Burgers equation where most DL methods are highly oscillatory. It looks like in Figure 1b though that the resulting solution profile from SPFNO still has damped oscillations, violating the maximum principle and monotonicity of the true solution.
- Nice visualization of the plots but Figure 1e is much too small.
- Nice and simple idea to use a sine basis for Dirichlet BC and cosine basis for Neumann BC.
- It is nice that the authors provide corresponding theory for their method and should that the BC are strictly satisfied.
- Interesting finding that it is easier to learn Dirichlet than Neumann BC which involved derivatives. The error in approximating the derivative also comes into the schem.

### Weaknesses
- It is better to first motivate the PDE problem definition and application in science and engineering before diving into Neural Operators specifically.
- There are missing references to other state-of-the-art SciML classes of methods other than Neural Operators including MeshGraphNets (Plaff et. al, ICLR 2021) and PINNs (Raissi et. al, 2019).
- One major weakness is that the related BOON method from ICLR 2023 "Guiding continuous operator learning through Physics-based boundary constraints" by Saad et. al, is not cited or compared to. This is the first method to enforce boundary conditions in arbitrary Neural Operators, resulting in a 2x-20x improvement.
- Multi-wavelet Neural Operator (Gupta et. al, ICLR 2021) is also not compared to in the evaluations.
- It is also incorrectly stated in the intro that most of the public datasets have periodic BC. Please find the datasets from BOON with Neumann and Dirichlet BC in the open-source repo https://github.com/amazon-science/boon. 
- The cited multi-wavelet NO (Gupta et. al, NeurIPS 2021) also leverages orthogonal polynomials as well as the state-of-the-art MeshGraphNets (Plaff et. al, ICLR 2021)
- The wide applicability of FNO is more due to the ease of the use and efficiency of the pytorch FFT implementation rather than its "straightforward training process".
- Sine and cosine bases may still not be ideal bases for hyperbolic conservation laws with shocks.
- The authors mention Robin BC but do not show experiments with it.
- The architecture sketch in Figure 2 can be moved to an Appendix since the architecture is very similar to FNO just with using separate sin/cos bases rather than Fourier, which also limits the novelty.
- Page 6 is sparsely filled with results and no text. There is room for additional analysis and more PDE experiments including see the lid-driven cavity flow Navier-Stokes Dirichlet BC test case from BOON https://github.com/amazon-science/boon.
- The results analysis is missing and without a discussion it is very difficult to interpret Figure 3.
- I'd like to see the proof for Neumann BC since that involved the derivative. Also the proof for Corollary A.1 is missing.

Minor
- Importance is spelled incorrectly in the abstract.
- The second sentence in the introduction is extremely long and should be broken up at the applications of FNO to weather forecasting.
- "after acceptance" rather than "after accepted" in last paragraph of the intro.
- Section 3 should be plural "Numerical Experiments"
- Commas missing after some equation
- "verry" close typo at the end of page 5.
- In 3.4, it should read "the task is to learn"
- Conclusion should read, "Although the paper focuses"
- "yeilds" typo in A.1

### Questions
1. Since sine or cosine basis is used, why are the oscillations damped with SPFNO on problems with shocks, i.e., Burger's equation here?
2. How can this method be extended to non-uniform grids?
3. Does LSM beat MeshGraphNets? The wording on this method being state-of-the-art is quite strong since it also doesn't compare to MeshGraphNets (Plaff et. al, ICLR 2021).
4. There is a lot of comparison to Liu et. al, 2022b (OPNO), what is the differences in terms of novelty since Liu et. al, 2022b also satisfies the boundary conditions.
5. In particular, in table 1a, it looks like OPNO wins in a majority of cases over the proposed CosNO for Neumann Bc - why is this?
6. The resolution $n$ is quite fine in Table 1 up to 4096 nodes for 1D Burgers which is extremely fine and not required for numerical methods in 1D. They can be accurate for $n=64, 128, 256$ and would have already converged to very low error on order of less than half machine precision $<1e-8$ by 4096. Why are such fine grids considered here and why is the error is still quite large on the order of $1e-2$?
7. Also then in 2D the resolution is much coarser - is there due to the computational issues moving from NO in 1D to 2D, which is the only reason DL methods should be considered over numerical methods. If both suffer the curse of dimensionality, its hard to see the use of DL methods.
8. Is $n$ supposed to be $N$?
9. It is mentioned that example 3.3 is from PDEBench and the baselines are taken from there - is it a fair comparison with the same hyperparameters and number of parameters for reproducibility.
10. Why does LSM perform better in Table 4 than Sin-NO in the worst case error?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper authors propose a new Spectral Operator Learning (SOL) framework for solving partial differential equations (PDEs) that involve Dirichlet or Neumann Boundary Conditions. To enforce non-periodic boundary constraints, the proposed method utilizes the discrete sine and cosine transform. The  proposed architecture was evaluated on a variety of PDE problems and compared against a few baselines including original FNO, Unet, Latent Spectral Models (LSM), and Orthogonal Polynomial Neural Operator (OPNP). The results indicate precise satisfaction of boundary constraints and higher overall accuracy for some test cases.

### Strengths
- It is great to see that there are researchers in the field of SciML who believe a pure data-driven approach with the hope of learning all physics constraints "implicitly" from data is NOT enough, and it is important to use readily available physics information in addition to input datasets.
- Authors considered a good variety of PDE problems with different applications and level of difficulty, and it is nice to see their approach is capable of enforcing Dirichlet and Neumann boundary constraints at machine precision level.
- It is also nice that authors shared their codebase and data, which are useful for benchmarking with other researchers.

### Weaknesses
- My main concern is about the originality of this work. The idea of using spectral methods/orthogonal polynomial transformation in neural networks for solving PDEs is not really a novel idea. Authors already mentioned two important methods, LSM and OPNO, which utilize the power of spectral methods, and they seem to have competitive overall performances. To my understanding, the only update in this work is the use of different orthogonal transformation (discrete Cosine or Sine) while keeping the rest of configuration the same as FNO. 
- I would like to challenge the statement of "neural operators trained on coarse grid can predict results on finer mesh". Authors should keep in mind that neural operators are purely data-driven methods and may not perform well on finer grids if they are not given the chance to learn the pattern in sub-scale grids. Also it is not scientifically appropriate to mention something (even from other papers) and not to provide numerical evidence supporting the statement.
- I found that there are a few closely related papers missing from the references. Just to mention few examples:
   - (BOON) Guiding continuous operator learning through Physics-based boundary constraints (https://arxiv.org/pdf/2212.07477.pdf) 
  - Neural Q-learning for solving PDEs ( https://www.jmlr.org/papers/volume24/22-1075/22-1075.pdf)
  - Physics-Embedded Neural Networks: Graph Neural PDE Solvers with Mixed Boundary Conditions (https://proceedings.neurips.cc/paper_files/paper/2022/file/93476ae409ae3246e22a9d4b931f84ed-Paper-Conference.pdf)
  - Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physics-informed neural networks (https://www.sciencedirect.com/science/article/pii/S2405844023060280)
  - and more!
It would be nice that authors make sure that they have done a comprehensive literature review and this will further improve the credibility of the research conducted. For example, the first reference mentioned above actually enforces boundary conditions by systematically adding additional layer to a neural operator architecture, and could be a great baseline for comparison. 
- Lastly, the presentation of some sections of the paper could have been further improved. For example, the evaluation procedure is not described clearly. It is not clear what resolutions considered during training and what is considered during evaluation, whether the evaluations are in-domain predictions (seen at training time) or out of domain, what is the number of replicates of the same problem and whether they reported ensemble results or the results come from one single run, etc. These details are important for future benchmarking with other researchers and authors should

### Questions
As addressed briefly above in "weaknesses" section, I wonder if the numerical results reported came from one single run of experiment or it is the ensemble of multiple runs? It is important that authors conduct multiple experiments of the same problem with random seeds and report both mean and standard deviation to better illustrate the robustness of their results.
- For majority of the results one can see that some baselines such as OPNO for Burgers equation, and LSM for Darcey flow can perform similarly or even better than the proposed method. Could authors provide more detailed discussions on the results and comment on why their method may not result in the most accurate predictions among all baselines. This shows that although boundary constraints are satisfied precisely then there are other sources of errors in their method. What are potential sources?
- Did authors conduct any experiment to investigate the "resolution invariance" property of neural operators, because they list it as one of their contributions? For this, it would be nice to train their model on coarse mesh and evaluate on finer mesh, and vice versa and see how model predictions look like for each scenario.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor
