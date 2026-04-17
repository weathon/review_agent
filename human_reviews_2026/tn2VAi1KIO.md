# GenCP: Towards Generative Modeling Paradigm of Coupled physics

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Real-world physical systems are inherently complex, often involving the coupling of multiple physics, making their simulation both highly valuable and challenging. Many mainstream approaches face challenges when dealing with decoupled data. Besides, they also suffer from low efficiency and fidelity in strongly coupled spatio-temporal physical systems. Here we propose GenCP, a novel and elegant generative paradigm for coupled multiphysics simulation. By formulating coupled-physics modeling as a probability modeling problem, our key innovation is to integrate probability density evolution in generative modeling with iterative multiphysics coupling, thereby enabling training on data from decoupled simulation and inferring coupled physics during sampling. We also utilize operator-splitting theory in the space of probability evolution to establish error controllability guarantees for this “conditional-to-joint” sampling scheme. We evaluate our paradigm on a synthetic setting and three challenging multiphysics scenarios to demonstrate both principled insight and superior application performance of GenCP. Code is available at this repo: https://github.com/AI4Science-WestlakeU/GenCP.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents GenCP, a novel paradigm for multiphysics simulation that learns coupled physics from decoupled training data. 

**Key Innovation:** The elegant combination of flow matching's ODE formulation with operator splitting theory. The method trains separate flow matching models on decoupled data (learning $p(f|g)$ and $p(g|f)$), then composes them via Lie-Trotter splitting during inference to generate coupled solutions $p(f,g)$.

**Theoretical Contribution:** Establishes rigorous convergence guarantees by connecting probability density evolution with operator splitting.

**Significance:** This work bridges generative models with classical numerical methods, providing both theoretical rigor and practical efficiency for solving expensive coupled simulation problems.

### Strengths
- **Intuitive decomposition with strong empirical validation.** The core idea of learning coupled physics through separate conditional models is elegant and practical. The paper convincingly demonstrates this works: models trained only on decoupled data successfully capture strongly coupled dynamics, which surrogate baselines completely fail to model despite similar numerical errors. This validates that the probabilistic formulation genuinely captures coupling mechanisms.

- **Rigorous theoretical foundation bridging generative models and numerical analysis.** The connection between flow matching's velocity field decomposition ($v = v^{(f)} + v^{(g)}$) and Lie-Trotter operator splitting is non-trivial and principled. The formal convergence analysis via Wasserstein-1 distance provides error controllability guarantees rare in neural PDE solvers, with explicit tracking of splitting error ($\tau$) and learning error ($\varepsilon_f, \varepsilon_g$).

- **Careful decoupled data construction avoiding information leakage.** The experimental design modifies boundary conditions to ensure zero coupling information in training data while maintaining physical plausibility. This proves the method learns underlying conditional physics rather than memorizing coupling patterns.

- **Insightful analysis of baseline failure modes.** Beyond reporting metrics, the paper explains why deterministic surrogate methods fail: they miss oscillatory dynamics despite low numerical errors because they cannot handle mode errors and stochastic behavior. This clarifies that GenCP's probabilistic modeling addresses fundamental limitations of deterministic learning for strongly coupled systems.

### Weaknesses
- **Notation density limits accessibility.** The formulation rapidly introduces weak continuity equations and Lie-Trotter splitting without sufficient intuitive scaffolding. Key notations need clearer explanation before formal treatment.
    
- **Incomplete Decoupling.** The experiments acknowledge (Appendix D) that one training direction uses coupled rather than truly decoupled data due to negligible structural deformation in genuinely decoupled scenarios. This "half-decoupled" compromise contradicts the paper's emphasis on training exclusively from decoupled data, suggesting the method still requires partial access to expensive coupled simulations in practice.

### Questions
- The paper assumes decoupled data is easier to obtain (line 47), but your experimental protocol suggests otherwise. You still need to run solvers to generate it. Would it be more accurate to position GenCP's value as enabling the "reuse" of historical single-physics data or "integration" of cross-institutional datasets, rather than claiming acquisition cost advantages in general?

- Experimental Design: The paper acknowledges in Appendix D that structure-conditioned-on-fluid training uses coupled rather than truly decoupled data. Could the authors provide ablation studies comparing: (1) fully decoupled training for both directions, (2) the current "half-decoupled" setting, and (3) fully coupled training? This would help quantify the performance trade-offs and clarify whether truly decoupled training is feasible for both directions, which is central to the paper's claims.

- The success of GenCP relies on the assumption that decoupled datasets $\mathcal{D}_f$ and $\mathcal{D}_g$ contain sufficient information to recover the joint distribution. In your experiments, this is guaranteed by construction, since both datasets are derived from the same coupled solver with consistent physical parameters. Can you characterize the necessary and sufficient conditions on decoupled data for GenCP to work? Specifically:
  - Must $\mathcal{D}_f$ and $\mathcal{D}_g$ share the same physical parameters?
  - What level of parameter mismatch is tolerable between the two datasets?
  - Are there failure modes where learning $p(f|g)$ and $p(g|f)$ separately is provably insufficient to reconstruct the joint distribution $p(f,g)$?
  - How can practitioners assess the "coupling information completeness" of their real-world decoupled datasets before committing to training?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduces GenCP, a generative modeling framework for simulating coupled multiphysics systems with applications to fluid-structure interaction. Their core contribution is training on decoupled data which does not require coupled simulation data during the training process. During inference,however, the  fully coupled system can be predicted.

### Strengths
- According to the Reviewer, the general idea presented in the paper of integrating operator-splitting theory with generative flow models is novel. Using decoupled training and coupled inference later to reduce the cost of training data could potentially be very impactful as many AI4Science ML applications deal with the issue of spending significant compute on training data generation.

- The authors present theoretical guarantees for their framework. The method is not just a heuristic but directly connected to operator splitting theory used in numerical schemes.

### Weaknesses
Despite the core idea being novel, the Reviewer is unable to recommend the paper for acceptance at ICLR due to the following major issues:
- One of the central motivation of the paper is that the training data generation using decoupled systems only is cheaper but this was unfortunately never quantified. The reviewer recommends evaluating the cost of training data generation for coupled and decoupled generation.
- The comparisons to baselines should also include frameworks trained on the fully coupled data to estimate the change in accuracy by using decoupled data only.
- The authors only apply their method to, in the setting of FSI, small problems. However, as stated in the Appendix for one of their examples the authors do not use decoupled data only but coupled training data as the decoupled data contained no meaningful structural motion. According to the Reviewer, this undermines one of the central statement of the paper. If already for a small example, using decoupled training data only is insufficient, the Reviewer would expect for more complex system with more non-linear coupling the same issue. This would significantly limit the applicability of the new approach.
- The obtained relative errors on the coupled data test set seem relatively high and would be too high for most engineering applications. The Reviewer acknowledges that the authors perform better than reported baselines but the gap between results on the decoupled validation set and the coupled test set seems to high and would discourage currently training on decoupled data in the Reviewers opinion. Moreover, the Reviewer encourages the authors to add the aforementioned baselines on a fully coupled dataset.
- The authors use convolutional building blocks and do therefore require a regular grid. For FSI which often involves complex boundaries this is a significant limitation. In Figure 6, the cylinder is not round but the discretization artifacts are clearly visible. The authors should comment on this limitation.

### Questions
See weaknesses and:
- For the two FSI examples, the authors just report single error values but the Reviewer would have expected a distribution due to the probabilistic approach. Are these posterior mean values ?

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
This paper proposes a novel and elegant generative paradigm—GenCP—for coupled multiphysics simulations. By formulating the coupled physics modeling problem as a probabilistic modeling problem, the authors' key innovation lies in combining probability density evolution from generative modeling with iterative multiphysics coupling. This enables training on decoupled data and inference of coupled physics processes during sampling. The authors evaluate their paradigm on a synthetic dataset and two challenging fluid-structure interaction scenarios to demonstrate GenCP's fundamental insights and superior application performance. This is an article about applying AI to engineering physics.

### Strengths
Combining probability density evolution with iterative multiphysics coupling in generative modeling is a novel approach, and the problem setting is very clear.

### Weaknesses
The theoretical part of this paper is excellent, but I think the experimental part could be scaled up further.

### Questions
Have you considered a larger-scale experiment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The submission proposed a flow-based model to handle multiphysics simulation. The idea is simply: training two flow models independently but using another variables as the condition. It reduces the training complexity. I appreciate the method, but not satisfied with current experiments part.

### Strengths
I think the proposed method is easy to follow and the result is good. Also, I think the problem the authors are trying to solve is meaningful.

### Weaknesses
1. I think the experiments are not enough. It all concentrates on fluid dynamics, but it is better to combine fluid and rigid objects. It is convenient to treat the whole fluid system as a domain.

2. Also for the experiments part, I feel like the submission should add a baseline: training a single v(x_t, y_t) for all variables (coupled data), to prove the decouple the data is beneficial.

### Questions
Line 202, Line 207, I believe in the flow model, the vf = f1-f0, not f1-zf, the same for vg.

### Soundness
3

### Presentation
4

### Contribution
3
