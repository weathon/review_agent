## Human Reviewer 1

### Summary
This paper proposes Flow Expander (FE), a framework that expands the support of pretrained flow or diffusion models under verifier constraints. The method formulates verifier-constrained entropy maximization as a mirror-descent optimization problem over probability measures, consisting of two alternating steps: an Expansion step that increases entropy and explores new modes, and a Projection step that enforces validity through a soft-verifier function. The authors provide theoretical convergence guarantees and demonstrate the approach on toy 2D examples and QM9 molecular conformers. The work is theoretically well-motivated, connecting flow-based generative modeling with constrained optimization, but its empirical validation and baseline clarity remain limited.

### Strengths
**Strong motivation and relevance.**
The paper tackles an important limitation of pretrained flow and diffusion models—namely, their inability to explore beyond the data manifold while maintaining sample validity. The idea of integrating verifier-based constraints into generative model fine-tuning is timely and relevant for scientific design applications (e.g., molecular or material generation).

**Principled formulation.**
The proposed *Flow Expander* framework is grounded in a clear optimization principle: verifier-constrained entropy maximization. Casting this as a mirror-descent problem in the space of probability measures provides a mathematically elegant and unified view of exploration under validity constraints.

**Theoretical completeness.**
The paper presents a solid convergence analysis. The proofs, though dense, follow established mirror-descent theory and offer formal justification for the proposed update rule.

**Potentially general concept.**
The introduction of a *soft-verifier* mechanism is conceptually interesting and, if properly extended, could provide a practical interface between learned generative models and rule-based or simulation-based validity filters used in real-world design workflows (e.g., high-throughput screening, physical constraints, or chemical property checks).

### Weaknesses
**Limited and unconvincing experiments.**
The empirical evaluation is restricted to 2D toy examples and a small-scale QM9 conformational generation task. These setups are insufficient to demonstrate practical effectiveness or scalability of the proposed method. The results primarily serve as proof-of-concept demonstrations rather than evidence of real-world impact or generalization capability.

**Unclear and unverifiable baselines.**
The paper cites Uehara et al. (2024, Section 8.2) as the source of the “CONSTR” baseline. However, that section contains only a theoretical corollary without any algorithmic description, implementation details, or experimental setup, making it impossible to reproduce or verify the reported baseline results. Moreover, many components of the proposed method appear to be directly borrowed from S-MEME (e.g., the mirror-descent formulation, entropy maximization objective, and theoretical convergence argument), yet the paper does not clearly delineate which parts are newly introduced and which are adapted from prior work. This lack of transparency in baseline selection and methodological novelty significantly undermines the credibility and reproducibility of the experimental claims.

**Severe clarity and notation issues.**
The paper suffers from significant readability and presentation problems:
1. Misuse of notation and inconsistent subscripts/superscripts.
2. line (234) incorrectly states $\delta\mathcal{G}(\mu)\in F(\mathcal{X})$, though the functional derivative should be a function over $\mathcal{X}$.
3. Algorithm 4 is referenced but missing.
4. Symbols such as $F(\mathcal{X}), \mathcal{G}_t$, and $\mathcal{L}(Q)$ are repeatedly overloaded, making the derivations unnecessarily hard to follow. These issues significantly reduce the accessibility of an otherwise theoretically interesting paper.

**Applicability and experimental realism.**
The introduction of *soft-verifiers* is an interesting and potentially powerful concept, as it could, in principle, allow high-throughput filtering strategies—commonly used in drug and material discovery—to be incorporated into generative models. However, the paper does not provide convincing examples where this mechanism becomes practically important. Beyond toy examples, the molecular conformational experiments hold limited chemical relevance: in conformational space, the goal is typically to reproduce the Boltzmann distribution rather than to impose external validity filters, so the verifier concept is of marginal utility.

### Questions
**Separation from S-MEME.**
Many components of the proposed Flow Expander (e.g., entropy maximization, mirror-descent formulation, convergence proof) appear closely aligned with S-MEME. Could the authors clearly specify which elements are newly introduced in this paper (e.g., verifier projection) and which are inherited from S-MEME or related prior works?

**Applicability to discrete domains.**
The proposed formulation assumes a continuous variable space, where functional gradients and mirror-descent updates are well defined.
How could this framework extend to **discrete or combinatorial domains** such as molecular graphs or protein sequences, where the notion of a variational gradient is not clearly defined?
In particular, can the *soft-verifier* idea be adapted to these settings, which are arguably more relevant to real-world drug or material discovery?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
1

---

## Human Reviewer 2

### Summary
Flow and diffusion models are typically pre-trained on limited available data. As a result, they tend to generate samples from only a narrow portion of the feasible domain.
To address this limitation, the authors assume access to a verifier and propose adapting a pre-trained flow model so that its induced density expands beyond regions of high data availability. They pose the key question:
“How can we leverage a given verifier to adapt a flow or diffusion model to generate designs beyond high data-availability regions while preserving validity?”
The authors consider two types of verifiers:
•	Strong verifier: a function nu: X -> {0,1} that characterizes validity exactly, i.e., nu(x)=1 if and only if x is valid.
•	Weak verifier: a function that acts as a filter—it rejects some invalid designs but may fail to detect others (formally, nu(x)=0 => x is invalid).

### Strengths
Major contributions: 
•	Flow Expander (FE), a principled probability-space optimization scheme
•	A theoretical analysis of the proposed algorithm
•	An experimental evaluation of FE

It is a well-written paper, with new ideas and interesting results. 

I think many researchers in our community will appreciate this paper.

### Weaknesses
•	The paper is somewhat dense and not always easy to follow.

•	The numerical experiments are somewhat limited. I appreciate both the illustrative examples and the results on the molecular design task, but I wish the paper included more high-impact, real-world examples where verifiers exist.

### Questions
•	In the description of Continuous-time Reinforcement Learning, states and actions have been defined, but I think the definition of reward is missing.

•	Line 175: If Omega_v is not a bounded set, then without further constraints, there is no maximum entropy distribution. Is there a simple way to generalize Problem 5 to this unbounded setting?

•	Instead of maximizing entropy, can we get reasonably good results by simply maximizing the variance of the distribution instead?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper addresses the problem that pre-trained generative models (flows, diffusion) tend to sample from a narrow part of the valid design space, which is a key limitation for scientific discovery. The authors propose to "expand" the model's density by leveraging an external verifier. The paper formalizes this into two problems: Global Flow Expansion (using a perfect strong verifier) and Local Flow Expansion (using an imperfect weak verifier). 

To solve these, the paper introduces Flow Expander (FE), a mirror descent algorithm. A key idea is to formulate the optimization objective over the entire noised state space of the flow process ($Q^\pi = \{p_t^\pi\}_{t \in [0,1]}$) rather than just the final time-step $p_1^\pi$. This is claimed to provide a principled way to avoid score divergence issues near $t=1$. The paper provides theoretical convergence guarantees and presents experiments on 2D illustrative tasks and a molecular conformer generation task (QM9).

### Strengths
* Addresses how to leverage pre-trained generative models to explore novel and valid regions of a design space, moving beyond the original data distribution.
* Formalizes the problem into Global Flow Expansion (using a strong verifier) and Local Flow Expansion (using a weak verifier).
* Lifts the optimization objective from the final time-step ($p_1$) to the entire noised state space ($Q^\pi$) to theoretically mitigate the score divergence problem that occurs as $t \to 1$.

### Weaknesses
* One weakness is the use of potentially uninformative baselines. The paper compares FE (a "search + constraint" method) against "search-only" (S-MEME/FDC) and "constraint-only" (CONSTR). This comparison is not fully informative, as FE is designed to outperform them. A fair and important baseline would be unconstrained exploration (FDC/S-MEME) followed by post-hoc rejection sampling using the verifier. Without this, the practical value of FE's complex optimization is unknown.
* The method's reliance on a differentiable verifier is a considerable practical limitation. The chosen solver (Adjoint Matching, Alg. 3) requires gradients from $\log v(x)$. Most real-world verifiers for scientific discovery (e.g., RDKit SanitizeMol, physics simulators) are black-box and non-differentiable. The paper's workaround (smoothing a simple function in App G.1, G.5.2) does not solve this general problem, limiting the method's applicability.
* The experimental validation is not fully convincing. The 2D experiments can be sensitive to tuning, and the QM9 dataset is too small-scale to demonstrate robustness.
* There is a notable absence of hyperparameter ablation studies for the key parameters ($\alpha, \gamma_k, \eta_k$).
* This is particularly concerning for the L-FE molecular experiment. The chosen parameters ($\alpha=9$, $\gamma_k=0.00001/(1+k)$) are so conservative that the effective step size is $\tilde{\gamma}_k \approx 0.0001$ and the KL weight $\beta=0.9$. This suggests the model is moving very little from the pre-trained state. The claimed high validity (81%) is likely just the original model's high validity, not a product of the algorithm.
* As discussed in Appendix H, $\alpha$ and $\gamma_k$ are entangled, jointly determining the effective step size $\tilde{\gamma}_k$. This implies the method is likely quite sensitive to hyperparameter tuning, but this important aspect is not analyzed.
* The computational cost appears very high. The algorithm requires $\approx 2 \times K \times N$ full model fine-tuning runs (e.g., $2 \times 8 \times 4 = 64$ in the L-FE experiment). This may be impractical for large-scale problems, and the paper makes no analysis of this cost or scalability against other baselines.
* A potential theory-practice gap exists. The convergence guarantees (Thm 5.2) rely on assumptions (E.1, E.2) about the solver, but there is no proof or justification that the actual solver used (Adjoint Matching, Alg. 3) satisfies these assumptions.

### Questions
* Can you provide a direct comparison against the FDC + post-hoc rejection sampling baseline? This would be very helpful to demonstrate that your complex constrained optimization is superior to a simple filter.
* How do you propose to use FE with a truly black-box, non-differentiable verifier (e.g., a hard RDKit sanitization check)? This seems to be the most common and important use case.
* Please address the question that your L-FE experiment parameters ($\alpha=9$, $\gamma_k \approx 0.00001$) are so conservative the model is "not moving" from the pre-trained state. Can you provide a hyperparameter ablation study for $\alpha$ and $\gamma_k$ to show how the diversity/validity trade-off changes?
* To help us understand the method, can you provide an ablation showing the results of running only the EXPAND step and only the PROJECT step for $K$ iterations?
* Can you provide an experiment for Global Flow Expansion on a real-world dataset, not just a 2D toy problem?

I would be willing to raise my score if the authors can thoroughly address the concerns raised in the weaknesses and questions section. In particular, the concerns regarding the experiments (ablation, large-scale dataset).

### Soundness
2

### Presentation
3

### Contribution
1

### Rating
2

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes Flow Expander (FE), a general framework for expanding the support of pre-trained flow models to cover the valid design space more uniformly. The method uses verifiers—functions that check the validity of samples (e.g., physical or chemical constraints)—to guide exploration and improve the generative model’s coverage beyond its initially narrow region.

### Strengths
- Introduces a method to expand flow models using strong or weak verifiers to expand diversity.
- The scheme allows practical, gradient-based fine-tuning compatible with existing flow and diffusion models.

### Weaknesses
- What is $\lambda_t$ and $\gamma_t$ in Eq. 8 and 9? Are these weightings and discount factors? And how are they chosen?
- How do you parametrize the verifier function $v(x)$, is it modeled like a Gumbel-Softmax? Also, I wonder if it's strict to have $v(x)$ to be bounded, is the method extendable if one has an unbounded $v(x)$?
- I am not sure about the downstream applications of this method in drug discovery, as currently it's only restricted to the bounded verifier objective to check validity or bonds. Usually, this can also be done via BO in latent space by embedding priors with GP or just generally with rejection sampling in latent space to get valid molecules only, similar to an explore and exploit-based method. If the $v(x)$ can be extended to the unbounded domain such that one can use molecular properties to guide it, then it would be a good contribution.
- Does the model incorporate whether the verifier is noisy?
- Could FE be viewed as iteratively reshaping an implicit energy landscape defined by verifier penalties?

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3