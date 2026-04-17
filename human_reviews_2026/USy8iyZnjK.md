# Trail Mix: Adaptive Interpolation of Optimizers with Convergence Guarantees

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Optimizers are central to modern deep learning, yet no single algorithm consistently excels across architectures or datasets. Existing methods of adaptively mixing optimizers to combine complementary strengths are promising, but are restricted to narrow optimizer families or lack rigorous guarantees, leaving a gap between theory and practice. To fill this gap, we present TrailMix, an adaptive interpolation framework that is general across all first- and quasi-second-order methods. On the theoretical front, we prove that convex combinations of optimizers satisfying a mild alignment condition preserve standard convergence rates in non-convex, convex, and strongly convex or PL regimes. For the challenging same-timescale setting, we establish a novel analysis method by lifting the stochastic dynamics to a population-level Fokker-Planck PDE, for which we prove stability using a joint free-energy Lyapunov function. Algorithmically, we extend this framework with fairness normalization, trust-region clipping, and a curvature-awareness reward that stabilizes the meta-weights and enables smoother training. These additions allow TrailMix to behave like an ensemble when optimizers are complementary and to concentrate weight when one dominates, without breaking convexity. Our empirical evaluations on an optimizer set including AdamW, Lion, SOAP, Scion, and MARS show that TrailMix consistently matches or outperforms the strongest single optimizer across a wide range of analytic loss surfaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a new optimizer that adaptively mixes the update rules of several base optimizers and characterize its convergence under certain regularity assumptions. Experimental validation on artificial, simplified landscapes is provided to support the main claims.

### Strengths
1. The analysis is solid and framed under moderately general assumptions.
2. The use of continuous-time modeling (ODEs) to establish convergence is appealing and, as often, insightful.
3. Some experiments are provided to substantiate the theoretical claims.

### Weaknesses
1. **Missing literature review on continuous-time models.**
   The paper lacks a substantive review of recent continuous-time modeling (CTM) work for optimizers, both ODE- and SDE-based: I provide some pointers below, and encourage the authors to explore the (very active) literature to find more recent references in this field.

2. **Limited clarity.**
   Several relevant optimizers (MARS, Scion, SOAP, Lion) are included as baselines in the experiments, but the paper offers only **class-level** guarantees (first-order/adaptive, SPD-preconditioned, inertial) rather than **optimizer-specific** analyses. Please map each optimizer explicitly to the theoretical templates, state the conditions under which it satisfies the alignment assumptions, and if necessary, also the applicability of the KC Theorem.


3. **Questionable practicality and experimental relevance.**
   The experiments focus on nonconvex test functions of limited practical relevance. If the goal is to compete with AdamW and other methods that have been shining and crafted in LLM training, the evaluation should include such realistic workloads. As recognized by the authors, the need for multiple shadow states suggests a significant memory overhead that is likely prohibitive in such relevant settings. Challenging state-of-the-art optimizers only on toy problems is insufficient and misleading; the evaluation should include scenarios where those baselines are known to be strong.

4. **Alignment should be measured in realistic settings.**
   Alignment diagnostics should be reported on large-scale experiments. Otherwise, the method may be operated outside the assumptions that underpin the analysis, risking unstable behavior.

### Questions
1. **On the alignment condition and ODE approximation assumptions.**
   While the analysis is interesting and recovers standard bounds, it is unclear whether the alignment condition truly holds for Adam, or whether the ODE approximation of Adam satisfies the assumptions of the KC theorem. In particular:
   - The manuscript assumes gradients are bounded both above **and** below. Why is a lower bound on gradients needed if convergence is desired? It appears this may be used to ensure the vector field \(g\) in the KC theorem is Lipschitz, which is not the case for Adam without this assumption. But with this assumption, actual convergence may be precluded, which seems contradictory.
   - The paper does not explicitly decompose the Adam increment into a deterministic component plus a martingale term. For reference, [1] derives an SDE model for AdamW that might provide the required decomposition. Otherwise, a new derivation has to be provided.

2. **On related continuous-time literature.**
   Have you considered including a proper literature review of papers using ODEs and SDEs to model optimizers? The paper overlooks a substantial body of prior work analyzing optimization algorithms through continuous-time formulations. This literature provides the theoretical groundwork for interpreting optimizers as dynamical systems and would help position the contribution more clearly. I list several pointers below, and [1] offers a convenient entry point (Related Works and Appendix A) with many additional references. Although [1] emphasizes SDEs, many cited works also contain ODE-based analyses directly relevant here.

---

**[1]** *Adaptive Methods through the Lens of SDEs: Theoretical Insights on the Role of Noise.*
Enea Monzio Compagnoni, Tianlin Liu, Rustem Islamov, Frank Norbert Proske, Antonio Orvieto, Aurélien Lucchi.
*International Conference on Learning Representations (ICLR), 2025.*

---

## Relevant prior work on ODE/SDE analyses of optimization algorithms
1. **Helmke, U. & Moore, J. B. (1994).**  
   *Optimization and Dynamical Systems.* Springer London.  
   — Classical textbook connecting continuous-time dynamical systems and optimization via gradient flows.

2. **Su, W., Boyd, S., & Candès, E. (2014).**  
   *A differential equation for modeling Nesterov’s accelerated gradient method: Theory and insights.*  
   *Advances in Neural Information Processing Systems.*  
   — Foundational ODE model for Nesterov acceleration; initiated the modern line of continuous-time analyses.

3. **Li, Q., Tai, C., & Weinan E. (2017).**  
   *Stochastic modified equations and adaptive stochastic gradient algorithms.*  
   *International Conference on Machine Learning (ICML).*  
   — Derives stochastic modified equations for stochastic gradient algorithms, laying the foundation for weak ODE/SDE approximations.

4. **Li, Q., Tai, C., & Weinan E. (2019).**  
   *Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations.*  
   *Journal of Machine Learning Research, 20(1): 1474–1520.*  
   — Provides a rigorous mathematical foundation for weak SDE approximations of SGD and related methods.

5. **Orvieto, A. & Lucchi, A. (2019).**  
   *Continuous-time models for stochastic optimization algorithms.*  
   *Advances in Neural Information Processing Systems 32.*  
   — Introduces a general ODE/SDE formalism for analyzing SGD, momentum, and adaptive algorithms, establishing links between discrete-time optimizers and continuous-time limits.

### Soundness
3

### Presentation
3

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
This paper presents TRAILMIX, a framework that adaptively combines multiple optimizers during model training. It addresses the lack of theoretical guarantees for such methods by providing a novel "same-timescale" convergence analysis using a Fokker-Planck PDE approach. The framework includes theory-motivated algorithmic improvements like fairness normalization and curvature awareness to create a dynamic, stable, and robust optimizer mixture.

### Strengths
- The paper provides novel convergence guarantees for same-timescale optimizer mixtures.

- The algorithm's components are directly motivated by the theoretical analysis.

### Weaknesses
- Evaluation is mainly on analytic functions; large-scale deep learning experiments are needed to demonstrate real-world utility.

- Potential computational and memory overhead of the proposed methodology is not analyzed.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TrailMix, a framework that tracks the states of several optimizers (update direction, momentum buffer, etc.) during training and forms a convex combination of the update directions. This becomes the update direction of TrailMix. In other words, TrailMix is a dynamic mixture of optimizers. The authors demonstrate that TrailMix can effectively set the learning rate (by tracking several optimizers with different LR) and can converge under parameter shift (when the loss landscape changes abruptly). These experiments are designed to simulate artificially constructed functions. The authors provide convergence guarantees under L-smoothness, bounded direction, and strictly positive gradient alignment assumptions in the full-batch regime. The rates are provided in the non-convex, convex, and PL/strongly convex settings with fixed mixture weights. The authors also provide convergence proofs in the same-time scale setting where mixture weights also evolve in time.

### Strengths
- The convergence proofs look correct to me (yet they are more or less standard), and most of the parts are clearly described.

- The toy examples are interesting and demonstrate desirable characteristics of the proposed framework, when the loss landscape exhibits various trends that make optimization difficult when using only one algorithm.

- Convergence guarantees demonstrate that the proposed framework is expected to converge as long as all optimizers in the set also converge.

### Weaknesses
- Convergence analysis is performed under restrictive assumptions of bounded gradient and gradient alignment. Providing convergence guarantees under a more realistic set of assumptions (for example, a bounded gradient can be replaced by a weak growth condition) is something more preferable as it allows demonstrating convergence for a broader class of functions. I acknowledge that it might be difficult to consider a more general setting, but the paper would benefit significantly from a more sophisticated convergence analysis. 

- Lack of experiments on real problems, like training neural networks, since their loss landscape is less obvious than those presented in the paper.

- Although the authors demonstrate some benefits of such a mixture of optimizers, more practical questions would be more interesting to see in the paper (see the questions section). I encourage providing better ablation studies, even on toy examples, where some practical training details (e.g., learning rate warm-up, clipping, etc) are necessary to converge faster (e.g., on a quartic function, one can show that clipped SGD converges faster than SGD; proving that TrailMix tends to give more weight to clipped SGD would show its practical relevance).

### Questions
- In all experiments, there is at least one optimizer that should work. For example, in the case of parameter shift, SGD is the safest algorithm since it does not use gradient history. However, I am curious to know what happens if all algorithms in the set use gradient evolution. Does TrailMix still perform good enough in such a setting?

- I am also interested in knowing how TrailMix behaves when training language models with a large enough batch size, where we have an Adam-SGD gap. If we train an LLM with TrailMix(Adam, SGD), do we see that Adam receives larger weights most of the time or not? In general, it would be interesting to see how the weights are evolving, and if Adam is always preferable or if there are parts of the training where they both get similar weights. I guess that we can recover some variation of Adedamix optimizer.

- Another interesting question using TrailMix is the study of learning rate scheduling. If we run TrailMix with the set of Adam with different lrs. I am curious if it is possible to recover the warmup and decay stages.

- I suggest adding references to the places where the authors use some well-known facts (e.g., the first part of the proof of Lemma 2; the proof of Lemmas 5 and 6, etc). 

- Can the authors provide more details about step 3 of the proof of Theorem 6?

- I suggest adding the discussion also around grafting of optimizers (Agarwal et al., Disentangling Adaptive Gradient Methods from Learning Rates, 2020), since it's closely related to the idea of mixing optimizers, yet in a different way. 

- How is $\lambda^\prime_{t+1}$ defined in line 10 of Algorithm 1?

- How do $c(\lambda_t)$ appear in the RHS of (8). I believe the RHS of (8) should have only terms independent of $t$.

- Are the authors sure that (178) is always correct? What if the gradient norm is small $\\|\nabla f(x_t)\\| \ll 1$? In this case, the first term in (178) might be significantly larger than the second one. This implies that the alignment might break near the convergence. Could the authors comment on this?

### Soundness
3

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
2

### Summary
The core idea of the TRAILMIX is to compute a weighted average of the update steps by a pool of base optimizers, where the weights are learned and adjusted on the fly. This adaptive optimizer mixtures are interesting, the authors provide the proof to show its first convergence guarantees.

The contributions of this paper are two folds.
a. PROOF. The first convergence guarantees proof of TRAILMIX is done as follows. The stochastic dynamics are lifted to a population-level Fokker-Planck PDE. Stability is then proven using a joint free-energy Lyapunov function, which combines the optimization object with the Shannon entropy of the mixture weights. The proposed TRAILMIX can achieved fairness normalization through keeping the mixture weights within the simplex interior, which is critical for the entropy-based Lyapunov function.

c. EXPERIMENTS. On a range of 2D test functions, TRAILMIX is shown to consistently match or outperform the best single optimizer in its ensemble.

### Strengths
TRAILMIX seems to be original contribution. I appreciate the theoretical analysis. The use of a PDE and a joint free-energy Lyapunov function seems to be a novel approach for analyzing the optimizer convergence.

The proposed work provides a lot of rigorous treatment of the theoretical claims.

This paper is well-written and the core problem is well motiviated.

This paper provides a good theoretical foundation for the field of meta-optimization.

### Weaknesses
This paper has very limited experiments. All experiments are performed on 2D analytic loss surfaces. This leaves the method's practical usefulness for deep learning and broad unproven. It is not very clear how these benefits of the proposed approach actually work in the real-world high-dimensional, non-convex landscapes.

The TRAILMIX has O(K) time and memory overhead, as it has to maintain and update K shadow copies of the base optimizers. For deep learning model training, It is already very bad for a simple optimizer like ADAMW in terms of memory overhead and time overhead.

The ensemble curation is unclear. The paper uses a large hand-picked set of 24 optimizers with varied hyperparameters. It is unknown how sensitive is TRAILMIX to a bad ensemble.

### Questions
1. Can we run experiment on CIFAR-10 with ResNet-18 using the proposed method?  Would the proposed TRAILMIX with K=24 works?

2. TRAILMIX has the hyperparameters, such as lr_meta, how to tune those? Can we find some sensitive experiments to show?

3. How does the TRAILMIX work when base optimizer is bad? e.g. If we just have a bunch of Adam with different learning rates, would that work?

4. It is known that Adam can fail to converge even on simple convex problem, how can we have the assumption 3 hold if the base optimizers themselves may not convergent?

### Soundness
3

### Presentation
3

### Contribution
4
