# Mirror Mean-Field Langevin Dynamics

- Decision: Reject
- Scores: 4, 2, 6, 8

## Abstract
The mean-field Langevin dynamics (MFLD) minimizes an entropy-regularized nonlinear convex functional on the Wasserstein space over $\mathbb{R}^d$, and has gained attention recently as a model for the gradient descent dynamics of interacting particle systems such as infinite-width two-layer neural networks. However, many problems of interest have constrained domains, which are not solved by existing mean-field algorithms due to the global diffusion term. We study the optimization of probability measures constrained to a convex subset of $\mathbb{R}^d$ by proposing the mirror mean-field Langevin dynamics (MMFLD), an extension of MFLD to the mirror Langevin framework. We obtain linear convergence guarantees for the continuous MMFLD via a uniform log-Sobolev inequality, and uniform-in-time propagation of chaos results for its time- and particle-discretized counterpart.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper extends mean-field Langevin dynamics (MFLD) to tackle mean-field optimization problems constrained within a convex subset of $\mathbb{R}^d$. To this end, the authors introduce the mirror mean-field Langevin dynamics (MMFLD), which integrates MFLD into the mirror Langevin framework. They establish linear convergence of the continuous-time MMFLD under a uniform logarithmic Sobolev inequality (LSI) and further prove uniform-in-time propagation-of-chaos results for both its time-discretized and particle-discretized counterparts.

### Strengths
* The constrained mean-field optimization problem is important to the machine learning community due to its broad and interesting applications, and this paper presents an effective algorithmic approach to address it.

* The paper provides a comprehensive review of relevant prior work.

* The MMFLD formulation, along with its convergence and discretization analyses, is largely comparable to those in the unconstrained setting and constitutes a relatively straightforward extension.

### Weaknesses
* The practical applicability of the proposed method remains unclear due to the strong assumptions and the limited, toy-level empirical results.

* The proposed scheme does not constitute a genuine discretization, as it assumes exact simulation of the Brownian motion. Moreover, the analysis largely builds upon existing results from prior works, such as [Ahn & Chewi, 2021] and [Vempala & Wibisono, 2019], and offers limited novelty.

### Questions
* The (mirrored) smoothness assumptions are invoked in several theorems, but the corresponding constants are not explicitly reflected in the statements. For instance, is Assumption 2 required for Theorem 2.3? If so, why do the constants $M_1$ and $M_2$ not appear in the theorem? Similarly, is Assumption 5 necessary for Theorem 3.2? And is it also used in Theorem 4.1?

* I understand that the main contribution of this paper is theoretical. However, prior works on mean-field optimization typically include experiments on training two-layer neural networks. It would be valuable if the authors could provide a similar experiment with constrained parameters, which would greatly enhance the practical relevance and applicability of the paper.

* Regarding the experiment, is $\eta = 3 \times 10^{-3}$ chosen as the optimal stepsize for both methods? Different algorithms may exhibit different sensitivities to the stepsize, so using the same value without tuning may raise concerns about fairness in comparison. In addition, since this is an $N$-particle algorithm, it would be helpful to evaluate its sensitivity to the number of particles—for example, by testing $N \in \{256, 512, 1024, 2048, ...\}.

**I will be happy to raise my score if the authors can address my concerns on the assumptions and experiments.**

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies an extension of mean-field Langevin dynamics to constrained settings using ideas from mirror Langevin dynamics. They obtain linear convergence of continuous time dynamics under a uniform log-Sobolev inequality, and also give a propagation of chaos result to give a result that is discrete in space and time. A numerical experiment on the simplex is given that shows some slight advantage to the mirror mean-field Langevin dynamics.

### Strengths
- The paper is generally well written and easy to understand. The theorem statements are generally clear to me and the proofs are readable and easy to follow.
- This paper represents a natural extension of mean-field Langevin dynamics using a mirror map. As was done in the non-mean-field case, this extension has desirable properties of relying on relative Lipschitz types of assumptions, and also naturally maintains the constraints of the problem.
- The results utilize the full range of available tools to prove a convergence bound for a particle distribution in discrete time.

### Weaknesses
- The theorems given seem to be standard extensions of existing results in the literature. The heavy lifting seems to have been done in past works like Nitanda et al '22, Nitanda '24, and Nitanda et al '25. Because of this, I am worried that this work is more of a synthesis work than giving some novel and new ideas that would be sufficient for publication in ICLR.
- Coupled with the above limited theoretical novelty, there is a lack of experimental evidence. The authors only give one low-dimensional experiment that barely shows an advantage for MMFLD.
- There is little attention paid to a motivating example. Why should the reader care about constrained sampling? Grounding this problem in real problems of interest to the machine learning community would greatly strengthen the position of this paper.

### Questions
- For the continuous setting, in Assumption 4 the authors assume a uniform LSI. It would be useful if the authors could provide a discussion of the cases when this holds rather than offloading this to references. The same comment goes for Assumptions 6 and 8.
- The modified Wasserstein distance is not symmetric in $\mu, \mu'$? Does this have any connnection to a Bregman Wasserstein divergence, or some other "distance" of interest?
- The constant in Theorem 4.2 is exponential in $D$, and for common barrier mirror maps, $\nabla \phi$ is surjective. Doesn't this mean that $D$ is unbounded? The authors should comment on this.
- Where do I see the self concordance parameter $\gamma_1$ in the statement of Theorem 4.2?
- Can the authors comment on their results in comparison with the analysis of projected Langevin (Bubeck et al '15), projected SGLD (Lamperski '21), and other recent methods for sampling from convex bodies (like Gu et al '24)? Having a more in depth theoretical comparison of the advantages of a mirror approach in this setting would be useful, and could also help to guide experimental settings to show where this method has a true advantage.

References:
Bubeck, Sebastien, Ronen Eldan, and Joseph Lehec. "Finite-time analysis of projected Langevin Monte Carlo." Advances in Neural Information Processing Systems 28 (2015).
Lamperski, Andrew. "Projected stochastic gradient langevin algorithms for constrained sampling and non-convex learning." Conference on Learning Theory. PMLR, 2021.
Gu, Yuzhou, et al. "Log-concave sampling from a convex body with a barrier: a robust and unified dikin walk." Advances in Neural Information Processing Systems 37 (2024): 69230-69298.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors consider a combination of mirror Langevin dynamics and mean-field Langevin dynamics, analysing the setting of entropy-regularised functionals on constrained convex domains. They provide convergence guarantees for the continuous-time flow under logarithmic Sobolev inequalities and develop guarantees for a time- and particle-discretized scheme. They also provide experiments comparing the scheme to projected mean-field Langevin dynamics, showing that their scheme attains a lower final loss.

### Strengths
* The authors target a fairly important and surprisingly open problem, since mean-field dynamics are used to understand two-layer neural networks and mirror descent is frequently used in constrained optimisation.
* The paper is very well-written.
* The guarantees are strong and are under relatively standard conditions in this area (e.g., uniform LSI).

### Weaknesses
* The majority of the proof techniques appear to be borrowed or adapted from other papers (e.g., Nitanda et al., 2022; Jiang, 2021 and Nitanda, 2024), so the work may have limited technical novelty at the proof level.
* The analysis of the discretized algorithm assumes that the pure diffusion step (Algorithm 1, step 5) can be **simulated exactly**. The authors note this is for "simplicity of exposition", but this is rarely possible in practice and creates a gap between the theory and the implementation (which used a one-step discretization).
* The discrete-time convergence analysis (Section 4.3) is presented for the specific setting of the mean-field neural network risk minimization problem, which may limit the perceived generality of the result.

### Questions
* What are the primary technical novelties of this work at the level of the proof, beyond the synthesis of existing analytical frameworks?
* Regarding Weakness 2: Can the authors comment on the error introduced by *not* simulating the diffusion step exactly? Would a practical, one-step discretization of this diffusion term impact the final convergence guarantee in Theorem 4.2?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces and studies mirror mean-field Langevin dynamics, which can be used to solve constrained distributional optimization problems. The authors establish the exponential convergence of this dynamics under a (mirror) log-Sobolev inequality akin to the Euclidean counterpart, and also carry out a complete discretization analysis on both time and the number of particles.

### Strengths
This paper provides a clean and novel analysis of mirror mean-field Langevin dynamics as a generalization of mean-field Langevin, and the numerical illustration demonstrates that this can be a better idea to solve distributional optimization problems compared to projected mean-field Langevin. This is interesting in particular since currently there are not many well-studied algorithms for distributional optimization that work well in high dimensions beyond the mean-field Langevin dynamics.

### Weaknesses
* I think the authors can better motivate the study of mirror mean-field Langevin by showing what new settings can be unlocked by their analysis, e.g. for training weight-constrained two-layer neural networks or for generative modeling.

* Since the discretization cost of Step 5 is not analyzed, it could potentially be helpful to have, perhaps an informal, discussion of why simulating this step is easier than simulating a Brownian motion on $\mathcal{X}$ (and consequently performing MFLD on $\mathcal{X}$).

### Questions
A typical framework for the theoretical study of two-layer nets is to constrain the first layer weights to live on the unit sphere, and allow the second layer weights to be unbounded. A challenge here is that one can not show uniform LSI due to the unbounded weights of the second layer. One way to remedy the issue is to perform bilevel optimization as in [1], which reduces the problem to MFLD on the unit sphere with a bounded uniform-LSI constant. However, [1] still requires performing MFLD on the unit sphere, without discretization guarantees. A concrete application of the results of this paper can be to instead use mirror MFLD after the bilevel reduction, to provide an end-to-end guarantee for training two-layer networks in the mean-field regime.

[1] G. Wang et al, "Mean-Field Langevin Dynamics for Signed Measures via a Bilevel Approach." NeurIPS 2024.

### Soundness
3

### Presentation
3

### Contribution
3
