# Adaptive Canonicalization with Application to Invariant Anisotropic Geometric Networks

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Canonicalization is a widely used strategy in equivariant machine learning, enforcing symmetry in neural networks by mapping each input to a standard form. 
Yet, it often introduces discontinuities that can affect stability during training, limit generalization, and complicate universal approximation theorems. 
In this paper, we address this by introducing *adaptive canonicalization*, a general framework in which the canonicalization depends both on the input and the network.
Specifically, we present the adaptive canonicalization based on prior maximization, where the standard form of the input is chosen to maximize the predictive confidence of the network.
We prove that this construction yields continuous and symmetry-respecting models that admit universal approximation properties. 

We propose two applications of our setting: (i) resolving eigenbasis ambiguities in spectral graph neural networks, and (ii) handling rotational symmetries in point clouds. 
We empirically validate our methods on molecular and protein classification, as well as point cloud classification tasks. Our adaptive canonicalization outperforms the three other common solutions to equivariant machine learning: data augmentation, standard canonicalization, and equivariant architectures.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes *adaptive canonicalization*, a form of canonicalization that depends on the input and network to alleviate concerns about continuity (i.e., slight changes in the input do not change the canonical form) and about universal approximators. The paper also shows how to use this canonicalization for molecular, protein, and point cloud classification tasks.

### Strengths
- The paper is well-structured, and the motivation is appropriately highlighted.
- The paper constructively develops the theoretical background to prove the claims.
- The experiments consistently show improved performance of using adaptive canonicalization for wide-variety of tasks.

### Weaknesses
I have a few minor questions, which are highlighted in the next section.

### Questions
- No equivariant tasks (i.e., segmentation) were studied within the paper, which is one of the more relevant tasks where such a method should be evaluated.
- One of the primary benefits of canonicalization-based methods is the adaptation of large pre-trained models in a zero-shot manner to avoid significant re-training cost. However, the experimental designs proposed in the paper require re-design and re-training of the downstream network. It would be great if the authors could add a discussion of this in the paper, along with potential ways to use adaptive canonicalization with pre-trained models.
- Could you discuss and contrast your proposed method with the following lines of work [1, 2]?
- How does the distribution of canonical samples look? Does it retain some structure or orientation (or specific rotations, e.g., for images, say for a horse, we get for humans, as described, with legs down), or is there no such orientation once the canonicalization is *adaptive*?
- Can you provide the average over multiple runs in Table 3? 
- Computational requirements are not discussed, i.e., what was the training budget (time and compute requirements) for A-NLSF vs other methods? It would be unfair if the proposed method requires a significantly higher training budget.
- How do the theoretical results in this paper compare with the results of [3]?

1. Equi-Tuning: Group Equivariant Fine-Tuning of Pretrained Models. Basu et al., AAAI 2023.
2. Efficient Equivariant Transfer Learning from Pretrained Models. Basu et al., NeurIPS 2023.
3. On the Universality of Invariant Networks. Maron et al., ICML 2019.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Canonicalization is widely used in equivariant network but it introduces discontinuities that cause many problems. This paper proposes an adaptive canonicalization via prior maximisation to address this problem. The paper also presents applications of the proposal method. The authors theoretically and empirically verify their results.

### Strengths
The authors aim to address a key problem in equivariant neural networks: that canonicalization can cause problematic discontinuities. They propose a new, interesting, and well-motivated method. They present solid theory to support their claims.

### Weaknesses
My concern is mostly about the experiments. 

The performance improvement in experiments on OGB looks marginal. For the experiments on OGB, gains are only 0.0114 in ogbg-molhiv, 0.0061 in ogbg-molpcba, 0.0134 in ogbg-ppa, while the standard deviations in ogbg-molhiv are 0.0132/0.0152. The standard deviations on ModelNet40 are not reported.

No ablation study is presented. For anisotropic nonlinear spectral filters, I quoted, "our formulation partitions the spectrum into disjoint frequency bands," but no experiments were conducted on how to partition the spectrum. Also, "we consider the GSO as the normalized graph Laplacian," and uses an MLP, without rationale or empirical validation. For anisotropic point cloud networks, the authors use backbones DeepSet and DGCNN, but not validate the choice. 

The experiemtns rely on very many hyperparameters but didn't report their sensitivity. In Appendix F, the authors mentioned graid size, sinusoidal period, noise level, hidden dimension, etc.

Experiments on more datasets are encouraged, such as ShapeNet.

The code is not submitted.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces new results on the continuity of canonicalization functions used for architecture agnostic equivariance. Based on this analysis, it proposes an optimization based method for canonicalization that respects continuity. The method is validated experimentally on spectral graph processing and point cloud classification.

### Strengths
- Understanding the continuity of canonicalization methods is an interesting problem given that these methods are becoming more widely used
- The theoretical results are sound and rigorous as far as I could check
- The experimental performance of the proposed method on the tasks presented is competitive
- The detailed appendix is appreciated

### Weaknesses
- I think generally the paper should do a better job at discussing related works beyond the introduction and appendix
    - In many canonicalization frameworks, the canonicalization is learned and already depends implicitly on the prediction network, since it is trained using gradients from the prediction network. We can define the combination of the canonicalization architecture, initialization, and optimization process as the "adaptive canonicalization", which then becomes a function of the prediction function. In that sense, I don't think "adaptive canonicalization" is really a novel concept.
    - Likewise "prior maximization adaptive canonicalization" is closely related to the optimization canonicalization method introduced in Kaba et al. 2023. Similar ideas have also been used and proposed in subsequent papers such as Schmidt & Stober, 2024, Shumaylov et al. 2024, Singhal et al. 2025. I think it is necessary to acknowledge these works better and not overstate the novelty of these ideas
- The framework is developed with a high degree of generality, which does not refer to group actions until section 2.4. However, in the end the only applications considered are really only group actions. I feel like the analysis should either be simplified to consider the specific case of group actions (with possibly the more general version in appendix) or applications outside group actions should be examined. In the current state, the amount of formalism seems overkill given the problem considered.
- I think the construction for continuous canonicalization introduced in 2.3 (proved in theorem 9) is closely related to the weighted frames of Dym et al. 2024. Can the authors comment on that and discuss connections in that section of the paper?
  - If the proposed canonicalization scheme is not a type of weighted frame, it would be great if the authors could better highlight the core ingredient that makes the scheme continuous.
- More generally, I would appreciate if the authors could discuss their theoretical results more and try to distill them into insightful conclusions
- I think section 3 on applications should be significantly rewritten to improve ease of understanding and reduce the amount of mathematical formalism. In its current state, this section that introduces applications will be quite difficult to understand for practitioners.
- Minor corrections
    - Typos:
        - line 137: "vanish at infinity is" - > "vanish at infinity if"

### Questions
- An important part of the implementation that was not clear to me is how the optimization is performed for each group. Solving these optimization problems is expected to be costly in general and to introduce significant overhead. This should be discussed more transparently
- Likewise, I didn't understand which priors are maximized in practice. Can the authors elaborate and clarify that in the main text.

### Soundness
3

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
The paper introduces adaptive canonicalization, where the canonical map depends on both the function/network and the input: a family $\rho_f : G \to K$ such that models have the form $g \mapsto f(\rho_f(g))$. Under an equicontinuity condition on ${f \circ \rho_f(g)}$, the authors prove a universal approximation theorem: if a base class $\mathcal N$ is universal on $C_0(K,\mathbb R^D)$, then ${\theta \circ \rho_\theta : \theta \in \mathcal N}$ is universal for adaptive canonicalized functions. They instantiate this via prior maximization, where a transformation family $\kappa_u$ and monotone priors $H=(h_d)$ are used to pick a canonical representative by maximizing $h_d(f_d(\kappa_u(g)))$ over $u$, making $\rho_f$ depend on $f$. They show prior maximization yields continuous, symmetry-preserving models and, under a symmetry-preserving construction, that $f \circ \rho_f$ can represent all continuous symmetry-preserving functions, again with universal approximation guarantees. Two applications are given: (i) A-NLSF spectral GNNs that resolve eigenbasis ambiguities and allow anisotropic filters, and (ii) adaptive canonicalization for point clouds (AC-PointNet / AC-DGCNN) handling SO(3) while retaining directional information. Experiments on graph benchmarks (TUDataset, OGB) and ModelNet40 show consistent improvements over data augmentation, standard canonicalization, and equivariant architectures.

### Strengths
1. **Elegant and general theory.** Adaptive canonicalization and prior maximization are simple to state yet powerful, giving continuity and universal approximation while avoiding classical discontinuity pitfalls.

2. **Strong link to symmetry theory.** The symmetry-preserving construction and characterization of all continuous symmetry-preserving functions via $f\circ\rho_f$ provide a satisfying representation-theoretic story.

3. **Practical instantiations.** A-NLSF neatly resolves spectral eigenbasis ambiguities and yields anisotropic filters; AC-PointNet/AC-DGCNN give rotation-invariant yet directionally aware point-cloud models, with solid improvements over data augmentation, standard canonicalization, and VN-based equivariants.

4. **Good positioning in current literature.** The work is clearly framed against discontinuity results for canonicalization and complements other directions like probabilistic frames and probabilistic symmetry breaking.

### Weaknesses
1. **Approximate maximization not analyzed.** The theory assumes exact $\arg\max_u$ for prior maximization; in practice one uses discretization or iterative search. The impact of $\delta$-suboptimal maximizers on continuity and approximation guarantees is not quantified.

2. **Focus on classification tasks.** The prior design and applications are tailored to one-vs-rest classification; how to extend to regression or more structured outputs is left open.

3. **Limited variety of groups in experiments.** The empirical section covers spectral band symmetries and SO(3) for point clouds; it would be interesting to see at least one example with more complex/non-compact groups (e.g., SE(3) in continuous domains), though this is not required for acceptance.

### Questions
1. In practice you maximize $h_d(f_d(\kappa_u(g)))$ approximately. If the chosen $u$ is within $\delta$ of the true maximum, can you bound the resulting deviation in $f\circ\rho_f$ or in the universal approximation error (e.g., as an additive $\mathcal O(\delta)$ term)?

2. What specific $h_d$ do you use (logit, probability, margin)? Do different monotone choices substantially change performance or optimization stability, and do any non-monotone choices appear useful despite not fitting the theory?

3. Many applications involve non-compact groups (e.g., SE(3)). Under what additional conditions (e.g., priors that decay at infinity, restricted parameter ranges) do you expect the prior maximization construction to remain continuous and symmetry preserving?

4. Since $\rho_f$ depends on the evolving network $f$, did you observe any instabilities (e.g., rapid switching of canonical representatives) during training? Have you tried freezing $\rho_f$ after some epochs and training only the classifier/regressor, and if so, how does that affect accuracy?

5. Do you foresee obstacles to applying adaptive canonicalization to other domains (e.g., SE(2) image rotations/scalings, sequence symmetries), or is it mostly an implementation effort at that point?

### Soundness
3

### Presentation
4

### Contribution
3
