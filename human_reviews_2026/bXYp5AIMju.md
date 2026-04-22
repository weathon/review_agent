# Learning Exposure Mapping Functions for Inferring Heterogeneous Peer Effects

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Peer effect refers to the difference in counterfactual outcomes for a unit resulting from different levels of peer exposure, the extent to which the unit is exposed to the treatments, actions, or behaviors of its peers. In practice, peer exposure is typically captured through an explicitly defined exposure mapping function that aggregates peer treatments and outputs peer exposure. Exposure mapping functions range from simple functions like the number or fraction of treated friends to more sophisticated functions that allow for different peers to exert different degrees of influence. However, the true function is rarely known in practice and when the function is misspecified, this leads to biased causal effect estimation. To address this problem, the focus of our work is to move away from the need to explicitly define an exposure mapping function and instead introduce a framework that allows learning this function automatically. We develop EGONETGNN, a graph neural network (GNN), for heterogeneous peer effect estimation that automatically learns the appropriate exposure mapping function and allows for complex peer exposure mechanisms that involve not only peer treatments but also attributes of the local neighborhood, including node, edge, and structural attributes. We theoretically and empirically show that GNN models that use peer exposure based on the number or fraction of treated peers or learn peer exposure naively face difficulty accounting for such influence mechanisms. Our evaluation on synthetic and semi-synthetic network data shows that our method is more robust to different unknown underlying influence mechanisms when compared to state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new, custom GNN architecture, EgoNetGNN, for the network interference problem. The idea is to learn the underlying exposure mapping behind the network interference.  EgoNetGNN is more expressive than standard GNN architectures that have been used for this task in previous works, allowing EgoNetGNN to exhibit superior counterfactual estimation performance compared to such competing baselines, which is validated on synthetic and semi-synthetic interference settings.

### Strengths
Developing a novel GNN architecture that can automatically learn exposure mappings for heterogeneous peer effect estimation is a significant contribution over using out of the box GNNs. The authors also motivate this novel architecture by showing, theoretically, a gap in expressivity. The definitions and claims around these results are clearly stated. The architecture of EgoNetGNN is clearly stated as well as the training procedure. Within the given experimental settings, EgoNetGNN exhibits generally superior performance to the baseline methods, and in scenarios where that it is not the case it is explained that the baseline method is able to express the given specific exposure, while a strength of EgoNetGNN lies in expressiveness across each of eg. mutual connections, clustering coefficient, attribute similarity. This reflects EgoNetGNN being able to learn the appropriate exposure mapping in general while baseline methods may be more limited. Experimental details are presented well and ablation studies are included, strengthening the empirical results.

### Weaknesses
The weaknesses of the paper do not rest in whether it adequately addresses the scope that it claims to cover, but rather the significance of the scope and setting compared to previous baselines. In particular, at least as written, it felt that exposure mechanisms were chosen not because of practical relevance but because this is where the improvement compared to baselines emerges.

Even beyond the scope of GNN based approaches, as a method for estimating counterfactuals under network interference, EgoNetGNN relies on the, at least in some settings, strong assumption of known network structure and features. While reasonable for some setting such as perhaps a social media platform, this limits the scope of the work to the broader network interference problem.

### Questions
Could the authors provide further significance for the practical relevance of exposure mechanisms that they demonstrate superior performance over? For example, empirical works motivating the given exposure mechanism, etc. 

Would it be possible to compare the performance of EgoNetGNN to the baselines under exposure mechanisms that more of the baselines actually are expressive enough to handle? Basically if practitioners were to use your method not knowing wha the true underlying exposure mapping should be, and it ends up being one that other methods could also express, what kind of relative performance might we expect?

If a practitioner wanted to EgoNetGNN but had missing graph information, would there be any recommended modifications or imputation methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses an important and underexplored problem: *learning exposure mapping functions automatically* in causal inference under interference.
Its motivation is strong, and the proposed framework is technically original and relevant to both the causal inference and graph learning communities.
The theoretical ambition (Sec. 3.3) and experimental comprehensiveness (Sec. 4) are commendable.

However, under **top-tier conference standards**, several aspects are insufficiently rigorous:

1. The causal identification relies on strong, untested assumptions.
2. Theoretical guarantees (e.g., consistency, identifiability) are not formally proven.
3. Experimental evidence conflates representational expressiveness with causal validity.
4. Interpretability and computational scalability remain weakly supported.

### Strengths
* **Novel problem definition:**
  The paper formalizes *exposure mapping function learning* as a distinct subproblem, bridging a gap in causal inference with interference (Sec. 1–2).

* **Architectural innovation:**
  EGONETGNN combines ego-network extraction, masked weighting, and exposure encoding, with coverage-based losses designed for invariant and expressive exposure learning (Sec. 3.1–3.2).

* **Comprehensive experiments:**
  The study spans multiple network types and exposure mechanisms, demonstrating robustness across topologies and influence structures.

* **Theoretical ambition:**
  Expressiveness analysis (Sec. 3.3; Appendix A.4) connects to prior results on GNN motif-counting power (Chen et al., 2020) and causal motifs (Yuan et al., 2021).

* **Transparent ablations:**
  Ablation studies (Table 2) clearly show that the feature encoder and masked weights contribute meaningfully to performance.

* **Reproducibility commitment:**
  Code release and detailed configuration documentation (Appendix A.6) facilitate replication.

### Weaknesses
1. **Strong and untested causal assumptions.**
   The identification (Prop. 1) depends on *unconfoundedness* and *neighborhood interference* (Assumptions 2–3). These are unrealistic in most observational networks and are not empirically validated.

2. **Lack of formal theoretical guarantees.**
   The expressiveness argument shows motif-counting capability but does not establish *consistency* or *generalization bounds* for ( \hat{\phi}_e ) or ( \hat{f}_y ). No identifiability result ensures that learned exposures converge to true exposures.

3. **Unclear causal vs. representational validity.**
   Reduced εPEHE may result from representation fitting rather than correct causal identification. There are no counterfactual or ablation diagnostics that isolate causal correctness.

4. **Limited interpretability.**
   The learned exposure embeddings ( \rho ) lack transparency. Table 3 reports correlations but provides no qualitative insight into which node/edge structures the model captures.

5. **Scalability not demonstrated.**
   EGONETGNN’s per-node ego-network extraction and masked MLP operations increase complexity by roughly ( \rho_E \times \text{avg}(d) ) over MPGNNs (Sec. 3.3), yet no runtime or memory benchmarks are reported.

6. **Statistical reliability and variance.**
   The paper reports mean εPEHE but omits confidence intervals, variance decomposition, or sensitivity to hyperparameters ( \lambda_{\text{cov}}, \lambda_{\text{ent}}, \lambda_{\text{sp}} ).

7. **Incomplete comparison to modern baselines.**
   Proximal and double-negative-control causal estimators (Tchetgen et al., 2020; Miao et al., 2024) are cited but not evaluated. Transformer-based causal GNNs are missing.

8. **Potential endogeneity in joint learning.**
   Because ( \phi_e ) is optimized via outcome loss ( y ), exposure representations may absorb post-treatment bias. No regularization or identification constraint ensures causal directionality.

### Questions
1. **On causal identification and assumptions**

   * You rely on *unconfoundedness* and *neighborhood interference* (Assumptions 2–3) for identifiability.
     Could you clarify **how these assumptions are justified or tested** in your semi-synthetic datasets?
     For instance, are confounders explicitly simulated and controlled?
     If not, how do you ensure that violation of these assumptions does not bias the estimated heterogeneous peer effects (HPE)?
   * Have you considered or tested *partial interference* (interference beyond immediate neighbors)? If so, could you show how EGONETGNN’s performance changes under that violation?

2. **On the learned exposure mapping function ( \hat{\phi}_e )**

   * Can you provide **interpretability or visualization** of the learned peer exposure embeddings ( \rho )?
     For example, do certain dimensions correspond to structural motifs (e.g., triangles, degree centrality, clustering coefficients) or node attributes?
   * How do you ensure that ( \hat{\phi}_e ) captures *causal influence* rather than merely *correlated structural patterns*?
     Have you tried using random edge rewiring or attribute permutation tests to probe causal validity?

3. **On theoretical soundness**

   * The paper claims improved expressiveness compared to standard MPGNNs (Sec. 3.3). Could you provide **formal statements or empirical ablations** verifying that EGONETGNN can indeed represent specific causal network motifs (e.g., closed triads) that GCN/GIN cannot?
   * Is there any **consistency guarantee** that the learned exposure mapping converges to the true one as sample size grows? If not, can you at least discuss conditions under which your end-to-end loss (Eq. 11) provides an unbiased estimator of the HPE?

4. **On training stability and loss design**

   * The overall loss combines several terms (Eq. 11). How sensitive are results to the hyperparameters ( \lambda_{\text{cov}}, \lambda_{\text{ent}}, \lambda_{\text{sp}} )?
     Could you include a sensitivity plot or table showing their impact on εPEHE or embedding coverage?
   * How does the *coverage loss* quantitatively affect the learned exposure distribution—does it prevent mode collapse or just rescale embeddings?

### Soundness
2

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
4

### Summary
This paper tackles the problem of estimating heterogeneous peer effects under network interference when the true exposure mapping function is unknown or misspecified.
The authors propose EGONETGNN, a graph neural network framework that learns the exposure mapping function automatically from ego-net structures and contextual features.
Comprehensive experiments on synthetic and semi-synthetic datasets demonstrate the effectiveness of the proposed method.

### Strengths
1. The experiments are comprehensive.
2. The paper provides the source code.

### Weaknesses
1. The framework still relies on the unconfoundedness and neighborhood-interference assumptions (Assumptions 2–3), which may limit its causal interpretability in real-world settings. If these assumptions are violated, the model’s learned exposure representations and effect estimates could become biased, as the identification no longer holds.

2. The abstract and introduction emphasize theory as part of the contribution, but the main text does not explicitly present any novel theoretical results.
3. Though scalability is discussed (Appendix A.4), the runtime grows with ego-network size, and large-graph applicability remains an open concern.
4. The paper only validates the model on synthetic and semi-synthetic datasets, without real-world intervention data.
5. Appendix A.4 provides runtime measurements only for EGONETGNN itself but does not include baseline comparisons. A relative runtime or scalability comparison with the baseline methods would make the analysis more complete.

### Questions
1. What would happen if the unconfoundedness or neighborhood-interference assumptions (Assumptions 2–3) are violated? In particular, the neighborhood-interference assumption restricts interference to first-order neighbors, which may not hold in real-world networks. How sensitive is the method to such violations?

2. The model is computationally about ρε​×avg(d) times more expensive than a standard MPGNN, which may limit its scalability to large networks. 
Regarding scalability, the authors mention that sampling or parallelization could alleviate the computational burden. Could the paper provide quantitative comparisons of runtime and memory usage?

3. How does EGONETGNN perform in noisy networks or when some node features are missing? Does it still outperform the baselines under such imperfect conditions?

4. What are the key differences and advantages of EGONETGNN compared to recent methods such as AEMNet (ICASSP 2025) and CauGramer (ICLR 2025)? Could the framework potentially be extended to integrate attention or transformer-based mechanisms?

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
This paper proposes a new method for estimating peer effects in the presence of interference. Instead of relying on assumptions about the exposure mapping function, their method is able to learn this function from data. The authors show the performance of their method on a variety of network datasets.

### Strengths
- The authors propose a novel method for an important and interesting problem.
- The authors extensively validate their method on synthetic data and a wide variety of different exposure mapping functions

### Weaknesses
In Section 3.2 the authors explain TARNet and CFR and how they apply balancing to their objective. As I understand it, in the original paper [1] the architecture of these two methods is the same; the difference is that CFR uses representation balancing with an IPM metric. In section 3.2, however, it seems like these two methods have a different architecture. It would be helpful if the authors could clarify this, and maybe also change the confusing naming of their two variants.

Related to this, the authors write that IPM balances the distribution $P(c,p|t)$. However, I am not sure I understand why this would be desirable.  In CFR, the representations are being balanced with respect to the treatment of interest. Here, given that the goal is to estimate peer effects, the representations are being balanced to a variable that is not the exposure of interest (which would be $\rho$). Could the authors give some more explanation as to why balancing with respect to the individual treatment would give better peer effect estimates?

It is unclear how the authors select the hyperparameters, especially the loss weights $\lambda$. Hyperparameter selection is a very important problem in treatment effect estimation because you cannot optimize them based on the loss of interest (which contains unobservable counterfactuals) [2]. Given that hyperparameter selection can have a very large impact on results, I would like the authors to discuss this in the paper.

On line 361 the authors write that they report PEHE with respect to peer effect estimation for 5 different simulations. What exactly is meant by simulations in this case? Does this mean that the data is always different in each simulation? Would it make more sense to use the same DGP, but with different random initializations of the model?

[1] Shalit, U., Johansson, F. D., & Sontag, D. (2017, July). Estimating individual treatment effect: generalization bounds and algorithms. In International conference on machine learning (pp. 3076-3085). PMLR.

[2] Curth, A., & Van Der Schaar, M. (2023, July). In search of insights, not magic bullets: Towards demystification of the model selection dilemma in heterogeneous treatment effect estimation. In International conference on machine learning (pp. 6623-6642). PMLR.

### Questions
Why do you only evaluate based on peer effect estimation performance? If I understand the method correctly, I would think that you could also learn direct and total effects?

On line 260-261: "Masked weights promotes representation that is invariant to irrelevant contexts ..." Why do masked weights promote this?

Could you explain why the two aggregations for $\rho$ are enough and relevant for all different exposure mapping functions? Are there any exposure mappings that the method cannot learn?

How well does your method perform compared to the baselines when the baselines make the right assumption about the exposure mapping function? I think this would provide additional insights into the performance of the different methods.

### Soundness
2

### Presentation
2

### Contribution
2
