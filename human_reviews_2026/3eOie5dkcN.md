# Data Fusion–Enhanced Decision Transformer for Stable Cross-Domain Generalization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Cross-domain shifts present a significant challenge for decision transformer (DT) policies. Existing methods typically rely on a single simple filtering criterion to select source trajectory fragments and stitch them together. They match either state structure or action feasibility. However, the selected fragments still have poor stitchability: state structures can misalign, the return-to-go (RTG) becomes incomparable when the reward or horizon changes, and actions may jump at trajectory junctions. As a result, RTG tokens lose continuity, which compromises DT's inference ability. To tackle these challenges, we propose Data Fusion–Enhanced Decision Transformer (DFDT), a compact pipeline that restores stitchability. Particularly, DFDT fuses scarce target data with selectively trusted source fragments via a two-level filter, Maximum Mean Discrepancy (MMD) mismatch for state-structure alignment and Optimal Transport (OT) deviation for action feasibility. It then trains on a feasibility-weighted fusion distribution. Furthermore, DFDT replaces RTG tokens with advantage-conditioned tokens, which improves the continuity of the semantics in the token sequence. It also applies a $Q$-guided regularizer to suppress junction value and action jumps. Theoretically, we provide bounds that tie state value and policy performance gaps to MMD-mismatch and OT-deviation, and show that the bounds tighten as these two measures shrink. We show that DFDT improves return and stability over strong offline RL and sequence-model baselines across gravity, kinematic, and morphology shifts on D4RL-style control tasks, and further corroborate these gains with token-stitching and sequence-semantics stability analyses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DFDT (Data Fusion–Enhanced Decision Transformer), a method that enhances Decision Transformer policies for cross-domain shifts by fusing scarce target data with selectively filtered source trajectories. Using a two-level filter (MMD for state alignment and OT for action feasibility) and replacing return-to-go tokens with advantage-conditioned tokens, DFDT improves stitchability and semantic continuity. The approach demonstrates improved returns and stability across domain shifts in D4RL-style tasks, supported by theoretical performance bounds.

### Strengths
## Strengths

- This paper is well-motivated, and mostly well-written
- This paper is easy to follow and the studied topic is of importance in the context of reinforcement learning community. It is always important to develop more general and stronger transfer algorithms in RL, especially that this paper focuses on the sequence modeling in off-dynamics  RL, which is less explored before
- The main experiments are extensive, including numerous dynamics shift scenarios
- The authors include theoretical analysis to provide better guarantees for the proposed method (despite that some of the theoretical results resemble those in prior works, they are still interesting and bring some insights into the cross-domain reinforcement learning)

### Weaknesses
## Weaknesses

- One vital component in DFDT is the optimal transport term that acts as a data filter. To the best of the reviewer's knowledge, the proposed OT component is the same as that in OTDF (c.f. Eq. 31 and Eq. 32). I think it is okay to use some prior good tricks, but the authors should clearly state their true contributions and the differences against prior works
- Another vital component in DFDT is the MMD part. I am a bit curious about the choice of MMD measurement. Why do you use MMD rather than other metrics? There should be various choices of distance measurement. Also, why not just use another OT? Apart from the OT, do we  really need the MMD part? The authors wrote in Line 237 that *If only one criterion is used (MMD or OT),  the other term remains, highlighting their complementarity*. This can be vague, and the authors should clearly state what MMD controls in the bound and what OT controls.
- In Lines 151, the authors use $s,a,r,s^\prime$ for OT, while OTDF does not use $r$. Can the authors explain the necessity of including the 1-dim reward channel? Also, since the authors use $s,a,r,s^\prime$ for measuring the Wasserstein distance, why do you name it *OT-based Action Credibility*? This should also be related to states. Are there any connections between the effects of MMD and OT here? I think there should be, since they involve almost the same elements.
- What is the insight behind the Q-regularized term? The proposed DFDT method includes many components, while their contributions are not clearly investigated. This paper can benefit greatly by including a detailed ablation study
- Some technical details are presented in the appendix, e.g., the command network. This can be confusing.

I think the above points are comparatively easy to address, and I would be happy to raise my score if the authors can address my concerns and questions

### Questions
I have some additional questions:

- Theorem 3.1 presents a novel performance bound for cross-domain offline RL, but since it deals with the offline scenario, I think there should be some terms like the behavior policy in the source domain, the behavior policy in the target domain, etc. There do not exist such terms in Theorem 3.1, any comments here?
- Line 281, the authors comment that they adopt a multi-step target for Q functions. Do you actually use this trick?
- Apart from the command network and the advantage function estimation, the proposed method should also be applicable to TD-based methods like IQL. Have you ever conducted such experiments? Why do you choose to use DT as the backbone?
- Can you include a detailed parameter study in the main text or the appendix? Currently, it is unclear how sensitive DFDT is to the introduced hyperparameters.

There are numerous minor issues in the main text and the appendix. I just list some of them below. Please double-check your manuscript,
- Line 86, \\$Q\\$-regularized
- Line 220, Assumption B.1?
- Line 313, gravity scales the magnitude of \\$g\\$
- Line 373, *The computing method of normalized returns is described in Sec.*

### Soundness
3

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
3

### Summary
This paper presents data fusion-enhanced decision transformer (DFDT) for cross-domain offline RL to restore token continuity. DFDT introduces components including MMD-based trajectory filtering and OT-based sample reweighting, as long as reweighted advantage conditioning and Q regularizer. The theory bounds claim that the performance gap is bounded by the computed MMD and OT score. Experiments are conducted on MuJoCo-like tasks across various dynamics shifts, which demonstrate DFDT outperforms previous baselines.

### Strengths
1. The idea of incorporating the Decision Transformer into the cross-domain offline setting is interesting; the main components of DFDT, such as two-level data filtering and advantage conditioning, are novel.

2. Experimental results across various datasets and dynamics shift types show that DFDT consistently outperforms the compared baselines.

3. This paper includes detailed descriptions of the motivation and methodology, which help the readers understand the proposed approach.

### Weaknesses
1. The core motivation of the paper is, the cross-domain setting would cause poor stitchability, which is due to misaligned state structures, shifted reward and horizon, and the jumping action at trajectory junctions. However, the main challenge in cross-domain offline setting compared to the pure offline setting is dynamics shift, which is considered by this work and previous works. So I think the poor stichability could be simply explained by dynamics shifts. Please correct me if I am wrong. If so, what is the necessity of using both MMD and OT-based data filtering instead of just using constrastive learning like IGDF or OT like OTDF for dynamics-aware data filtering? If not, what is the difference between the dynamics shift and the claimed factors? Also, what does reward shift mean here? The reward function between source and target domain remains the same, right?

2.  The source fragments are selected and reweighted based on MMD and OT score. It is claimed that MMD measures the state structure similarity  and OT indicates the action credibility. But I still don't understand why the state distribution should be similar between source and target domain. In my view,  as long as the dynamics between source and target transitions stays similar, then there is no need to restrict the state distribution. Moreover, when computing OT score, the vector $v$ is a concatenation of transition $(s,a,r,s^\\prime)$, and the OT score should reflect the distribution distance between the whole transition, why it could represent action feasibility? Also, this design is quite similar to OTDF, and OTDF uses the OT score to represent dynamics shift. So is there any difference between dynamics shift and action feasibility?

3. My major concern on this paper lies in the theoretical interpretations. **Theorem 3.1 cannot support the claim that the performace bound is affected by MMD and OT since the bound is trivial.** Note that a condition $\\Delta _ \\pi$ (which measures the policy difference between learned policy and the optimal policy under target dynamics) is used for derivation, and the derived bound includes the term $\\Delta _ \\pi$ with a scale $\\frac{1}{(1-\\gamma)^2}$. However, this condition is too strong such that we can directly obtain a tighter performance bound given $\\Delta _ \\pi$, which is not relavant to MMD or OT. Specifically, using lemma B.2 in [2] and the fact than $|Q|\\leq \\frac{r _ {max}}{1-\\gamma}$, it is obvious that $|J _ T(\\pi^\\star _ T)-J _ T(\\pi _ {mix})|\\leq\\frac{C}{(1-\\gamma)^2}\\Delta _ \\pi$. Therefore, the performance bound in Theorem 3.1 is too loose to be meaningful, and cannot support the major claim in this paper.

4.  OTDF uses the OT score for data filtering and reweighting. It seems that the only difference of the proposed two-level data filtering and reweighting pipeline is to use MMD score for filtering instead of OT score. Could the authors give more interpretations and experimental results on whether OT-based data filtering is feasible here? 


5. It seems that the whole pipeline of DFDT is quite complex and computing intense, including computing the MMD and OT score, training transformer blocks and Q networks. But no computational cost is discussed and compared in limitation part or appendix.

6. I find a related work RADT [3] which is not discussed in the paper. RADT also addresses the cross-domain setting via decision transformer, with return augmentation techniques. Could you discuss the difference between your work and RADT?

7. (minor) Figure 1 includes the text “OT filtering for a”, which is easy to be misunderstood. The OT does not involve data filtering but only reweighting.

[1] Contrastive Representation for Data Filtering in Cross-Domain Offline Reinforcement Learning. ICML 2024

[2] Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint. ICLR 2025

[3] Return Augmented Decision Transformer for Off-Dynamics Reinforcement Learning.

### Questions
Please see the weaknesses for the concerns. I am inclined to rejection before my concerns are addressed.

### Soundness
2

### Presentation
3

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
The paper proposes Data Fusion–Enhanced Decision Transformer (DFDT) to improve cross-domain generalization of DTs.
DFDT fuses source and target data via MMD-based state alignment and OT-based action feasibility, replaces RTG tokens with advantage-conditioned tokens, and adds a Q-guided regularizer for smoother transitions.
The method provides theoretical performance bounds and shows improved stability and returns on D4RL-style domain-shift benchmarks.

### Strengths
* The two-level fusion filter (MMD + OT) is conceptually clear and theoretically supported.

* Replacing RTG with advantage-conditioned tokens and using a Q-guided regularizer are well-motivated improvements for stability.

* Extensive experiments on diverse domain-shift benchmarks show consistent and meaningful performance gains.

### Weaknesses
* There are some minor typographical issues, e.g., in Line 86 regarding the notation of $Q$.

* The overall presentation and exposition are difficult to follow, making it hard for readers to fully grasp the method and its motivation.

* Evaluation focuses on simulated domains; real-world or larger-scale transfer results would strengthen the paper.

### Questions
* How are the advantage-conditioned tokens computed, and why are they better than RTG tokens for sequence continuity?

* What is the computational cost and sensitivity of the proposed MMD–OT fusion filter in large-scale settings?

* Can DFDT generalize to real-world or partially observed domains, beyond simulated D4RL tasks?

* Could you include comparisons and a discussion with recent cross-domain offline RL studies (e.g., PSEC, DmC)? Incorporating these baselines and analysing the differences would strengthen the paper and better position the proposed method within the current literature.

Reference:

Liu, T., Li, J., Zheng, Y., Niu, H., Lan, Y., Xu, X., Zhan, X. Skill expansion and composition in parameter space. In International Conference on Learning Representations, 2025.

Van, L. L. P., Nguyen, M. H., Kieu, D., Le, H., Tran, H. T., & Gupta, S. DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning. arXiv preprint arXiv:2507.20499.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Data Fusion-Enhanced Decision Transformer (DFDT), a method designed to stabilize Decision Transformer (DT) policies when adapting from a rich source domain dataset to a scarce target domain dataset under dynamics shifts (e.g., gravity, morphology). The core idea is to restore "token-level stitchability" in combined trajectories. DFDT achieves this using a two-level filtering mechanism: Maximum Mean Discrepancy (MMD) for filtering state-structure misalignment, and Optimal Transport (OT) for weighting action feasibility. The method also replaces the brittle Return-to-Go (RTG) tokens with feasibility-weighted advantage-conditioned tokens and adds a Q-guided regularizer. The authors provide theoretical bounds linking performance gaps to the MMD and OT "stitchability radii" and show significant empirical gains and stability improvements over strong offline RL and sequence-modeling baselines across various MuJoCo tasks.

### Strengths
- The paper clearly identifies the main problem—poor stitchability leading to unstable token semantics—and presents a well-structured solution.
- DFDT achieves large and consistent performance improvements over recent, competitive cross-domain baselines (OTDF, IQL, DADT, QT) across three distinct dynamics shifts (morphology, kinematics, gravity).
- The idea of combining MMD (state alignment) and OT (action/transition feasibility) into a single, two-level, token-aware filtering pipeline is novel for cross-domain Decision Transformers. Replacing RTG with weighted advantage tokens is a strong, domain-adaptation-specific design that addresses the incommensurability of returns across shifting environments.

### Weaknesses
- Assumption 3.2 (Approximate fiber-constancy of V) is strong, stating that the value function $V(s)$ is nearly constant over states mapped to the same latent code $f_\phi(s)$. The authors should clarify if/how the training of the state encoder $f_\phi$ is explicitly designed to enforce this value-sufficiency assumption in practice.

### Questions
- Calculating the MMD and OT distances for the entire source dataset is a costly pre-computation step. Can the authors provide a concrete comparison of the total wall-clock time required for this pre-computation phase versus the entire training time (e.g., 100k steps) for a standard DT baseline, especially for the larger D4RL source datasets?
- The OT distance is highly sensitive to the cost function $C$. While the cost is mentioned as "cosine" in the appendix (Table 3), the paper does not justify why this specific 1-Lipschitz cost is optimal for capturing action feasibility shifts compared to simpler or more complex alternatives.

### Soundness
3

### Presentation
4

### Contribution
4
