# Latent Adaptation of Foundation Policies for Sim-to-Real Transfer

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
The sim-to-real problem remains a critical challenge in the real-world application of reinforcement learning (RL). The conventional sim-to-real methods heavily rely on resource-intensive re-training of the policy network to adapt to new domains, which limits the flexibility of the deployment of RL policies in ever-changing environments. Inspired by human locomotion, where individuals adjust their gait to new surface conditions without relearning the skill of walking, we introduce Latent Adaptation of Foundation Policies (Found-adapt), a framework that decouples this problem into skill acquisition and environment adaptation. Our method first pretrains a foundation policy on unlabeled offline trajectories from the source simulator, capturing diverse long-horizon behaviors as reusable skills. At deployment, instead of retraining the policy, we perform efficient latent space adaptation: a small amount of target-domain data is collected to refine a latent representation through an adapter network that incorporates parameter efficient alignment, which produces a task-ready controller under various system dynamics. This adaptation occurs entirely in the latent space, avoiding costly policy optimization while enabling robust transfer. Empirical results across multiple locomotion tasks and dynamic variations demonstrate that our method significantly reduces the sim-to-real gap. Further sensitivity analysis provides interesting insights into the requirements for data quality and applicable situations. These findings highlight how foundation policies with latent adaptation could serve as a general and efficient paradigm for real-world RL deployment.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces Found-adapt, a framework for efficient sim-to-real transfer in reinforcement learning that adapts pretrained foundation policies through latent-space alignment rather than policy retraining. The method first trains a foundation policy on offline simulator trajectories in a Hilbert-space embedding that encodes reusable behavioral primitives. At deployment, adaptation is performed in three lightweight stages: a cross-domain least-squares latent alignment that integrates simulator and limited real data, a MetaDynamic network that extracts a permutation-invariant dynamics signature from the target domain, and a signature-guided adapter that refines the latent vector to align behavior with new dynamics. This parameter-efficient approach enables adaptation without modifying policy weights. Extensive experiments on locomotion tasks (Walker, Cheetah, Quadruped, AntMaze, Kitchen) under varying gravity and friction demonstrate that Found-adapt substantially narrows the sim-to-real performance gap compared to strong baselines (GAT, UGAT, PAD), achieving similar performance with orders-of-magnitude lower adaptation time. Detailed ablations, sensitivity analyses on data corruption, and studies linking adaptation loss to performance further support the framework’s robustness and practical potential for real-world RL deployment.

### Strengths
1. The paper reframes sim-to-real transfer as latent alignment rather than policy re-optimization. This perspective—adapting a frozen foundation policy by operating in its latent space—is conceptually fresh and could generalize across domains and tasks. The Hilbert-space foundation policy formulation (Sec. 4.1) gives a principled way to tie skill structure to temporal dynamics, aligning well with recent trends in foundation models for control.

2. The method is well-structured and mathematically grounded. Equations (1–6) and Figure 2 outline a coherent pipeline linking the least-squares latent solution, MetaDynamic signature, and adapter update. The combination of permutation-invariant dynamics encoding and parameter-efficient adaptation is elegant and interpretable.

3. Experiments span multiple environments and perturbations (gravity G1–G4, friction F1–F6), with comparisons to both task-specific (PAD) and dynamics-specific (GAT/UGAT) baselines (Table 1, Figures 3–9). Found-adapt consistently ranks among the top methods, often within 5-10% of PAD but with orders-of-magnitude faster adaptation (seconds vs. hours).

4. The ablation study (Figure 4a) carefully isolates the contribution of the initial alignment, dynamics signature, and adapter. The sensitivity analyses on data corruption (Figures 10–11) and the strong correlation between adaptation loss and performance improvement (Figure 6) demonstrate a rare level of diagnostic rigor for sim-to-real studies.

5. The approach avoids retraining the policy or encoder entirely, performing adaptation through small-scale optimization over a latent vector. This is appealing for real-world robotics, where data collection is costly. The inclusion of analysis on robustness to noisy and sparse target-domain data (Sec. 5.1 (4)) adds credibility to practical deployability.

### Weaknesses
1. Dynamics signature lacks true transition information

The MetaDynamic encoder (Sec. 4.2B) computes η solely from unordered states, ignoring actions and successor-state relations. As a result, η summarizes state visitation statistics rather than transition dynamics P(s′|s,a). This makes η non-identifiable and may cause adaptation to correct policy-induced distribution shifts rather than genuine system-dynamics changes.

2. Latent update relies on unstable closed-form inversion

The least-squares solutions (Eqs. 2-3) require inverting ΦᵀΦ and AᵀA without any regularization or conditioning safeguards. Appendix C provides no analysis of rank deficiency or noise sensitivity. In low-data or ill-conditioned regimes, this can yield unstable latent estimates, undermining the method’s reliability under realistic sim-to-real conditions.

3. Adapter optimization is underconstrained

The adapter gθ is trained with a simple projection loss (Eq. 5), which constrains the latent only within the row space of Φ_tar. Without explicit regularization or geometric constraints toward z_src, multiple gθ can fit the same projections, causing overfitting and inconsistent behavior in closed-loop execution.

4. Target reward requirement contradicts the “unlabeled” claim

Although the paper emphasizes adaptation from limited unlabeled data, both Eq. 3 and Eq. 5 require real-domain reward labels. This assumption is not addressed or justified, and in many real-world settings such rewards are unavailable or sparse. The omission makes the “label-efficient” narrative misleading and limits practical applicability.

### Questions
please address the concerns above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Found-adapt, a novel framework for addressing the sim-to-real transfer challenge in reinforcement learning. The key contribution is a latent space adaptation method that enables foundation policies to transfer across domains without expensive policy retraining. The approach consists of three main components: (1) cross-domain initial alignment via weighted least-squares regression, (2) a permutation-invariant MetaDynamic network that extracts dynamics signatures, and (3) a signature-guided adapter that refines latent representations. The method is evaluated on multiple locomotion tasks under varying system dynamics (gravity and friction variations), demonstrating effective sim-to-real gap mitigation while maintaining computational efficiency.

### Strengths
1. Clear problem formulation: The paper addresses a critical deployment challenge where traditional sim-to-real methods require expensive retraining for each new environment, and motivates the solution through an intuitive biological analogy of how humans adapt existing motor skills to new conditions without relearning from scratch.

2. Rigorous theoretical foundation: The approach is built on solid mathematical principles with formal analysis of how environment changes affect policy performance, closed-form solutions with clear assumptions, and well-justified design choices where each component addresses specific aspects of the adaptation problem.

3. Exceptional computational efficiency: The method achieves adaptation in seconds compared to hours required by baseline approaches, representing orders of magnitude speedup by freezing most network parameters and performing lightweight updates only in the latent space.

### Weaknesses
1. Limited domain coverage: The experimental validation is restricted exclusively to locomotion tasks, limiting the claimed applicability as a general foundation policy framework.

2. Sim-to-sim evaluation only: All experiments use simulated environments with modified physics parameters rather than actual real-world robots, raising concerns about whether the method can handle the full complexity of real-world dynamics.

3. Limited baseline comparisons: The paper primarily compares against Direct-Transfer, GAT/UGAT, and PAD while missing comparisons with recent domain randomization methods, meta-RL approaches designed for adaptation, and other foundation model techniques, which weakens the empirical validation of the claimed advantages.

### Questions
1. Scalability to high-dimensional action spaces: All tested environments have relatively low-dimensional continuous action spaces - how does the method scale to high-dimensional or discrete action spaces, and are there theoretical or practical limits to the dimensionality it can handle?

2. Comparison of adaptation mechanisms: How does your latent space adaptation compare conceptually and empirically to other lightweight adaptation strategies such as context-based meta-learning, fine-tuning only the final policy layers, or dynamic rescaling of actions? What makes latent adaptation specifically advantageous for the sim-to-real problem?

### Soundness
3

### Presentation
2

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
This paper proposes Found-Adapt, a latent adaptation framework for sim-to-real transfer of pre-trained foundation policies, extending the Hilbert-space foundation model (HILP; Park et al., 2024). The approach performs latent-space adaptation rather than full policy retraining. It first applies a cross-domain latent alignment via weighted regression to initialize a latent vector, updating the pre-trained model using a batch of target-domain data. Because this linear update approximates only a local warping of the latent space, the authors introduce a permutation-invariant MetaDynamic summary of target-domain states to encode dynamics differences, followed by a nonlinear adapter optimized for latent reward consistency. Experiments on locomotion tasks under simulated dynamics perturbations show improved transfer relative to direct deployment and domain randomization.

### Strengths
- The paper tackles the important problem of efficient sim-to-real adaptation for foundation policies, a critical and emerging area in RL.
- The method adapts policies through latent-space updates and avoids costly retraining.
- The paper is generally well organized and builds upon the prior HILP framework in a coherent way.
- The two modules (cross-domain alignment and nonlinear adapter) play distinct roles—one theoretically motivated, the other more practical and performance-driven.
- The paper compares against relevant baselines including direct deployment, domain randomization, and PAD.
- Results show consistent improvements over naive transfer and domain randomization, reaching similar performance to PAD.

### Weaknesses
While the paper is technically sound and addresses a timely topic, several issues limit its clarity and impact:

- Comparison to PAD: 
  - The paper reports comparable performance to PAD but frames PAD as incomparably inefficient. However, PAD’s longer pre-training time is not directly comparable to Found-Adapt’s adaptation cost, since that cost corresponds to task pre-training rather than the adaptation method itself. PAD achieves similar reward recovery with only a few hundred online samples, compared to the 5k target samples used for Found-Adapt. The comparison would be more balanced if adaptation sample efficiency (rather than pre-training) were highlighted.
  - Although the architectures differ, both methods effectively correct representation drift between domains by learning mappings that preserve task-relevant latent consistency. PAD adapts the feature extractor online via self-supervision, while Found-Adapt adds a learned adapter that refines the latent vector offline. The similarity in effective mechanism could explain their comparable empirical performance.

- The paper does not analyze how the Hilbert embeddings change under dynamics perturbations and after adaptation. Since part of HILP’s original appeal lies in the interpretable geometry of the latent space, understanding how perturbations distort these embeddings—and whether adaptation restores that structure—would be a great addition. Without this analysis, it is unclear whether the adaptation preserves the interpretable properties of the original foundation policy framework.

- Unclear role of key components: 
  - The linear realignment step largely reuses the HILP latent regression with added target data, but details such as the weighting parameter λ are not explained or analyzed. 
  - The nonlinear MetaDynamic adapter is presented as a black box; its architecture, representation, and training details are missing, making it difficult to assess generality or reproducibility.
  - It is unclear whether, after a nonlinear adapter is added, the target alignment is still needed at all, or one could use the original (i.e. $\lambda = 0$) with similar performance.

- The number of gradient steps and sample efficiency of the adaptation process are not well described. Questions about overfitting or stability with different training and architecture remain open.
- The work builds heavily on Park et al. (2024), reusing the Hilbert-space policy and regression-based task prompting. The main novelty lies in the addition of the adapter, which, though reasonable, feels more heuristic than conceptually grounded.
- All tests are sim-to-sim under controlled perturbations; no real-robot experiments are shown, despite strong emphasis on sim-to-real transfer. This limits the external validity of the claims.

- Presentation issues: 
  - Figure 5B is unreferenced.
  - Line 439 refers to a missing table ("Table ???").
  - The files in anonymous repository link are currently inaccessible, with message: “The requested file is not found.”

### Questions
1. What is the exact architecture, representation, dimensionality and training procedure of the MetaDynamic model? Is there prior work motivating this structure for cross-domain transfer?

2. How is the weighting parameter λ defined and tuned? Does its optimal value depend on target batch size?

3. What is the performance without the initial alignment ($\lambda = 0$, i.e. step 1 using only sim data)?

4. Could the proposed framework support online adaptation similar to PAD?

5. How do perturbations affect the Hilbert-space embeddings? Is there any analysis of representation drift between simulator and target domains?

### Soundness
3

### Presentation
2

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
This paper addresses the critical challenge of sim-to-real transfer for reinforcement learning policies. The authors propose "Found-adapt," a framework that builds upon foundation policies (specifically, the HILP[1] framework) to enable efficient adaptation from a source simulator to a target domain with different dynamics. The core idea is to decouple skill acquisition (pre-trained in sim) from environment adaptation via a task vector $z$ in the Hilbert task space.

The proposed adaptation mechanism consists of three stages. First, a **Cross-Domain Initial Alignment** step computes an initial latent vector $z_{src}$ via a weighted least-squares regression from both sim and real data. Second, a **Dynamics Signature** is generated using a pre-trained, permutation-invariant network to encode a "dynamics signature" $\eta$ from a small batch of target-domain data. Finally, in the **Signature-Guided Adaptation** stage, a small adapter network, $g_{\theta}$, takes both $z_{src}$ and $\eta$ as input is then fine-tuned for a small number of steps on the target-domain to get the final, adapted latent vector $z_{final}$, which is subsequently used by the frozen foundation policy.

The authors evaluate their method on several "sim-to-sim" locomotion tasks (e.g., Walker, Cheetah) with varying dynamics (friction, gravity). The results show that Found-adapt can mitigate the performance gap and offers a parameter-efficient alternative to retraining the entire policy.

### Strengths
* Important Problem: The paper tackles the sim-to-real gap, which remains a significant and high-impact obstacle for the practical application of reinforcement learning in robotics and the real world.
* Promising Framework Idea: The core concept of adapting the *latent space* of a foundation policy, rather than fine-tuning the policy network itself, is appealing. It suggests a modular, parameter-efficient, and fast adaptation method.
* Conceptually Sound Decomposition: The idea of decomposing the adaptation problem into an initial task-aligned latent ($z_{src}$) and a separate dynamics signature ($\eta$) is interesting. It attempts to distinguish "what to do" from "how the environment behaves."

### Weaknesses
* Clarity of Contribution (vs. HILP): **(major)** The paper dedicates a very large portion of its methodology (Sec 4.1) to a detailed recapitulation of the HILP framework [1], which is existing prior work. This positioning obscures the paper's own, new contributions. The actual novel parts are confined to Sec 4.2. Of these, Sec 4.2.A (weighted regression) is a straightforward and minor extension of HILP's original least-squares solver. This leaves Sec 4.2.B/C as the main contribution, however, the `MetaDynamic` network (Sec 4.2.B), is not clearly explained. The paper states it is "trained on simulator data and frozen at deployment" but provides no details on its training objective, loss function, architecture, or the data used for its pre-training. This is a major omission that makes the contribution difficult to assess and the work impossible to reproduce.

* Weak Experimental Validation (Sim-to-Sim): The paper's abstract and introduction frame the problem as "real-world application" and "sim-to-real transfer." However, all experiments are conducted in a "sim-to-sim" setting (transferring between different MuJoCo physics parameters). While this is a common practice, the lack of a single real-world hardware experiment significantly weakens the paper's central claims. 

* Marginal Empirical Results: The empirical results are not strongly convincing. In Table 1 and Figure 3, the proposed method ("Ours") shows only marginal improvements over the baselines. In some cases (e.g., Stand and Flip tasks under G3/G4 gravity), the "Ours" method appears to be outperformed by the PAD baseline, suggesting it may not be uniformly more robust. 
* Missing Related Work (RMA): The "Adaptive Policy Networks" related work section is notably incomplete. It omits the highly relevant and successful line of work on Rapid Motor Adaptation (RMA) like[2-5], which also uses a latent-conditioned policy ($z$) to adapt to different dynamics.

* Missing Critical Ablation (Direct Latent Optimization): The paper's primary claim is that its ($z_{src}$, $\eta$) $\rightarrow$ $g_{\theta}$ $\rightarrow$ $z_{final}$ pipeline is an effective adaptation mechanism. However, it fails to compare against a much simpler and highly relevant baseline: directly optimizing the latent vector $z$ itself using data from the target environment. This approach (treating $z$ as a small set of learnable parameters at deployment) has shown great success in related work (e.g., RTR [5]), which also adapts a latent-conditioned policy for sim-to-real). It is unclear if the proposed `MetaDynamic` network and adapter $g_{\theta}$ are necessary at all, or if simply fine-tuning $z_{src}$ would achieve the same or better results.

### Questions
The main questions are elaborated in the weakness part, here are some minor questions:

* In the ablation study (Fig 4a), the `F(init+dyna)` variant performs very poorly. This implies the adapter $g_{\theta}$ is not just combining $z_{src}$ and $\eta$ but reconciling them. Could the authors explain why this simple combination fails so badly and what this reveals about the learned representations of $z_{src}$ and $\eta$?
* How was the hyperparameter $\lambda$ (the weight for target-domain data in Sec 4.2.A) selected, and how sensitive is the performance of $z_{src}$ (and $z_{final}$) to this choice?
* How sensitive is the framework to the amount of target-domain data ($M$)? What is the minimum number of target samples required for the `MetaDynamic` network to extract a useful signature $\eta$ and for the adapter $g_{\theta}$ to be effectively tuned?

### Soundness
3

### Presentation
4

### Contribution
2
