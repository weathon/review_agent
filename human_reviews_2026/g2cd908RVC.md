# Learning-To-Measure: In-Context Active Feature Acquisition

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Active feature acquisition (AFA) is a sequential decision-making problem where the goal is to improve model performance for test instances by adaptively selecting which features to acquire. In practice, AFA methods often learn from retrospective data with systematic missingness in the features and limited task-specific labels. Most prior work addresses acquisition for a single predetermined task, limiting scalability. To address this limitation, we formalize the meta-AFA problem, where the goal is to learn acquisition policies across various tasks.  We introduce Learning-to-Measure (L2M), which consists of i) reliable uncertainty quantification over unseen tasks, and ii) an uncertainty-guided greedy feature acquisition agent that maximizes conditional mutual information. We demonstrate a sequence-modeling or autoregressive pre-training approach that underpins reliable uncertainty quantification for tasks with arbitrary missingness. 
L2M operates directly on datasets with retrospective missingness and performs the meta-AFA task in-context, eliminating per-task retraining. Across synthetic and real-world tabular benchmarks, L2M matches or surpasses task-specific baselines, particularly under scarce labels and high missingness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces and studies meta–active feature acquisition: from retrospective datasets where features are missing, learn a single meta, in-context policy that sequentially acquires features without per-task retraining. The method, Learning-to-Measure (L2M), uses a transformer to (i) estimate output uncertainty for partially observed inputs and (ii) choose the next feature by minimizing a surrogate for greedy conditional mutual information (CMI). It also provides a causal result stating that, under missingness-at-random, exclusion, and positivity, the target CMI on retrospective datasets equals the CMI computed from complete cases. It empirically tests L2M on synthetic GP and semi-synthetic (Metabric, MiniBooNE, MIMIC-IV) tasks, as well as an MNIST block-acquisition task.

### Strengths
- **Theoretical soundness.** The paper provides a novel and clear meta-formulation of Active Feature Acquisition and theoretically extends single-task greedy surrogates to actions conditioned on variable-length contexts.
- **Flexible implementation.** The authors first pre-train a meta-predictor, then train a meta (blocked) policy that makes training more seamless on retrospective datasets with missing features.
- The paper provides **reasonable empirical evidence** that L2M helps when labels are scarce or when the missingness is significant. The paper is also transparent about assumptions and limitations.
- The method and proofs are clearly written, as well as the architecture and training details for reproducibility.

### Weaknesses
- **Baselines** are simple MLP or RL AFA methods trained per task. Meta-learning methods with potential minor per-task tuning are missing.
    - Some tasks show L2M close to “static” or even random strategies, which suggests limited headroom and potentially lower performance to stronger baselines. On this line, authors could add an oracle to bound possible gains and contextualize the significance of their method.

I don’t find any other major actionable weaknesses; my remaining primary concern is the significance of the contributions, since the pieces make only minor adaptations to existing theory or ideas. This prevents me from increasing my score, but I am open to changing my opinion regarding the novelty based on the author’s response.

### Minor weaknesses

None of these items has a significant impact on my score, but I encourage the authors to consider incorporating some of them.

1. **Architecture choices are not substantiated.** They make some ad-hoc choices (e.g.,  masked features concatenated with masks). However, no ablations are showing whether their different design choices that depart from standard practice matter. 
2. **Causal identifiability is brittle.** The paper acknowledges that MAR, exclusion, and positivity are strong assumptions. Still, given that it uses synthetic data, I would encourage an empirical stress test (e.g., simulate MNAR or partial violations of positivity) to assess robustness to mild violations of these assumptions.
3. **Pretraining priors are small.** I would encourage exploring an experiment that pre-trains on fully synthetic Bayesian networks in tabPFN-style [1]. This is a natural extension that could significantly increase the empirical significance.
4. The method section says the framework can leverage pretrained LLMs, but experiments do not evaluate LLM-based backbones, and effectiveness is unclear. This should be stated as future work or demonstrated empirically.
5. **Cost-sensitive variants.** Realistic AFA requires per-feature costs; even a small study (variable costs on MIMIC-IV labs) would strengthen the paper.
6. **L121**: Typo “Time-invaryiant”

[1] Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2022). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. *ArXiv*. https://arxiv.org/abs/2207.01848

### Questions
No specific questions; please take a look at the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of active feature acquisition (AFA) by reformulating it as a meta-learning task. The authors argue that traditional, single-task AFA methods are not scalable and struggle with retrospective datasets that have systematic missingness and scarce labels. To solve this, the paper introduces Learning-to-Measure (L2M), an in-context learning framework. L2M uses a sequence model (a Transformer) that can adapt to a new, unseen task by conditioning on a "context" of a few labeled samples from that task. Experiments on synthetic and semi-synthetic tasks derived from real-world tabular datasets (MIMIC-IV, Metabric, etc.) show that L2M outperforms task-specific AFA baselines (like GDFS and DIME), especially in the challenging (and realistic) regimes of high missingness and scarce labeled data (small context sizes).

### Strengths
-	While the practical usefulness is somewhat questionable, framing AFA as a meta-learning or in-context learning problem is a departure from the previous works, which directly addresses the critical issue of scalability and adaptation that plagues traditional single-task AFA models.

### Weaknesses
-	The motivation for a non-greedy AFA framework is sound—identifying when certain features act as strong indicators for acquiring subsequent, highly informative ones (as illustrated in the paper’s chest-pain triage example) is indeed valuable. However, similar motivations have been explored in recent work [A], which also highlights the limitations of conditional mutual information objectives. Moreover, reinforcement learning–based approaches that incorporate discounted future rewards are already capable of addressing such sequential dependency issues, weakening the novelty claim of the proposed formulation. Moreover, the proposed method adopts a greedy, one-step CMI maximization policy. This is a significant drawback, as it fails to capture multi-step acquisition strategies and may result in myopic decisions, as highlighted in [A].
-	It is unclear how Equation (1) leads to different acquisition behaviors when missingness indicators are included as input features. The experiments do not explicitly analyze or visualize how the model leverages these indicators to guide acquisition decisions.
-	The paper claims that the feature acquisition policy is meta-learned across datasets. However, in practice, the policy generates a categorical distribution over a fixed feature dimension $d$, assuming a shared feature space across all tasks. This assumption substantially limits the method’s applicability to real-world settings where feature spaces often differ across domains or datasets.
-	The missingness patterns in experiments are synthetically constructed by the authors rather than drawn from naturally incomplete real-world datasets. Evaluating the approach on datasets with naturally occurring missingness would provide stronger evidence of its robustness and practical utility. 

[A] Norcliffe et al., “Stochastic Encodings for Active Feature Acquisition,” ICML 2025.

### Questions
- Have the authors considered ablations in settings known to be difficult for greedy policies (e.g., high feature redundancy)? How would L2M compare to a non-myopic, multi-step lookahead policy such as [A], even a simple one, built on top of L2M's excellent uncertainty-quantifying sequence model?

- The procedure for setting the blocked policy is not clearly described. How are the blocked features determined (i.e., how is R_j=0assigned)?

- The meta-learning is demonstrated on tasks sampled from synthetic (GP) or semi-synthetic (BNN) priors. It is unclear how well this pre-training will generalize to a distribution of truly diverse real-world tasks. The mixed results on "real tasks" (Figure 8) suggest the BNN prior may not be rich enough, which could undermine the central meta-learning claim.


[A] Norcliffe et al., “Stochastic Encodings for Active Feature Acquisition,” ICML 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work extends the existing active feature acquisition (AFA) from single-task to multi-task scenarios based on the meta-learning paradigm — Learning-to-Measure (L2M). Specifically, the authors highlight that maximizing the conditional mutual information (CMI) between adding the new features $X_j$ and the prediction $Y$ based on the current state $\underline{X}_t$ should consider proper uncertainty quantification for the unseen task under missingness in historical data. After proposing the theoretical proof of the identification of CMI with missingness in the current state, they propose a surrogate optimization problem to estimate uncertainty quantification during the pre-trained (meta-learning) stage and train the feature acquisition policy model and the original models. The empirical results demonstrate the competitive uncertainty estimation and query performance on both synthetic and real-world datasets.

### Strengths
1. This work provides the solid theoretical proof to claim the feasibility of the estimation of the CMI from the missingness in the historical data.
2. The authors carefully construct the surrogate optimization problem with a clear roadmap from the theoretical proof.
3. The empirical results show that the feasibility of their ideas could be used for the transformers model and real-world datasets, which would appeal to more researchers to follow up on this direction.

### Weaknesses
1. While the authors provide examples about the missing features on the historical data, such as clinical data in Sec 1, they assume that the features are time-invariant in Sec 3, which sounds conflicting to me. Is my understanding of the time-invariant features incorrect?
2. Although the authors provide good demonstrations on the real-world tabular datasets, these datasets' dimensions still seem not big enough. I encourage the authors could try large-dimensional tabular datasets such as (1) Musk, (2) Bioresponse, and (3) Diabetes 130-US Hospitals.
3. Typo in Line 242, *with sufficient flexibility ~~while~~ while providing principled...*.

- (1) Chapman, D. & Jain, A. (1994). Musk (Version 1) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZK5B.
- (2) https://www.openml.org/search?type=data&status=any&id=4134&sort=runs
- (3) Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.

### Questions
1. In your illustrations in Figure 2, the missingness of a column might only be for a few records (rows). During training, how do you handle missing records and those that remain?

### Soundness
3

### Presentation
3

### Contribution
3
