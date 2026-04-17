# Towards Noise‐Robust Multi‐Agent Imitation Learning via Global Credit Sequence Decoding

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Multi-Agent Reinforcement Learning (MARL) has emerged as a promising approach to solving complex decision-making problems such as multi-agent collaboration. To avoid the difficulty of designing complex reward functions, researchers increasingly adopt imitation learning. Classical methods extend single-agent imitation learning to multi-agent settings by matching distributions from expert demonstrations. However, noisy or low-quality trajectories within these demonstrations can mislead joint policy optimization, leading to significant performance degradation. This study introduces a sequential autoregressive architecture that models global dependencies among agents, facilitating adaptive credit assignment and policy optimization. The architecture theoretically enhances the variance of joint advantages and rewards, addressing issues like vanishing gradients and mode collapse caused by noisy demonstrations. Experiments show that our method achieves a significant performance improvement on multiple benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MILD^2 — a transformer-based generative adversarial imitation learning framework that explicitly models global dependencies among agents to improve robustness against noisy expert demonstrations.
The core idea is that conventional multi-agent GAIL frameworks suffer from low reward variance and vanishing gradients when facing noisy or suboptimal demonstrations.
The authors theoretically show that modeling joint dependencies can increase advantage variance, thereby enhancing training stability, and empirically validate their claims on SMAC, Google Football, Bi-DexHands, and MA-MuJoCo benchmarks, where MILD² achieves consistent improvements over several baselines.

### Strengths
- The paper identifies a concrete and important issue in multi-agent imitation learning — low reward variance and poor robustness to noisy demonstrations.

- It provides a theoretically grounded argument showing that dependency-enhanced discriminators can increase joint advantage variance and alleviate gradient vanishing.

- Experiments span both discrete (SMAC, Football) and continuous (Bi-DexHands, MA-MuJoCo) domains, demonstrating consistent empirical gains.

### Weaknesses
1. Missing baselines.

The paper omits several relevant offline MARL baselines and recent transformer-based approaches, such as [1–4]. Including these would strengthen the empirical comparison and clarify the contribution beyond architectural novelty.

2. Unclear task selection.

While multiple benchmarks are considered, only a few tasks from each benchmark are used. The rationale for this selection is not clearly explained, raising concerns that results might be cherry-picked.

3. Incomplete ablation analysis.

The ablation study does not explore key factors, such as varying the weight λ of the reward variance regularization term.
In addition, the claim of “discovering higher-rewarding behaviors beyond expert demonstrations” implies potential out-of-distribution generalization, which is not thoroughly analyzed or quantified.

4. Robustness not empirically supported.

Although robustness to noisy demonstrations is emphasized as a main contribution, the paper lacks explicit experiments where demonstration noise is systematically varied.

[1] Offline Pre-trained Multi-agent Decision Transformer

[2] Offline Multi-agent Reinforcement Learning with Knowledge Distillation

[3] Counterfactual Conservative Q-learning for Offline Multi-agent Reinforcement Learning

[4] Offline Multi-agent Reinforcement Learning with Implicit Global-to-Local Value Regularization

### Questions
1. Could the authors specify the exact objective function used for the MILD^2-GD variant (without dependency modeling)?

2. The sequential autoregressive generator introduces inter-agent dependencies — does this affect decentralized execution at test time?

3. Could the same variance-regularization effect be achieved via simpler covariance-based or entropy-based regularizers, without requiring a transformer-based discriminator?

### Soundness
2

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
4

### Summary
This paper proposes MILD$^2$, a sequential autoregressive approach, to address the vanishing gradient problem in independent action distribution matching, which arises from low reward (or advantage) variance. The authors theoretically demonstrate that MILD$^2$ achieves a higher advantage variance compared to the independent approach. The experiments comparatively illustrate the changes in reward variance and performance improvements when using MILD$^2$.

### Strengths
- The paper provides a theoretical proof that state, sequential conditional action distribution matching is superior to independent state, action distribution matching in terms of advantage variance. This offers a theoretical basis for resolving the aforementioned problem.

- The experiments effectively demonstrate that a higher reward variance slows down the discriminator's training, leading to improved performance.

### Weaknesses
- The methodological novelty appears to be limited. The approach seems quite similar to applying the transformer architecture for sequential autoregression from MAT [1] to MAGAIL [2]. It largely looks like an application of MAT to multi-agent imitation learning.

- The term "global dependency" is frequently used (e.g., "global dependency-enhanced discriminator"), but it is not clearly defined within the paper.

- While the paper theoretically suggests that using global dependency can improve performance, there is no theoretical proof quantifying this improvement or demonstrating that it can resolve mode collapse.

- There seems to be a misalignment between the motivation presented in Figure 1 and the problem the paper aims to solve. Figure 1 appears to suggest that distribution matching should differ based on the actions of important versus unimportant agents. It is unclear how this aligns with the concept of "global dependency distribution matching."

- Given the limited research in the field of multi-agent imitation learning, the ablation studies currently in the appendix seem significant enough to be included in the main paper.

- The paper appears to be missing references to recent work in multi-agent imitation learning [3, 4]. If possible, a comparison with the algorithms from these papers would be beneficial.


[1] M. Wen, et al., “Multi-agent Reinforcement Learning is a Sequence Modeling Problem,” NeurIPS 2022.

[2] J. Song, et al., “Multi-agent Generative Adversarial Imitation Learning,” NeurIPS 2018.

[3] L. Yu, J. Song, S. Ermon, “Multi-agent Adversarial Inverse Reinforcement learning”, ICML 2019.

[4] T.V. Bui, T.A. Mai, T.H. Nguyen, “Inverse Factorized Soft Q-Learning for Cooperative Multi-agent Imitation Learning,” NeurIPS 2024.

### Questions
- Are there any structural differences between the proposed transformer-based encoder-decoder and the architecture used in MAT? If so, what are the key distinctions?

- What is the rationale for comparing the proposed method with offline MARL methods in Table 1?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes MILD², a MAIL method that performs global-dependency-enhanced distribution matching by (i) a Transformer encoder-decoder discriminator that outputs per-agent expert logits under Lipschitz regularization and (ii) a sequential autoregressive policy to mirror cross-agent dependence and reduce gradient variance. The theory formalizes why MAGAIL yields low-variance/imbalanced updates under noisy demos and shows that modeling dependencies increases advantage variance relative to independent frameworks. Experiments on SMAC / GRF / Bi-DexHands / MA-MuJoCo show consistent improvements over MAGAIL, CQL-MA, ICQ-MA, TD3-BC, and OMAR, plus robustness when up to 50% of the dataset is noisy.

### Strengths
Addresses a real MAIL failure mode (noisy demonstrations ⇒ weak discriminator signals and poor credit assignment) with a unified theory-to-architecture path.

Practical architectural choices (Transformer with Lipschitz-aware regularization; autoregressive policy) that operationalize the theory.

Evidence spans multiple domains; ablations indicate that dependency modeling drives larger reward dispersion and more stable discriminator dynamics; noise-robustness is strong.

### Weaknesses
Baselines under-represent modern MAIL variants (centralized/explicit-dependency discriminators, improved MA-AIRL).

Permutation-equivariance not addressed; sequential ordering may induce bias for homogeneous agents.

No efficiency/scaling analysis; attention cost may be prohibitive as #agents grows.

Using variance as a proxy for signal quality risks amplifying wrong signals; no complementary discriminability metrics (e.g., margins, mutual information).

Benchmark coverage could expand (more MA-MuJoCo tasks; success-rate reporting in Bi-DexHands)

### Questions
Will you add stronger MAIL baselines or centralized/correlated discriminators to control for the source of improvement?

How do you mitigate ordering bias (e.g., order shuffling, set/graph-equivariant designs, or order-ensemble training)?

What are the GPU time/memory footprints vs. agent count and observation size?

Can you report signal-quality metrics (margin/InfoNCE) and mis-specification stress tests to ensure variance ≠ noise amplification?

Any results with human-collected noisy demos or OOD state coverage?

### Soundness
2

### Presentation
3

### Contribution
2
