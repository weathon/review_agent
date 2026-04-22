# GEPO: Group Expectation Policy Optimization for Stable Heterogeneous Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
As single-center computing approaches power constraints, decentralized training becomes essential. However, traditional Reinforcement Learning (RL) methods, crucial for enhancing large model post-training, cannot adapt to decentralized distributed training due to the tight coupling between parameter learning and rollout sampling. For this, we propose HeteroRL, a heterogeneous RL architecture that decouples these processes, enabling stable training across geographically distributed nodes connected via the Internet. The core component is Group Expectation Policy Optimization (GEPO), an asynchronous RL algorithm robust to latency caused by network delays or heterogeneity in computational resources. Our study reveals that high latency significantly increases KL divergence, leading to higher variance of importance weights and training instability. GEPO mitigates this issue by using group expectation weighting to exponentially reduce the variance of importance weights, with theoretical guarantees.  Experiments show GEPO achieves superior stability—only a 3\% performance drop from online to 1800s latency—and reduces the best-to-last gap by 85\% versus GSPO ($\Delta$=1.8 vs. 12.0) while attaining the highest scores, highlighting its effectiveness in decentralized, resource-heterogeneous environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes HeteroRL, an asynchronous framework for RLHF that decouples rollout sampling from learning to tolerate WAN latency and heterogeneous hardware. Its key algorithm, GEPO, replaces token/sequence importance ratios with a group expectation denominator, theoretically reducing IS variance—especially at high sampler–learner KL—and stabilizing training under policy staleness. Stronger last-epoch performance and robustness vs. GRPO/GSPO are shown on math-reasoning benchmarks under simulated delays.

### Strengths
- Originality: It makes sense by clear shift from token/sequence to group-level weighting tied to the real failure mode.
- Theory: It gives exponential variance reduction of importance weights in high-KL regimes; mechanistic gradient comparison clarifies why updates stabilize. 
- Empirics: GEPO maintains accuracy and avoids collapse across delays; best/last markedly stronger than GRPO/GSPO.

### Weaknesses
- Bias–variance trade-off not quantified. The authors should provide MSE curves vs. KL and group size, and analyze the green-region cases where variance may increase.
  
- Bound constants/tightness unclear.The theorem invokes constants (e.g., \(C\)) but does not estimate or validate their tightness on real training distributions; what about reporting empirical lower/upper bounds and how they vary with vocabulary/sequence length.
  
- Systems throughput and realism underreported.The method targets WAN latency, but the authoers do not provide wall-clock throughput (samples/s), utilization, or sensitivity to jitter/packet loss; it is better to add end-to-end performance, synchronization frequency, and utilization metrics.

- Missing off-policy/staleness baselines. I am curious about the comparison omits V-trace/IMPALA-like corrections and AREAL-style staleness controls, which are only mentioned in related works; the authors should add these baselines to separate gains from grouping vs. specific normalization.
  
- Sensitivity analyses incomplete.The stability claim depends on grouping and sampling, but the paper does not sweep group size \(G\), temperature/top-p, or KL-regularization × delay.

### Questions
As shown in weakness.

Btw, why placing the related work section in the final, which is not usual. And the related works seem limited and unorganized.

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
2

### Summary
This paper presents HeteroRL, a decentralized reinforcement learning framework designed for training LLMs across distributed and resource-heterogeneous environments. The paper introduces GEPO, including a new group expectation importance weighting mechanism to mitigate the instability caused by policy staleness. Theoretical results show that GEPO exponentially reduces the variance of importance weights under large KL, thereby stabilizing training in asynchronous and high-latency settings. Extensive experiments demonstrate the method's superior performance and stability on reasoning benchmarks.

### Strengths
1. The idea of reducing the variance of importance weights through group expectation weighting is conceptually elegant and practically powerful. It addresses the instability issue caused by large KL divergence in asynchronous or heterogeneous RL settings.
1. The motivation, instability and variance explosion in asynchronous or heterogeneous RL due to policy staleness, is well validated by experimental results. The results across both online and heterogeneous RL settings consistently demonstrate that GEPO achieves lower variance, smoother gradients, and higher stability compared to GRPO and GSPO.

### Weaknesses
1. The experimental validation appears limited in scope. GEPO is only evaluated on mathematical reasoning datasets (MATH, AIME, AMC) and with relatively small models (up to 8B parameters).
1. There seems an inconsistency between the formulation and the theoretical analysis. In Section 3.1, the paper explicitly states that "the vector $(q(y_1|x), …, q(y_G|x))$ does not constitute a valid probability distribution" since top-K/top-P sampling leads to $\sum_i q(y_i|x) \gg 1$. However, in Theorem 1 and its proof, the derivation assumes that $q$ is a normalized discrete probability distribution satisfying $\sum_i q(y_i|x) = 1$. How the theoretical guarantee holds when $q$ is not a valid probability distribution in practice?

### Questions
1. Could the authors clarify whether the ablation experiments comparing GEPO, GRPO, and GSPO are conducted under the same experimental settings as the main comparison in Section 4? If the setups are identical, how do these ablation results differ conceptually from the earlier baseline comparison?
2. Why does GEPO remove the normalization when calculating the advantage? In practice, would including this normalization affect the performance or training stability? 
3. It is unclear how the case study in section G is directly connected to the theoretical claim about variance reduction in importance weights.

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
5

### Summary
The paper introduces GEPO, a reinforcement learning algorithm designed for heterogeneous, high-latency distributed training of large language models. It builds on a new framework that decouples sampling and learning across asynchronous nodes. To address instability caused by stale policies, GEPO replaces standard importance weights with group-level expectation weights, theoretically reducing variance when policy divergence is high. Experiments on mathematical reasoning tasks with Qwen models show that GEPO achieves higher accuracy and significantly improved training stability compared to GRPO and GSPO under simulated latency.

### Strengths
1. The visualizations illustrate the main claims of the paper.
2. The paper targeted on an important bottleneck problem in large-scale distributed reinforcement learning.

### Weaknesses
1. The paper lacks experimental comparisons with other asynchronous policy optimization methods [1, 2, 3, 4].
2. Equation (1) appears very similar to PPO, except that the clipping function is removed.
3. In line 139, the variable $G$ is undefined; in Equation (1), it is unclear how $A(x)$ is computed, and in line 146, the input of $p$ is not specified.
4. While Theorem 1 demonstrates a reduction in the variance of the importance sampling coefficient, this result does not guarantee a corresponding reduction in the variance of the weighted term $A \cdot w$, because $A$ itself is a random variable. In general, even if one random variable has smaller variance than another (e.g., $\mathrm{Var}(Y)<\mathrm{Var}(Z)$), it does **not** necessarily follow that $\mathrm{Var}(XY)<\mathrm{Var}(XZ)$, unless $X$ is independent of both $Y$ and $Z$ and their expectations are zero.
5. The claim in lines 178–181 seems questionable. In extreme cases, clipping can suppress unstable tokens by assigning them zero weight, which may be beneficial since such extreme tokens usually correspond to unreliable generations. Thus, clipping might actually serve as a useful safeguard in these scenarios.

 [1] Rastogi, Abhinav, et al. "Magistral." *arXiv preprint arXiv:2506.10910* (2025).

 [2] Fu, Wei, et al. "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning." *arXiv preprint arXiv:2505.24298* (2025).

 [3] Zhong, Yinmin, et al. "StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation." *arXiv preprint arXiv:2504.15930* (2025).

 [4] Han, Zhenyu, et al. "AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training." *arXiv preprint arXiv:2507.01663* (2025).

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

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
The paper proposes HeteroRL, a decentralized RL framework that decouples rollout sampling from learning to tolerate Internet-scale latency and heterogeneous hardware. Its core algorithm, GEPO (Group Expectation Policy Optimization), replaces token/sequence-level importance weights with a group-level expectation to reduce the variance explosion that arises when the learner’s policy drifts from the samplers due to staleness. A theoretical result shows variance reduction grows (approximately) exponentially with KL divergence between policies, and experiments on Qwen3-1.7B/8B with math-reasoning benchmarks (AIME/MATH/AMC) report higher accuracy and markedly improved stability under simulated delays (up to 1800s).

### Strengths
**Clear systems problem + algorithmic handle**. The paper crisply diagnoses policy staleness in decentralized RL and ties it to KL growth → variance blow-ups; GEPO’s denominator uses a within-group expectation to damp variance in precisely that regime.

**Compelling empirical stability**. Under Hetero RL with max delay 64, GEPO improves best accuracy vs. GRPO/GSPO and, crucially, reduces best-to-last degradation by ~85% vs. GSPO (Δ=1.8 vs. 12.0). Curves show lower IW variance and smoother gradients.

**Theoretical insight**. Theorem 1 relates the variance gap (standard IS vs. group-expectation weight) to exp(D_KL), motivating why the method helps exactly when staleness is worst.

**Realistic setting**. The system model explicitly injects stochastic network delay and hetero nodes (Ascend + NVIDIA), aligning with decentralized community compute setups.

### Weaknesses
**Bias–variance trade-off left under-quantified in RL objective**. GEPO’s estimator is acknowledged as biased; while lower variance can help optimization, the paper does not quantify end-to-end bias in policy gradients or returns beyond variance plots. A small-bias claim would benefit from controlled ablations where true on-policy gradients are approximated (short-horizon toy MDPs) to measure bias vs. sample efficiency. (GEIW is described as biased but stable.)

**External validity beyond math-reasoning**. Results are limited to math QA on Qwen. It would help to show GEPO under non-text or mixed-modality tasks, or code/data generation tasks where rollouts have different length/entropy profiles. Current related work cites broader systems, but experiments remain narrow.

**Ablations & knobs**. The method depends on group size G and sampling strategy (top-k/p). Sensitivity plots for G, the truncation strategy, and the CPPO-KL coefficient under delay would strengthen the story. (Implementation notes mention CPPO-KL and latency simulation, but tuning studies are light.)

**Fairness to GSPO variants**. Since GSPO is sequence-level IS with clipping, it would be good to include GSPO + stronger clipping / trust-region or defensive mixture baselines to test whether variance spikes can be similarly tamed by tuned GSPO, not just GEPO’s denominator. The paper itself sketches “defensive sampling” as future work—great idea, but makes me wonder how close-tuned GSPO would get.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
