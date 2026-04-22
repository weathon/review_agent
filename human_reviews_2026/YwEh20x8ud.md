# BroRL: Scaling Reinforcement Learning via Broadened Exploration

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key ingredient for unlocking complex reasoning capabilities in large language models. Recent work ProRL has shown promise in scaling RL by increasing the number of training steps. However, performance plateaus after thousands of steps, with clear diminishing returns from allocating more computation to additional training.
In this work, we investigate a complementary paradigm for scaling RL: \textbf{BroRL}—increasing the number of rollouts per example to hundreds to exhaustively \textbf{Bro}aden exploration, which yields continuous performance gains beyond the saturation point observed in ProRL when scaling the number of training steps.
Our approach is motivated by a mass balance equation analysis allowing us to characterize the rate of change in probability mass for correct and incorrect tokens during the reinforcement process. We show that under a one-step RL assumption, sampled rollout tokens always contribute to correct-mass expansion, while unsampled tokens outside rollouts may lead to gains or losses depending on their distribution and the net reward balance. Importantly, as the number of rollouts per example $N$ increases, the effect of unsampled terms diminishes, ensuring overall correct-mass expansion.
To validate our theoretical analysis, we conduct simulations under more relaxed conditions and find that a sufficiently large rollout size $N$—corresponding to ample exploration—guarantees an increase in the probability mass of all correct tokens.
Empirically, BroRL revives models saturated after 3K ProRL training steps and demonstrates robust, continuous improvement, achieving state-of-the-art results for the 1.5B model across diverse benchmarks.
Notably, under the same training time, BroRL is both more data- and compute-efficient: large-$N$ rollouts reduce the number of filtered samples during dynamic sampling at the algorithmic level and shift generation from memory-bound to compute-bound at the hardware level, nearly doubling throughput compared to ProRL in our hardware setup, highlighting BroRL’s practicality for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BroRL, a new paradigm for scaling RLVR. Unlike ProRL, which improves model performance by increasing training steps but eventually plateaus, BroRL enhances performance by increasing the number of rollouts per example, i.e., broadening exploration to achieve continuous gains even after ProRL saturation. The method is grounded in a probability mass balance analysis, showing that as rollout numbers grow, the model consistently expands the probability mass of correct tokens while minimizing the influence of unsampled terms. Empirical results confirm that BroRL revives models that have stagnated after thousands of ProRL steps, achieving SOTA results for 1.5B-parameter models across diverse benchmarks. Moreover, under the same training time, BroRL proves to be more data- and compute-efficient, nearly doubling throughput by shifting generation from memory-bound to compute-bound execution, demonstrating strong practicality for real-world deployment.

### Strengths
1. Novel scaling paradigm: this work introduces a new way to scale RL via more rollouts per example, that complements existing methods like ProRL, offering an alternative to merely increasing training steps.


2. Theoretical grounding: the paper provides a clear analytical framework by mass balance equation analysis, explaining why increased rollouts improve the probability mass of correct tokens.


3. Empirical validation: theoretical findings in this paper are supported by simulations and experiments showing consistent performance gains, even after ProRL saturation.


4. Efficiency gains: the experimental results demonstrate higher data and compute efficiency; by shifting computation from memory-bound to compute-bound, BroRL nearly doubles throughput under the same hardware conditions.

### Weaknesses
1. Limited theoretical scope: the main analysis relies on a one-step RL assumption, which may not fully generalize to multi-step or real-world RL settings. It would be helpful if the authors can provide a discussion about the extension to multi-step RL settings.


2. Empirical range: validation is mainly conducted on mid-sized models, i.e., 1.5B parameters. Scalability of BroRL to larger models or a broader range of tasks remains to be demonstrated, particularly since efficiency is a key concern for this method.


3. Naive approach: the core idea of BroRL, i.e., simply increasing the number of sampled rollouts per example, is relatively straightforward. While effective, it may be considered a naive method, as improved performance could arise largely from brute-force sampling rather than a more principled algorithmic innovation.


4. Limited exploration techniques: many alternative strategies for encouraging sampling diversity or more efficient exploration are not considered or compared. It is unclear whether similar performance gains could be achieved without resorting to large-scale rollouts, for example, by using smarter exploration or diversity-promoting methods.


Minor point: Figure 2 is not immediately clear, even though its meaning becomes understandable from the later context. It would be helpful to clarify the figure or make it more self-explanatory.

### Questions
Q1. In Figure 1, does the x-axis represent training time only, or does it also include sample generation time? If it represents training time only, its significance may be limited, since the number of training samples for ProRL is relatively small.

### Soundness
3

### Presentation
2

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
This paper proposes BroRL, a reinforcement learning framework that scales the number of rollouts per prompt to improve training performance. The authors develop a mass transfer model to explain how increasing the rollout size N reduces the coupling of unsampled probability mass, thereby increasing the positive probability shift $\Delta Q_{pos}$. Experiments conducted on a token-level simulator and following the ProRL training setup show a small but statistically significant improvement in test scores. The paper also claims improved hardware utilization and generation throughput compared to standard RL methods.

### Strengths
- The mass transfer modeling is insightful. It provides a way to analyze the probability change in RL.
- Under the same GPU computing budget, BroRL achieves better performance than ProRL.

### Weaknesses
- Limited novelty. The core idea is simply increasing the rollout size from 16 to 512, which is incremental. The paper lacks a clear rationale for choosing 512 and does not systematically study how varying N affects RL dynamics. Discussion of moderate (e.g., 64) or extreme (e.g., 2048) rollout sizes is missing.
 
- Efficiency concerns. Prior work [1][2] shows that in later training stages, easy problems dominate. In BroRL, generating 512 rollouts for these trivial cases may waste substantial computation. While the paper reports higher generation throughput, it does not clearly link this to actual learning efficiency or performance improvement.

[1] Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts

[2] POLARIS: A POst-training recipe for scaling reinforcement Learning on Advanced ReasonIng modelS

### Questions
1. In the mass transfer analysis, larger rollout sizes seem consistently beneficial. Do the authors agree that "larger is always better"? Have you experimented with rollout sizes greater than 512?

2. During training, with 512 rollouts per question, what is the distribution of correct versus incorrect samples? Does the "easy-problem dominance" phenomenon mentioned in [1][2] also appear in BroRL?

### Soundness
2

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
2

### Summary
This paper investigates a novel dimension for scaling Reinforcement Learning with Verifiable Rewards in large language models. The authors posit that increasing the number of rollouts per prompt ($N$) to a large number (e.g., 512), rather than solely scaling the number of training steps, can lead to more stable and performant policy optimization. They provide a theoretical analysis based on a "mass balance equation" to show that a large N suppresses a potentially negative "unsampled coupling" term in the policy update, guaranteeing an increase in the probability mass of correct tokens. Empirically, they demonstrate that applying BroRL to a saturated ProRL model yields continued performance improvements on reasoning benchmarks and offers computational efficiency gains by shifting the GPU bottleneck from memory to compute.

### Strengths
- The idea of scaling the "width" of exploration (rollouts per prompt) instead of just the "depth" (training steps) represents an under-explored direction in RL scaling literature. It directly addresses the common problem of performance plateauing in prolonged RL training.

- The paper provides a theoretical foundation for its claims. The mass balance equation and the decomposition of $\Delta Q_{pos}$ offer a clear, mechanistic explanation for why increasing $N$ should lead to more stable learning by mitigating the variance from unsampled tokens.

- The experiments are generally well-designed to support the core thesis. The token-level simulation cleanly validates the theoretical dynamics, and the successful application to a large-scale model that had already plateaued is a strong, practical demonstration of BroRL's potential.

### Weaknesses
- Incomplete Ablation on $N$: The paper jumps from $N=16$ (ProRL) to $N=512$ (BroRL). A more granular ablation study with intermediate N values (e.g., 64, 128, 256) is missing. This would be crucial for understanding the scaling law's shape, identifying potential diminishing returns, and providing practical guidance for choosing N. The authors acknowledge this in Appendix B, but it weakens the current analysis.

- All large-scale empirical results are based on a 1.5B parameter model. This severely limits the generalizability of the findings. It is well-established in LLM literature that scaling laws and algorithmic behaviors can differ significantly between small models and larger-scale models. The claim that BroRL is a general scaling law for RLVR is not fully substantiated without validation on larger model sizes.

### Questions
- The theoretical analysis is performed with a TRPO-style linear surrogate. To what extent do you believe the insights from Theorem 1 hold for the actual clipped PPO/GRPO objective with a KL penalty?

- Why was N=512 chosen for BroRL? Was this based on preliminary scaling experiments, hardware constraints, or intuition? Given the lack of a sweep over N, it is unclear if this is an optimal or merely a sufficient value.

- How BroRL works on other RL algorithms (e.g. DAPO)?. The related work section acknowledges ProRL's step-scaling but does not sufficiently situate BroRL against other RL algorithm families.

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
3

### Summary
BroRL reframes RLVR scaling by turning up rollouts per prompt (N) rather than training steps. A mass-balance analysis decomposes the change in correct-token probability into a nonnegative sampled term plus an “unsampled coupling” term whose influence fades as N grows, so larger N drives net expansion of correct mass.  Simulations with a TRPO-style surrogate corroborate that sufficiently large N eliminates knowledge shrinkage and monotonically increases correct-token mass.  Empirically, continuing a 1.5B model that plateaus under ProRL after ~3k steps, BroRL sustains improvements across math/code/multidomain benchmarks while ProRL stalls or degrades.  Practically, big-N rollouts raise dynamic-sampling pass rate (62% vs 41%) and nearly double generation throughput (72.4 vs 36.5 samples/s) by shifting generation from memory-bound to compute-bound, improving algorithmic and hardware efficiency.  Overall, the paper argues that exploration breadth—not step count—is the critical, implementation-friendly axis for stable, scalable RLVR, turning saturated training into continued gains with better compute utilization.

### Strengths
1. The proposed method is simple and shows effective improvement across diverse benchmarks over the baseline method ProRL.
2. This work provides a clear mass-balance decomposition and shows sampled terms are always non-negative and the unsampled coupling term shrinks with larger rollout size (N). 
3. token-level TRPO-style experiments verify that big-N stabilizes updates, accelerates correct-mass growth, and eliminates knowledge-shrinkage.

### Weaknesses
1. The idea of "scaling the number of sampled responses" is not novel and has been proposed and studied in previous works [1,2], demonstrating its effectiveness. These works also prove that scaling the number of sampled responses can benefit the scaling trends of RL training.

2. The detailed training settings are not provided, including the base model, the starting point, and the training dataset. The lack of information hinders the reproduction and evaluation of this work.

3. The improvement is relatively marginal (~1% for most benchmarks) under similar GPU training hours. Though increasing the number of rollouts can boost further performance improvement in RL training, the performance gain is too marginal, which makes the contribution of this work less significant. The authors may further study how to achieve more significant scaling gains or demonstrate the effectiveness on larger models.

[1] Hou et al, T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling.

[2] Shao et al, DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2
