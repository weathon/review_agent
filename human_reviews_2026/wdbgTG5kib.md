# RL for Reasoning by Adaptively Revealing Rationales

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Learning in the combinatorially large output space of sequence generation problems is challenging as providing expert demonstrations scales poorly with sequence length, and RL struggles with sparse rewards. Between dense demonstrations in supervised training and no demonstrations in reinforcement learning lies an underexplored regime: partial supervision. We ask whether some classes of sequence learning problems become efficiently learnable by exploiting this gap. We address this by introducing adaptive backtracking (AdaBack), a per-sample curriculum learning algorithm that reveals a partial prefix of the target output. The supervision length is adjusted dynamically for each sample based on the model’s past reward signal, allowing it to incrementally learn to complete reasoning chains by conditioning on correct partial solutions. We investigate this intermediate regime between SFT and RL and argue that per-sample curriculum learning is more than a trade-off between efficiency and generality—it can succeed in tasks with long sequences of latent dependencies where SFT and RL both fail to generalize. Using a synthetic task with latent parity constraints, we show that AdaBack reliably solves problems that are otherwise intractable. On three mathematical reasoning benchmarks, DeepScaleR, MATH, and GSM8k, we find that AdaBack enables models to solve problems that RL alone cannot, acquiring new reasoning capabilities through incremental exposure to partial solutions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles the problem of RL for long-horizon, CoT reasoning, where exploration is hard because correct sequences are exponentially rare. It proposes AdaBack, a per-sample curriculum that, for each training example, reveals a variable-length prefix of the gold rationale and uses the reward signal to shrink or expand that revealed prefix over time, effectively doing a stochastic binary search on “how much to reveal.” The method is instantiated on top of GRPO and aims to occupy the underexplored middle ground between fully SFT on rationales and fully unsupervised RL. The authors show (i) on a synthetic chain-of-parities task that SFT, RL, and SFT+RL all fail, but AdaBack succeeds, and (ii) on MATH, GSM8k plus two harder GSM8k variants (Base-7 and Tensor-2) that AdaBack improves over RL and SFT+RL, sometimes even when starting from a base model rather than an SFT one (Table 1). They also identify a regime where AdaBack does not help -- when the base/instruct model has essentially memorized the dataset—arguing that AdaBack’s main value is in exploration-limited settings.

### Strengths
1. Originality: The paper identifies a concrete, underexplored regime “between SFT and RL” and operationalizes it as adaptive partial supervision that is per-sample, not global.
2. Experiments span standard math (MATH, GSM8k) and two carefully designed variants (Base-7 and Tensor-2) that stress symbolic shift and depth, and AdaBack improves or matches baselines in most cells of Table 1.
3. Fig. 4 suggest AdaBack expands the solution distribution, not just reweights existing solutions.

### Weaknesses
1. The benchmarks are too easy, especially GSM8k.
2. The key hyperparameter $\tau$ effectively sets how aggressively the supervision interval is shrunk/expanded, but there is no systematic sensitivity analysis (e.g., $\tau$=0.1, 0.3, 0.5, 0.7) to show robustness across tasks.
3. It seens like AdaBack is not that general. The authors focus on tasks that have a narrow scope.
4. The method keeps per-sample supervision intervals and falls back to global moving averages when samples are rarely seen; this is acknowledged as a limitation (Sec. 5), but the paper does not quantify how quickly performance degrades as dataset size grows. A synthetic experiment varying number of unique problems would clarify scalability.

### Questions
1. Could you conduct experiments on a more difficult benchmark such as AIME24 or AIME25?
2. You mention that AdaBack can be used with non-GRPO RL (via other value estimates). Did you try plain REINFORCE/RLOO, and if so, were the gains comparable?
3. Could you report variance / multiple seeds for the chain-of-parities experiment in Fig. 2?
4. How many rollouts per question were used in GRPO, and does AdaBack still help if we reduce that number (i.e., with weaker reward estimation)?
5. Could you give a systematic sensitivity analysis about the hyperparameter $\tau$?
6. In Table 1, SFT+AdaBack on GSM8k 3B is actually worse than AdaBack on the base model (70.7 vs 73.3). Can you clarify this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Authors identify shortcomings in SFT and RL for sequential reasoning tasks:
- SFT: data collection is expensive and theoretically insufficient for certain symbolic tasks given insufficient training data
- RL: outcome rewards are sparse and the policy really gets meaningful signal when a successful trajectory is found which may be exponentially unlikely (in solution length)

Authors propose a middle ground between SFT and RL that uses RL training with problem contexts augmented with prefixes of SFT train data. Moreover, the proportion of the SFT solution revealed to the model in training is dynamically adjusted based on model performance.

Their method produces encouraging results on GSM8K and MATH.

### Strengths
Clear conceptual contribution:
- The paper identifies a meaningful regime between SFT and RL and proposes a principled mechanism (adaptive partial supervision) rather than a heuristic curriculum schedule. This framing is conceptually clean and well-motivated.

Per-sample adaptive curriculum is novel and well-justified:
- Unlike prior curriculum or backtracking approaches that rely on global schedules or coarse segmentation heuristics (e.g., R3), AdaBack adapts per-instance based on reward signals. This allows difficulty to scale automatically in heterogeneous datasets.

Empirical Analysis:
- The training dynamics figures, pass@k exploration analysis, and comparisons across both base and SFT-initialized models provide a clear and multi-angle understanding of how the method behaves. The results are presented in a way that makes the mechanism’s effects interpretable.

Promising results on reasoning benchmarks: On GSM8k (including the proposed Base-7 and Tensor-2 variants) and MATH, AdaBack produces consistent improvements over standard RL pipelines

### Weaknesses
- limited scale: the models explored in this work are limited to 3B parameters
  - The deepseek R1 report asserts that smaller models have lower capacity from benefiting from RL to discover novel reasoning patterns and are better suited towards SFT based distillation from stronger models (which somewhat addresses the stated drawbacks of SFT, if a teacher model is available to produce substantial distillation data)
  - this point is somewhat understandable if it's resource constrained, but the scaling behaviour of this approach still should be further studied

- mitigated impact on instruction-tuned models: the comparison between standard RL and AdaBack on SFT initialized models shows minor differences with slow learning overall. SFT mid/post training pipelines have become fairly standardized in most major model releases, so is the proposition to replace these SFT steps with AdaBack instead?

- reliance on ground truth rationales inherits SFT's data collection drawback

### Questions
- How does compute compare across these methods?
  - SFT has the benefit of not needing to sample new sequences, so having a cost controlled comparison might be instructive.

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
4

### Summary
This paper proposes AdaBack, an adaptive backtracking algorithm that dynamically adjusts the level of supervision during reinforcement learning training for reasoning tasks. By revealing partial prefixes of target solutions and adapting the supervision ratio based on reward feedback, AdaBack aims to bridge the gap between supervised fine-tuning and RL, enabling models to learn complex reasoning chains more effectively. The authors demonstrate its efficacy on synthetic tasks (e.g., chain-of-parity) and real-world benchmarks (MATH, GSM8k), showing improvements over standard RL and SFT+RL baselines, particularly in out-of-distribution settings.

### Strengths
Evaluations span synthetic tasks and real-world reasoning benchmarks, demonstrating broad applicability.
The method is well-motivated and clearly explained, with intuitive visualizations of the training process.

### Weaknesses
The approach closely mirrors R3’s reverse curriculum framework, which also uses partial demonstrations to create a curriculum. The distinction—adaptive per-sample scheduling versus fixed stages—is valuable but not thoroughly differentiated in terms of impact.
While outperforming vanilla RL, comparisons to R3 or other curriculum-based RL methods are absent, leaving the reader uncertain about the method’s advantages over existing alternatives.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AdaBack, an adaptive curriculum method for reinforcement learning that dynamically reveals partial solution prefixes (rationales) based on performance feedback. For each example, AdaBack maintains a supervision ratio $\rho$ that determines the proportion of the solution prefix shown to the model. The model is then trained using standard RL with a verifiable reward on these partial solutions. The supervision ratio is updated via a stochastic “binary search” around a reward threshold. AdaBack aims to bridge the gap between supervised fine-tuning (dense feedback) and reinforcement learning (sparse feedback) by maintaining a non-vanishing reward rate. Experiments on MATH, GSM8k, and the symbolic manipulated variants of GSM8k demonstrate that AdaBack outperforms GRPO, especially on tasks with sparse rewards.

### Strengths
- The proposed method improves model performance on mathematical datasets. In particular, it demonstrates greater benefits on the Base-7 and Tensor-2 GSM8k datasets—variants of GSM8k created by symbolic manipulations and effectively reduce the density of reward signals during training. These results align with the goal of addressing the sparse reward problem.

- The paper includes and analyzes negative results (e.g., saturation on MATH for Llama‑3.2‑3B‑Instruct and Qwen‑2.5), which increases the paper’s value for future work. 

- The proposed method is well-motivated and clear-explained.

### Weaknesses
- The paper does not specify how many random seeds were used to compute test accuracy. Given the high randomness in LM sampling, this omission raises questions about statistical robustness, especially for Table 1.
- The method learns a per-problem parameter $\rho$ and initializes it using a global EMA with hyperparameter $\alpha$, but the choice and sensitivity of $\alpha$ are not discussed. 
- Although the approach is inspired by R3, no baseline comparison with R3 is included on the mathematical datasets, leaving open how much of the improvement derives from the adaptive curriculum itself.
- The experiments are limited to Llama models. Prior studies indicate that model families like Qwen behave differently in RL fine-tuning. Thus, the generality of AdaBack’s gains remains uncertain.
- The method requires on ground-truth rationales, which constrains scalability and comparability to pure RL approaches that rely only on verifiers.
- (Minor issue) Some recent work on curriculum learning in RL fine-tuning, such as AdaRFT (Shi et al., 2025), SEC (Chen et al., 2025), and RORL (Bae et al., 2025), is not discussed in the related work section.

Shi, Taiwei, et al. "Efficient reinforcement finetuning via adaptive curriculum learning." arXiv preprint arXiv:2504.05520 (2025).

Chen, Xiaoyin, et al. "Self-Evolving Curriculum for LLM Reasoning." arXiv preprint arXiv:2505.14970 (2025).

Bae, Sanghwan, et al. "Online difficulty filtering for reasoning oriented reinforcement learning." arXiv preprint arXiv:2504.03380 (2025).

### Questions
- How many random seeds were used for the accuracy results in Table 1?
- What value of $\alpha$ was used for the global average initialization, and how sensitive is final performance to this choice?
- What is R3’s performance on the MATH and GSM8k datasets under the same setup?
- Have you tested AdaBack on other model families, such as Qwen, to evaluate its generalization? I understand that Qwen may have saturated the MATH dataset, but there are other, more challenging datasets where a dense reward could still be beneficial.

### Soundness
2

### Presentation
3

### Contribution
2
