# A Fine-Grained Analysis of Pure Semantic Preference Alignment in Large Language Models

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Large language models (LLMs) are typically aligned with human preferences through methods such as direct preference optimization (DPO). While empirically successful, these approaches face well-known limitations, including length bias, reward hacking, binary preference assumptions, and the aggregation of heterogeneous preferences into a single scalar signal. In this work, we take an inverse perspective: rather than attempting to resolve these issues, we investigate an idealized setting, which we call the *pure semantic preference scenario*, where such confounding factors are absent. We show that even in this idealized setting, existing alignment methods still do not fully capture the preference. Our analysis further reveals that (i) on-policy algorithms align more effectively, (ii) models trained without an explicit reference model perform better, and (iii) preference-model–based approaches consistently outperform reward-model–based approaches. Motivated by these observations, we introduce *preference matching optimization* (PMO), a DPO-type method that admits a closed-form solution and provably better approximates the true preference distribution. Experiments on both practical and idealized settings demonstrate that PMO achieves comparable performance with existing alignment methods in the practical setting, while offering stronger theoretical grounding and better performance in the pure semantic setting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper looks at the behavior of preference alignment algorithms in an idealized "pure semantic preference scenario," which is a synthetic dataset designed to isolate semantic choices from confounding factors like length bias, reward hacking, and binary preference assumptions. In practice, this is a synthetic dataset constructed so only one word is changed between the two options.

The authors show that even in this controlled setting, existing alignment methods (like DPO and PPO-based RLHF) do not optimally recover the ground-truth probabilistic preferences. Their analysis highlights three findings: (i) on-policy algorithms are more effective at this task, (ii) reference-free methods (like SimPO) perform better than reference-based ones (like DPO), and (iii) what they term "preference-model-based" approaches (e.g., DPO) outperform "reward-model-based" ones (e.g., PPO-RLHF).

The paper identifies a preference-accuracy trade-off, where methods that better align with the probabilistic preferences on the synthetic task show reduced accuracy on standard knowledge benchmarks (e.g., MMLU), and vice-versa. The authors attribute this trade-off to the distorting influence of the reference model (pi_ref) used in methods like DPO.

To address this, they propose Preference Matching Optimization (PMO), a new DPO-style objective. It is derived from an RL objective that combines both KL-divergence (to pi_ref) and entropy regularization. The resulting closed-form loss (Eq 5) resembles DPO but with an attenuated reference model term, which should allow PMO to balance probabilistic preference matching with accuracy.

### Strengths
- An initial analysis on the simple, synthetic dataset is a good scientific approach to remove confounders
- The paper clearly identifies preference-accuracy trade-off using their synthetic task.
- The proposed PMO method is well-motivated. It is not an ad-hoc objective but is derived (Proposition 4.2) from a clear RL objective (Eq 4) that explicitly combines entropy and KL regularization.
- The ablation studies on the α (entropy) and β (KL) hyperparameters (Table 1 and 3) clearly show how they affect accuracy and KL metrics

### Weaknesses
- The paper evaluates "accuracy" using knowledge-based, multiple-choice benchmarks (MMLU, ARC, HellaSwag). These benchmarks are not great for evaluating the alignment performance of an algorithm. An alignment algorithm's success is measured by its ability to follow human preferences in generative, open-ended tasks. The authors should have evaluated their method on standard alignment benchmarks like AlpacaEval2 or ArenaHard. Without this, we only know that PMO doesn't degrade knowledge task performance, but we have no evidence that it leads to a better-aligned model in practice, which is the entire goal.

- The core idea of using entropy regularisation is not new. SimPO, which the authors compare against, is a reference-free model, which Proposition 4.1 shows is equivalent to a maximum-entropy formulation. The authors also discuss H-DPO in the appendix (and mention it in the main text as well), which explicitly adds an entropy term to the DPO objective. The main contribution seems to be the specific αH + βKL formulation. Given the prior work, it is unclear if this specific variant is a significant enough contribution over existing methods like SimPO and H-DPO.

- The finding that on-policy algorithms perform better is acknowledged by the authors as mirroring "broader evidence" (line 311), though this evidence is not cited.

- The preference-accuracy trade-off is also discussed in papers like [1], which also investigates the role of entropy. The discussion on the role of regularisation (Section 4.2) would benefit from citing work like [2] where similar discussions are already present.

[1] Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints, https://arxiv.org/abs/2309.16240

[2] Meta-Learning Objectives for Preference Optimization, https://arxiv.org/pdf/2411.06568

### Questions
- Why did the authors choose to evaluate accuracy on knowledge-based benchmarks (MMLU, etc.) instead of established alignment benchmarks (AlpacaEval2, ArenaHard)? The latter seems far more relevant for a paper proposing a new alignment algorithm.

- Distinction from H-DPO: The appendix (B.2) mentions H-DPO but claims the "underlying principles differ." However, the H-DPO objective (Eq 8 in their paper) and PMO's RL objective (Eq 4) both combine a reward term, an entropy term, and a KL/cross-entropy term. Why do the principles differ?

- The paper distinguishes between "preference-model-based" (DPO-like) and "reward-model-based" (PPO-RLHF) approaches. This terminology is not standard. A brief, explicit definition in the introduction would be helpful.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an idealized evaluation setting for preference alignment, termed the Pure Semantic Preference (PSP) scenario. The goal is to isolate semantic alignment quality from confounding factors such as response length, syntactic variations, and stylistic biases. Within this framework, the authors systematically analyze the behavior of existing alignment algorithms. To address the limitations observed, they propose a new method called Preference Matching Optimization (PMO), which integrates entropy regularization with a KL-divergence–based objective to better approximate the target preference distribution. Experimental results suggest that PMO achieves comparable performance to existing methods while offering improved probability fidelity and reduced preference collapse (PCI) in the PSP setting.

### Strengths
By analyzing alignment behavior under a controlled “semantic-only” setup, the paper offers valuable insights into what current preference optimization methods are actually learning.

### Weaknesses
1. All experiments are conducted on relatively small models (~1B parameters). It remains unclear whether the findings hold for larger models such as 7B or 14B.

2. While PPO and NashMD are briefly mentioned, the paper focuses primarily on off-policy approaches. The implications for practical on-policy RLHF remain underexplored.

3. The paper emphasizes PCI as a major issue, but it is not evident that preserving uncertainty is always desirable—certain tasks may actually benefit from more decisive behavior.

4. Lack of evidence linking KL improvement to preference quality. Although PMO shows better KL/PCL metrics, it is not demonstrated that these necessarily translate to improved human-aligned outcomes. It would strengthen the paper to show in which types of semantic preference tasks PMO better captures human intent.

5. On real-world benchmarks, the authors only report accuracy. Including additional measures would give a more complete view of alignment performance.

### Questions
see the Weaknesses.

### Soundness
3

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
This paper examines the fundamental limits of preference-based alignment methods like DPO under an idealized pure semantic preference setting, where confounding factors such as length bias and heterogeneous rewards are removed. The authors find that even in this setting, existing methods fail to fully capture true preferences. They show that on-policy training, removing the reference model, and using preference models lead to better alignment, and propose Preference Matching Optimization (PMO). PMO is a closed-form, DPO-style algorithm that more accurately approximates the true preference distribution.

### Strengths
1. The writing is clear and easy to follow.

2. PMO offers provable advantages and interpretable formulation.

3. Provides new insights into key factors affecting alignment.

### Weaknesses
1. The authors mention that their main focus is on reward hacking issues for RLHF tasks like length bias. It's unclear to me how to resolve the reward hacking issues.

2. I think it’s difficult to conclude that reliance on a reference model directly leads to differences in preference alignment or accuracy from Figure 2. A more convincing approach would be to control for models with comparable accuracy and then evaluate other alignment-related metrics.

3. Moreover, since entropy and KL divergence losses are standard components in reinforcement learning algorithms, the technical contribution of this work appears somewhat limited.

### Questions
Please refer weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the concept of a pure semantic preference setting, aiming to analyze how preference alignment methods like RLHF and DPO behave when non-semantic confounders (e.g., length, style, and structure) are removed. The authors propose a new variant called Preference Matching Optimization (PMO), which combines entropy and KL regularization in a DPO-style objective to better approximate human preference distributions.

### Strengths
1. The paper is well-written and logically organized, and the “pure semantic preference” idea is clearly articulated. It provides a clean testbed to isolate semantic alignment factors.

2. The derivation of PMO is mathematically sound and shows a solid understanding of the relationships among DPO, SimPO, and RLHF.

3. The experiments demonstrate a recurring trade-off between alignment and accuracy, lending support to the authors’ theoretical analysis.

### Weaknesses
1. The *pure semantic preference* scenario is so contrived that it provides limited insight into realistic preference alignment. Real-world human judgments involve complex, multi-dimensional signals (tone, politeness, factuality, etc.), none of which are modeled here. As a result, the practical implications of the findings are unclear.

2. The paper evaluates only small models (1B–3B), using synthetic datasets rather than genuine human preference data. No large-scale or qualitative studies are presented, making it difficult to assess whether PMO offers any tangible advantages in actual alignment pipelines.

3.  The claimed benefits of PMO over existing methods (e.g., DPO, SimPO, CPO) are small and often within noise range. The benchmark experiments show similar or even slightly worse results in some cases, suggesting that PMO’s advantages may be overstated.

4. PMO essentially interpolates between existing regularization techniques (entropy + KL), offering an incremental extension rather than a fundamentally new approach. The paper might be better suited as an empirical note rather than a full ICLR paper.

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2
