# Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Large Language Models (LLMs) using Chain-of-Thought (CoT) prompting excel at complex reasoning but generate verbose thought processes with considerable redundancy, leading to increased inference costs and reduced efficiency. We introduce a novel CoT compression framework based on step entropy, a metric that quantifies the informational contribution of individual reasoning steps to identify redundancy. Through theoretical analysis and extensive empirical validation on mathematical reasoning benchmarks, we demonstrate that steps with low entropy are indeed highly redundant. Our experiments reveal that an astonishing 80% of low-entropy intermediate steps can be pruned without significant degradation in the final answer accuracy across DeepSeek-R1-7B, 14B and Qwen3-8B. This finding sharply contrasts with random or high-entropy pruning, which severely impairs reasoning performance. Building on this, we propose a novel two-stage training strategy combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) reinforcement learning. This approach enables LLMs to autonomously learn to generate compressed COTs during inference by strategically incorporating [SKIP] tokens. Our method significantly enhances LLM inference efficiency while rigorously preserving accuracy, offering profound implications for practical LLM deployment and a deeper understanding of reasoning structures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes **step entropy** as a signal to identify and prune redundant segments within Chain-of-Thought (CoT) traces, aiming to make slow “deliberate” reasoning faster without sacrificing accuracy. Concretely, the step-level entropy for the \(i\)-th step \(S_i\) is defined as the sum of token entropies conditioned on the prior context, \(H(S_i \mid S_{<i})=\sum_j H(t_{i,j}\mid c_{i,j})\). The core hypothesis is that **low-entropy steps contribute little information** to the final answer and can be safely skipped. The authors present (i) an information-theoretic motivation; (ii) a pruning recipe that removes the lowest-entropy steps and replaces them with a special token (e.g., [SKIP]); and (iii) a training pipeline (SFT and GRPO) to teach models to emit compressed CoTs during inference. Empirically, they report substantial token reductions (often 16–57%) with modest accuracy degradation on math-style reasoning benchmarks.

### Strengths
- **Conceptual clarity:** A clear, information-theoretic criterion (step entropy) with an intuitive link to redundancy.  
- **Granularity choice:** Results indicate step-level pruning is more effective than naïve token-entropy pruning, suggesting the *step* is a useful unit.  
- **Practical interface:** Using a placeholder like [SKIP] makes the compression operationally simple; ablations on replacement strategies are helpful.

### Weaknesses
1. **Autoregressive dependency not directly addressed**: Evidence is largely post-hoc (compress after generating a full CoT). In AR decoding, earlier “redundant” content can steer later tokens; removing it afterward does not prove it can be skipped causally during generation. 
2. **Low-Entropy Steps Pruning shows no pre-training practical gain; current use is post hoc.** As implemented, “Inference with Compressed CoT” is applied **after generating the full CoT**, so it yields no acceleration and provides limited practical value before additional training. To make this genuinely efficient without extensive post-training, the method should be reframed as an inference-time control。
3. **Attribution vs. training data/compute:** The main contribution is both a compression rule and a data-construction pipeline (e.g., ~130k compressed pairs). Without baselines trained on identical data with matched optimization budgets, it’s hard to attribute gains to step entropy rather than more/better post-training.  
4. **Fixed compression ratio:** A static global pruning rate (e.g., “up to 80%”) ignores that redundancy varies by instance difficulty, dataset, and model size; no mechanism adapts compression per-instance.  
5. **Step segmentation heuristic:** Steps defined by formatting (e.g., `\n\n`) can be brittle. The paper does not validate segmentation accuracy or analyze sensitivity to finer/coarser granularity or token-entropy sparsity.

### Questions
1. **Causal necessity at inference:** Can you run **inference-time interventions** that compress the low-entropy steps while holding other decoding parameters fixed, and report accuracy relative to full-CoT? This would directly test whether those steps are unnecessary *causally* rather than *post hoc*.  
2. **Fair baselines on identical data/compute:** Train strong baselines (e.g., token/chunk compression, rule-based CoT pruning) on the same ~130k instances with the same training budget. Do your gains persist under these controls?  
3. **Difficulty-aware adaptivity:** Can the compression ratio be predicted per instance (e.g., via a learned controller that thresholds entropy or estimates a target depth \( \kappa(x) \))? 
4. **Segmentation robustness & granularity:** How exactly are “steps” defined and validated? Have you analyzed token-entropy distribution and how aggregation affects pruning decisions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces step entropy, an information-theoretic metric to quantify each reasoning step’s contribution in LLM Chain-of-Thought. By pruning up to 80% of low-entropy steps, the method reduces tokens by 16–57% with minimal accuracy loss across DeepSeek-R1 and Qwen3 models. A two-stage training framework combining SFT and GRPO further enables models to autonomously skip redundant steps via [SKIP] tokens. The approach outperforms prior CoT compression methods, offering an efficient and interpretable way to accelerate LLM reasoning.

### Strengths
1. The method is simple and easy to implement, requiring only entropy calculation and pruning based on low-information steps to construct the pruned CoT. It achieves strong performance, such as maintaining accuracy on Math500 even with a 30% compression ratio, demonstrating both effectiveness and efficiency.
2. The introduction of the SFT+RL framework makes the approach more practical. By allowing the model to learn when to skip redundant steps automatically, it extends the static compression method into a trainable and deployable solution.

### Weaknesses
1. The segmentation and granularity of reasoning steps are not rigorously defined. The approach relies on manually designed delimiters like \n\n, which may not generalize well across datasets or model architectures.
2. The definition of step entropy as the sum rather than the average of token entropies could bias the metric toward longer steps, potentially misrepresenting their true informativeness.
3. The presentation of Table 1 is not so good. A clearer organization would make the results easier to interpret.

### Questions
Please see the weaknesses.

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
This paper proposes a method to prune out extra reasoning steps in reasoning models, and then SFT+RL with the result to allow for more concise chains of thought.

### Strengths
1. Clarity: The paper is generally written pretty clearly, and the method is not very complex which is nice. I would say that the writing is a little repetitive (saying the same thing many times), but not to the point where it makes the paper hard to understand.
2. Significance: It seems that the method, while simple, is reasonably effective. It greatly compresses the resulting CoT at a reasonable loss in accuracy.

### Weaknesses
1. To be honest, the method feels rather "hacky" to me, inserting skip tokens based on heuristics. My feeling is that the community in general is trying to move towards methods that perform end-to-end RL in a more principled way rather than these sorts of processes.
2. Relatedly, while this paper proposes methods to compress chains of thought, there are other methods to directly control the length of chains of thought such as L1 (Aggarwal and Welleck). These seem simpler, more elegant, and can also be retro-fitted onto existing models. I was surprised that there was no discussion of this work, and it seems like it would be a competitive baseline.

L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning
Pranjal Aggarwal, Sean Welleck

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes step entropy as a principled, per-step information measure for CoT, shows that pruning low-entropy steps preserves accuracy while cutting tokens, and trains models to self-compress via SFT + GRPO with an explicit [SKIP] token. Theory upper-bounds each step’s information by its entropy (Lemma/Theorem 1), and experiments across DeepSeek-R1 and Qwen families show strong accuracy–efficiency trade-offs.

### Strengths
1. Using entropy as information proxy is clean. The paper defines step entropy by summing token entropies within a step and proves the conditional information contribution is bounded by this entropy, offering clear intuition for “skip low-entropy steps.” This is simple, transparent, and theoretically motivated.

2. Theorem 1 provides usable intuition. Bounding the information of any subset of steps by the sum of their entropies gives a direct justification for step-level pruning rather than token-level pruning. The token-vs-step ablation empirically supports this semantic unit choice.

3. Fine-tuning setup is sensible and practical. The two-stage SFT → GRPO pipeline, rewarding correctness and compression while discouraging degenerate [SKIP] flooding, is clear and leads to better compression than static pruning on hard sets (e.g., AIME 2024).

4. Empirical results are good across models/benchmarks. They show ~30–55% token reductions with minimal accuracy loss; on some tasks accuracy even improves. Cross-architecture results (DeepSeek-R1-7B/14B, Qwen3-8B) and comparisons to recent compression methods are solid.

### Weaknesses
1. Step segmentation heuristic.
Steps are segmented using simple newline heuristics. While this works reasonably, it can blur step boundaries or merge logically distinct thoughts. A sensitivity study with sentence-based or LLM-predicted segmentation would improve robustness.

2. Fixed 80% pruning threshold.
The global 80% rule is justified empirically but may not generalize across datasets or reasoning styles. An adaptive or learned κ could better reflect per-problem difficulty.

3. Unnormalized entropy may bias toward longer steps.
The paper uses total (unnormalized) entropy per step. While this matches the theoretical bound, longer steps automatically accumulate more entropy even when per-token uncertainty is low, potentially biasing the pruning policy. A length-normalized or mixed variant could help disambiguate whether information or verbosity drives retention.

4. Scope of benchmarks.
Most experiments center on math and logic tasks. Including one additional open-ended domain (commonsense, code, or writing) would broaden the evidence that entropy-guided compression generalizes.

### Questions
1. Entropy normalization:
Did the authors try normalizing entropy by step length (e.g., average or log-length scaling)? If so, how did this affect correlation with information contribution?

2. Alternative to entropy-based labeling:
Instead of relying purely on entropy, have the authors tried using an LLM itself to label which steps are informational or non-trivial (e.g., “steps that advance reasoning” vs. “repetitive or obvious steps”)? Such annotations could provide a complementary supervision signal for training or validating entropy thresholds.

3. Fine-tuning stability:
During GRPO fine-tuning, how sensitive are results to the [SKIP] penalty coefficient? Does the model ever collapse to always skipping or never skipping?

4. Adaptive threshold:
Could the target entropy ratio κ be dynamically chosen based on per-question entropy distribution?

### Soundness
3

### Presentation
4

### Contribution
3
