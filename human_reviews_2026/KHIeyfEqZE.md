# Collaborative Dual-Size Large Language Models with Dual-Stage Deferral Risk Control

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 0, 4, 2

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities, yet ensuring their safe deployment remains challenging. Existing safety mechanisms, while effective against malicious inputs, often degrade performance on benign queries due to over-conservative strategies. We propose the \textbf{D}ual-size LLM collaborative framework with \textbf{D}ual-stage deferral risk contro\textbf{L} (\textbf{DDL}), which integrates lightweight and heavyweight models with calibrated deferral mechanisms. Our approach formalizes the safety–efficiency trade-off as a constrained optimization problem that jointly considers prediction accuracy, computational cost, and safety risk. We provide theoretical guarantees showing that our mechanism achieves distribution-free risk control while minimizing unnecessary heavyweight computation. Extensive experiments on three datasets demonstrate that DDL effectively balances safety and efficiency, achieving performance and safety metrics comparable to state-of-the-art safety-aligned models while reducing average inference time by more than 65\%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DDL (Dual-size Large Language Model framework with Dual-stage Deferral Risk Control), a collaborative architecture that integrates a lightweight and a heavyweight LLM to balance safety and efficiency during inference. The key insight is that safety–efficiency trade-offs in LLMs form a Pareto frontier, where improving one often degrades the other.

### Strengths
1. The idea of combining dual-size models in a risk-calibrated deferral pipeline is conceptually novel and theoretically grounded.
2. The contribution addresses a key bottleneck in scalable safe deployment: unnecessary use of large models for benign inputs.
3. The writing is professional, structured, and easy to follow.

### Weaknesses
1. Limited analysis of failure cases. While DDL provides strong average performance, the paper does not report worst-case or outlier failure scenarios.

2. Token-based classifier generality. The safety classifier’s reliance on a single special token embedding may not scale to more complex multi-turn or multilingual scenarios. An analysis of embedding robustness or potential expansion (e.g., multi-token attention pooling) would enhance credibility.

### Questions
See Weaknesses.

Besides, how does DDL behave under distribution shifts (e.g., new unsafe categories not in calibration data)? Would threshold guarantees still hold?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper talks about building a multi-LLM framework with a classifier that scores the harmfulness of inputs. It is very difficult to make sense of the model size, dataset size, classifier architecture, etc because of multiple contradictions within the literature. The paper spent quite a few pages on providing theoretical proofs of the two thresholds for the classification, but in practice, the thresholds seem to be identical after automatic calibration on three different datasets.

### Strengths
I apologize that I cannot say any good words about this paper.

### Weaknesses
There are multiple fatal errors or inconsistencies in the paper.

The paper contains multiple fatal and naive errors that might indicate that either the author is extremely negligent or the LLM is hallucinating. 

For example, 

1) 

On line 218, the author introduced their work as a "Privacy Detection Framework". However, the paper is never about privacy preservation, and the term "Privacy Detection" simply does not make sense. 

2) 

In multiple cases (line 374, Table 1, Figure 3, line 1127, line 1130, line 1132), the paper mentions a model called "Qwen-2.5-1B" or "Qwen-1B" which does not actually exist. In Table 1, I can see Qwen-2.5 0.5B, 1B, and 1.5B at the same time, which makes it hard to believe that it is a human error. 

3) 

The dataset size does not add up. 

In lines 307 and 309, the paper claims that there are 20,000 training samples using Claude that contain 10k safe and 10k unsafe queries.

However, in line 1000 and Table 4, the number has changed to 30,000 in total and 15,000 safe and 15,000 unsafe. And they now claim that these samples are not produced by Claude, instead those are 5,000 from MMLU, 5,000 from BoolQ, and 5,000 from WikiQA. 

The most ridiculous number appears in line 1065, where the authors claim that the classifier is trained with 230K samples. 

4) 

The Classifier architecture keeps changing

On lines 145-147 the paper clearly stated that a simple linear classifier is used with only two parameters (W and b). 

Then, in lines 1058-1059 the authors say "For the trainable token-based safety classifier, we utilize a base encoder with 24 transformer layers." I could only assume that they might be referring to Qwen-2.5-0.5B? Because Qwen-2.5-1.5B already has 28 layers. 

Then, in the next few lines (line 1059-1060), the authors mentioned that the model architecture actually consists of three fully-connected layers (2048 → 1024 → 512) with GELU activation, which is completely different from what is described in the equation in lines 145-147. 

The plot twist is not over yet. In line 1172, Table 6, when performing the efficiency analysis, I can see linear and 2-layer MLP again. This 2-layer MLP is never mentioned until the 22nd page of this paper. 

5) 

Citation Error

In line 168, when citing Deepseek-V3, the author first cited Guo et. al, which was the citation for DeepSeek-Coder. Then in line 323 the authors cited Liu et al. for Deepseek-V3, which is the correct citation for Deepseek-V3. Such inconsistency might still be a human error? 

6) 

Identical auto-calibrated thresholds for three different datasets

In line 407, Table 2, it appears that the optimal thresholds set by auto calibration are all 0.73 and 0.35 for three different datasets. Through literature (line 1089), I believe they might be doing a grid search over 6,000 uniformly spaced threshold candidates in [0.1, 0.9] and ∆ = 0.01? (BTW, it is also mathematically the wrong number of thresholds) Overall, it is implausible to find the exact same threshold for all three datasets through a grid search. 


At this point, I believe it is highly likely that no one actually conducted the experiments or wrote the paper themselves. I may find more issues if I keep digging. For example, the bounding of the loss in the equation seems off and it can go beyond [0,1], and the numbers in many tables are suspicious, but there is no point in looking into it anymore.

### Questions
**I respectfully request clarification:** 

Were these experiments genuinely performed as described?
If large language models were used to draft portions of this manuscript, I would kindly ask the authors to acknowledge this and ensure that all technical details have been carefully verified against the actual implementation.

If the authors did not perform the experiment or write the paper themselves, I kindly request that the authors apologize for the unprofessional behavior, as it wasted reviewers' hours of time and might have made them question their own sanity.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose DDL, a Dual-size LLM collaborative framework with Dual-stage deferral risk controL, including a trainable token-based safety classifier based on hidden state of LLMs and a optimal threshold selection method to expand the Pareto frontier between response latency and classification accuracy.

### Strengths
- The writing and presentation of the paper are excellent, with a clear and logical flow.
- The paper provides a finite-sample theoretical guarantee for its proposed method.

### Weaknesses
- **Threat model.** The author assume a white-box scenarios, which can access to the hidden representation of LLMs. However, many commercial scenarios are not open-resourced, such as GPT and Claude. Only considering the white-box scenarios restricts the applicability of the method in real world.
- **Experiment setting.** The token classification dataset is generated by Claude, which may not reflect real adversarial attacks. Could the author show some examples of the generated unsafe prompt to show that they are stealthy and compare them with the real world unsafe prompts to show that they have the same distribution.
- **Experiment results.** The core contribution of this method is the token embedding classifier. The author only compare classifier architectures in Appendix I, but no comparison with other classifier-based defense methods, such as the Moderation API by OpenAI. The author should further specify the advantages of the proposed white-box defense methods over the Moderation API.

### Questions
- Would the unsafe prompts generated by Claude can be directly refused by the target LLMs? It is unclear that the author accounted for this when generating the unsafe prompts. How can we ensure that the generated unsafe prompts are stealthy enough so that they are not immediately refused by the first LLM?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes DDL, a system designed to ensure LLM safety while maintaining computational efficiency.

The framework employs two models of different sizes: the smaller model first judges whether a query is safe. If the confidence is high, it directly answers; otherwise, the query is deferred to a larger model. For highly uncertain cases, a third model (or a human verifier) conducts the final verification.

Experiments are conducted on modified “safety-critical” versions of QA datasets, demonstrating that the proposed method outperforms single-model baselines in both safety and performance.

### Strengths
- The proposed framework is conceptually simple for safe LLM deployment.
- The paper provides theoretical justifications, offering distribution-free risk control guarantees for the dual-stage decision process.
- The design of a token-based safety classifier is efficient, avoiding full model fine-tuning.

### Weaknesses
- Unconvincing experimental results (Table 1): The DDL system uses the small model first and defers to the large model when the confidence is low. Therefore, its overall performance should intuitively fall between the small and large model results. However, Table 1 shows that DDL even outperforms the larger model, which seems implausible. This raises questions about the reliability of the reported results. If this gain is due to the additional verification by DeepSeek V3 (the M3 model), it is not a fair comparison, because the pure models cannot use such external verification. The authors should report how often M3 was invoked, and the results without this external verification. Additionally, efficiency should be measured in terms of FLOPs or total model computation cost, rather than response time, which is highly dependent on infrastructure.
- Lack of comparison with guard models: The paper does not compare DDL against common guard-model approaches (e.g., using a safety classifier before model inference). Small guard models like [1] can efficiently improve safety without changing the main model's output distribution, providing a fairer baseline.
- Potential degradation for safe but complex queries: Since delegation depends on the safety classifier's uncertainty, DDL might incorrectly classify safe but difficult questions (e.g., college-level math) as "safe" and let the small model answer, leading to significant performance drops on such cases. A more nuanced delegation policy (e.g., combining difficulty estimation with safety) may be necessary.

[1] Lee et al., HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models, ICLR 2025

### Questions
- Clarify in Table 1 which metrics are better when higher vs. lower values (use $\uparrow$/$\downarrow$ symbols).
- Report the percentage of samples handled by each model (M1, M2, M3).
- Add an ablation comparing DDL with standard guard-model pipelines.

### Soundness
2

### Presentation
1

### Contribution
1
