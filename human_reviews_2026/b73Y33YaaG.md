# Can Confidence Estimates Decide When Chain-of-Thought is Necessary for LLMs?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Chain-of-thought (CoT) prompting has emerged as a common technique for enhancing the reasoning abilities of large language models (LLMs). While extended reasoning can boost accuracy on complex tasks, it is often unnecessary and substantially increases token usage, limiting the practicality of reasoning models in many scenarios. 
Recent models, such as GPT-OSS and Qwen3, expose controls that enable users to adjust the length of CoT or determine whether it is used at all. Yet, it remains unclear when CoT should be used: on some tasks it improves performance, while on others it provides little benefit or even harms performance. We address this challenge with confidence-gated CoT, where a model invokes reasoning only when confidence in its direct answer is low. To this end, we present the first systematic study of training-free confidence estimation methods for CoT gating. Specifically, we evaluate four training-free confidence estimation methods and compare them to a random baseline and an oracle that always knows when CoT is needed. Through extensive experiments, we show that existing training-free confidence measures can reduce redundant CoT and outperform randomly invoked CoT. However, the utility of individual confidence measures is inconsistent, varying with both the dataset and the model, underscoring the difficulty of deploying confidence-gated CoT in practice. By analysing both strengths and failure modes, our study highlights the potential and limitations of current methods and paves the way toward more reliable adaptive gating of CoT.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces and analyzes a confidence-gating method for dynamically deciding when CoT reasoning is necessary for LLMs. Specifically, the model first generates a direct answer and then estimates its own confidence using training-free signals, such as perplexity, P(True), margin sampling, and verbalized confidence. Based on this confidence score, a gating threshold determines whether to accept the direct answer or to invoke CoT. Experiments are conducted across models and tasks to discuss the effectiveness and limitations of the idea of using confidence to determine the use of CoT.

### Strengths
1. In general, the writing of the paper is clear and easy to follow.
2. The focus of the paper, on how to determine the necessity of using CoT, is meaningful and interesting. 
3. Multiple tasks are considered in the experiment section, which provides extensive results on how confidence-gated CoT behaves across different reasoning types and model scales.

### Weaknesses
1. A major concern is that the positioning of the paper is somewhat unclear.
While it introduces a method for confidence-based control of CoT reasoning, the proposed approach fails to consistently demonstrate strong performance across models and tasks. As a result, the paper shifts its focus toward analyzing whether confidence estimates can be used to decide when CoT is needed. However, since the analysis is conducted only on this specific implementation, the conclusions are method-specific and lack generality or broader applicability.
2. The novelty of the work might be limited. The idea of applying confidence to improve the efficiency of reasoning during inference has been applied and analyzed in multiple works [1][2][3]. While the implementation here is not identical to previous approaches, the conceptual contribution appears incremental, and the methodological design does not seem to introduce substantial innovation beyond existing frameworks.
[1] Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning
[2] Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models
[3] Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs
3. In some experiment sections, the authors mainly describe the results of the experiments without providing a comprehensive analysis to discuss the reasons behind the phenomenon and the corresponding insights. For example, why does the performance of the confidence-gating method perform better in certain tasks but worse in others? Why do P(true) and verbalized confidence sometimes (even with capable models) perform worse than randomly deciding the usage of CoT?

### Questions
1. In the paper, the authors mentioned that a calibration set is used for selecting the optimal threshold. Given that a calibration set can be used, why is it necessary to develop an online evaluation? Is there a clear comparison between the calibration-based Pareto threshold and the dynamic percentile thresholding?
2. Why do some lines in the trade-off plots (e.g., perplexity or margin confidence) contain very few points, especially at higher CoT-usage rates?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies confidence-gated CoT, a simple method that lets LLMs decide when to use reasoning based on training-free confidence signals. Using GPT-OSS and Qwen3 models, the authors show that these signals can cut unnecessary CoT usage and token cost while keeping accuracy nearly unchanged. However, the results are uneven across models and tasks, showing promise but also limitations in the reliability of the current confidence measure.

### Strengths
1. The paper tackles a clear and practical problem, deciding when CoT reasoning is necessary through an intuitive confidence-based gating framework.
2. The experiments are systematic and broad, covering multiple models and reasoning benchmarks with solid comparisons to random and oracle baselines.
3. Results demonstrate meaningful token savings with minimal accuracy loss, showing practical value for efficient reasoning.

### Weaknesses
1. I am concerned that the contribution of this paper is limited. If the goal is to propose an efficient reasoning method, then the performance is not good enough and there are many efficient CoT baselines to be compared with. If the goal is to simply analyze the confidence-gating, then the scope is way too narrow, as effectively measuring the confidence of LLM generation itself is still a problem to be solved. The insights from this paper are limited. 

2. The proposed confidence signals exhibit substantial variance across datasets and model scales, suggesting that their reliability depends largely on the underlying calibration quality of each model rather than on any consistent principle of confidence estimation. This is critical as if wrong confidence is used, the confidence-gating is even worse. This restricts the practical usage of confidence-gating methods.

3. The generalization of the thresholding method is questionable. Selecting the Pareto-optimal threshold relies on the calibration dataset, which may not generalize well to out-of-distribution data. This further indicates that the confidence-gating method studied in this work is far from mature and practical.

### Questions
See weakness.

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
This paper investigates whether the model confidence can decide when to invoke chain-of-thought (CoT) reasoning in large language models (LLMs). The authors study a training-free, confidence-gated CoT framework that answers directly when confidence is high and triggers CoT when confidence is low. Through offline and online experiments across multiple models and datasets, they show that confidence-gated CoT achieves the best trade-off between  accuracy and inference cost (in terms of tokens). The analysis also examines calibration (ECE/AUROC) of several confidence signals—verbalized scores, normalized perplexity, margin, and P(True)—and identifies strengths and failure modes.

### Strengths
- The paper addresses a timely and relevant problem and is supported by solid empirical evaluations.  
- The writing is clear, well-organized, and easy to follow.

### Weaknesses
- **Oracle and CoT routing interpretation** In Figures 2 and 3, the oracle triggers CoT only when a direct answer is incorrect (line 286). However, the gap between “always using CoT” and the oracle is very large, suggesting that CoT severely harms performance when applied indiscriminately. This raises a key question: does the oracle use only *one round* of CoT, matching other baselines? If so, the results may indicate that *offline CoT routing itself is ineffective*, which deserves clearer discussion.  
- **Claim vs. observed trends** Line 313 states that “Commonsense, soft reasoning, and knowledge tasks benefit the most from confidence-based gating.” However, Appendix E shows that many soft-reasoning and knowledge tasks do *not* exhibit strong gains, and several curves (e.g., Figure 9) resemble math-like trends rather than commonsense improvements. Further analysis is needed to justify the claim and explain task-dependent behaviors.  
- **Uncalibrated confidence undermines core comparisons** The main results rely on confidence-based thresholding, yet verbalized confidence is known to be poorly calibrated in LLMs without additional training. If model confidence is uncalibrated, thresholding becomes nearly equivalent to random selection, making the “random gating” baseline an insufficient point of comparison. To support the central claims, the paper should include results with *calibrated* confidence (e.g., confidence tuning, post-hoc calibration, or temperature scaling).  
- **Lack of practical guidance for deployment** Although confidence-based gating is evaluated extensively, the results vary significantly across tasks and models, making the method difficult to operationalize in real-world or online settings. The paper would be substantially strengthened by offering *practical guidelines*—e.g., which confidence metric works best for which task types, model families, or prompt formats.  
- **External validity beyond hybrid CoT models** The experiments are limited to hybrid “thinking-enabled” models. It remains unclear whether the proposed gating strategy generalizes to standard LLMs where CoT is elicited only through prompting. Ablations using different prompting-based CoT elicitation methods would help establish broader applicability.

### Questions
- **Missing data points in plots** In Figures 2 and 3, several points for the *margin* (orange) and *perplexity* (green) curves appear to be missing. Please clarify whether these are absent due to filtering, instability, or another implementation detail.

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
4

### Summary
This paper investigates whether confidence estimates from large language models (LLMs) can effectively determine when to employ Chain-of-Thought (CoT) reasoning. It systematically evaluates four training-free confidence methods for gating CoT, aiming to reduce unnecessary token consumption while maintaining task accuracy. The results demonstrate that these confidence measures can curtail redundant CoT tokens, yet their effectiveness varies significantly across different models and tasks. The study concludes that while confidence gating shows promise for efficient reasoning, current methods exhibit too much inconsistency for robust real-world deployment.

### Strengths
1. The calibration of verbalized confidence in LLMs is a critical research topic, and this study has significant practical implications.

2. The experimentation is comprehensive and in-depth, thoroughly demonstrating that the proposed methods can simultaneously reduce token usage while preserving accuracy.

3. The proposed confidence-gating pipeline is intuitive and elegant, offering a straightforward approach to effectively address the important problem of deciding when a model should engage in "fast" (direct) versus "slow" (CoT) reasoning.

### Weaknesses
1. While the paper introduces a promising research direction and evaluates several feasible methods, it lacks a unified or theoretically grounded framework, which limits its theoretical contribution.

2. In Figure 2, several confidence estimation methods underperform the random baseline. A theoretical analysis from the authors explaining why and under what conditions these methods fail would significantly strengthen the paper.

3. Some important related work for LLM confidence calibration is not discussed, e.g., [1,2].

[1] Xiong, M., Hu, Z., Lu, X., et al. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. ICLR 2024. 
[2] Zhou Z, Jin T, Shi J, et al. SteerConf: Steering LLMs for Confidence Elicitation. NeurIPS 2025.

### Questions
1. I am highly interested in whether this approach remains effective on state-of-the-art, closed-source models (e.g., future models like GPT-5 or Gemini 2.5 Pro) in terms of reducing token usage while maintaining accuracy. Experimental results on CoT gating based on verbalized confidence for these models would strengthen the work.

2. Could this methodology be extended to the ReAct framework? Tool-use paradigms also face a similar challenge, where simple tool calls may not require reasoning, whereas complex ones do. Experimenting on tool-using benchmarks, such as $\tau^2$-bench, could reveal further interesting phenomena.

### Soundness
3

### Presentation
4

### Contribution
2
