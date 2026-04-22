# Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks by employing test-time scaling. However, they often generate over-long chains-of-thought that, driven by substantial reflections such as repetitive self-questioning and circular reasoning, lead to high token consumption, substantial computational overhead, and increased latency without improving accuracy, particularly in smaller models. Our observation reveals that increasing problem complexity induces more excessive and unnecessary reflection, which in turn reduces accuracy and increases token overhead. To address this challenge, we propose Adaptive Reflection and Length Coordinated Penalty (ARLCP), a novel reinforcement learning framework designed to dynamically balance reasoning efficiency and solution accuracy. ARLCP introduces two key innovations: (1) a reflection penalty that adaptively curtails unnecessary reflective steps while preserving essential reasoning, and (2) a length penalty calibrated to the estimated complexity of the problem. By coordinating these penalties, ARLCP encourages the model to generate more concise and effective reasoning paths. We evaluate our method on five mathematical reasoning benchmarks using  DeepSeek-R1-Distill-Qwen-1.5B and  DeepSeek-R1-Distill-Qwen-7B models. Experimental results show that ARLCP achieves a superior efficiency-accuracy trade-off compared to existing approaches. For the 1.5B model, it reduces the average response length by 53.1% while simultaneously improving accuracy by 5.8%. For the 7B model, it achieves a 35.0% reduction in length with a 2.7% accuracy gain. The code is released at https://github.com/ZeweiYu1/ARLCP.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of computational inefficiency in Large Reasoning Models (LRMs) by identifying and targeting a specific failure mode over reflection, which is a pattern of repetitive and unproductive reasoning. The authors propose a novel reinforcement learning framework, Adaptive Reflection and Length Coordinated Penalty (ARLCP), designed to train models to produce more concise yet accurate reasoning paths.  It proposes a method based on empirical evidence linking over-reflection to incorrect answers, simultaneously improving reasoning accuracy while significantly reducing response length.

### Strengths
1. The paper offers an insightful framing of the efficiency problem by identifying over-reflection as a distinct, harmful behavior. The empirical analysis shows that incorrect answers correlate with higher reflection and token counts, providing a solid motivation for penalizing this behavior to improve both accuracy and efficiency.

2. The ARLCP framework is elegant and intuitive. Using the model’s own Reflection Token Count (RTC) as a real-time complexity proxy is a smart, self-contained design that removes the need for external classifiers. The adaptive penalty that balances reflection and response length based on complexity is a well-motivated and effective idea.

3. The experiments are thorough and convincing. ARLCP consistently outperforms strong baselines across multiple reasoning benchmarks and model sizes. The 1.5B model’s 5.8% accuracy gain with a 53% length reduction is a particularly impressive result.

4. Ablation studies and OOD tests (e.g., MMLU) demonstrate robustness and transferability. The results suggest that over-reflection is a general issue in reasoning models and that ARLCP provides a practical, broadly applicable mitigation.

### Weaknesses
1. The reliance on RTC as a proxy for complexity is a fragile heuristic. It can misclassify simple but confusing problems and, paradoxically, penalize the most reflective (and supposedly complex) cases most heavily—undermining the intended adaptivity.

2. The use of a manually curated list of ``reflection keywords'' is brittle and easily gamed. Models can paraphrase to avoid penalties, making this more of a short-term engineering fix than a scalable solution.

3. The paper introduces several crucial hyperparameters (e.g., $n_1, n_2, n_3, \lambda, \alpha$) that drive the penalty mechanism, but no sensitivity analysis is provided. This limits confidence in the method’s robustness and reproducibility.

4. The justification for using RLOO is incomplete. While alternatives are said to be unstable, the paper doesn’t discuss RLOO’s known high variance and potential training instability. Clarifying this trade-off would strengthen the technical argument.

### Questions
1. The paper states the goal is to "permit more extensive reasoning for complex ones," but the implementation penalizes reflection most heavily on problems with the highest RTC. Can you clarify this apparent contradiction between the stated goal and the reward mechanism? 

2. The reflection penalty relies on a fixed keyword list. Have you analyzed whether the model learns to circumvent this penalty using synonyms, and how do you see this lexical approach scaling to other models with different linguistic patterns?

### Soundness
3

### Presentation
4

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
The authors observe that reasoning models often over-reflect, producing repetitive self-questioning and circular reasoning that lead to excessive token usage and computational overhead. These reflective behaviors are typically signaled by words such as wait, hmm, or check again. Their analysis shows that as problem complexity increases, models tend to generate more unnecessary reflection.
To address this, the authors propose estimating problem difficulty based on the frequency of reflection keywords. They introduce a reinforcement learning framework that adaptively applies both a reflection penalty and a length penalty. This approach encourages efficient reasoning by discouraging redundant thought patterns while maintaining problem-solving effectiveness.
Experiments using DeepSeek-distilled models demonstrate reductions in reasoning length and improved accuracy compared with existing methods.

### Strengths
1. The idea of estimating problem difficulty through the frequency of reflection keywords is novel and interesting.
2. The method is well-motivated, conceptually sound, and shows empirical effectiveness.
3. The writing is clear and the presentation is good.

### Weaknesses
1. The method may not generalize well to other model families. It assumes that reflection behaviors are expressed through specific keywords, which is a strong and model-dependent assumption. It would strengthen the paper if the authors extended the analysis of reflection-token counts versus problem difficulty to other reasoning model families (GPT-OSS, Qwen 3 Thinking, etc.).
2. All experiments are conducted on a single model family (DeepSeek-distilled Qwen), limiting the generality of the findings.
3. The paper's contributions are incremental. Prior studies [1,2] have already established the correlation between reflection tokens and reasoning performance, and existing works [3,4] have explored the use of length penalties to encourage concise reasoning.

References

[1] Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning

[2] s1: Simple test-time scaling

[3] ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models

[4] Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning

### Questions
1. Does the proposed method also enable a non-reasoning model to acquire reasoning ability? The experiments only evaluate reductions in reasoning length for models that already possess reasoning capability.
2. The authors demonstrate that the adaptive penalty works effectively with RLOO. Would the approach generalize to other reinforcement learning techniques?

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
4

### Summary
This paper identifies the problem of **over-reflection** in Large Reasoning Models, where repetitive self-questioning causes inefficiency without accuracy gain. It proposes **ARLCP (Adaptive Reflection and Length Coordinated Penalty)**, a reinforcement learning method that dynamically adjusts reflection and length penalties based on problem complexity. Experiments on several math benchmarks show that ARLCP significantly reduces reasoning length while improving accuracy, achieving a better efficiency–accuracy balance than previous length-penalized approaches.

### Strengths
**Strengths**:  
1. The paper is well-written and clearly structured, making it easy to follow.  
2. It identifies a practical issue—over-reflection in large reasoning models—and provides a targeted solution.  
3. The proposed ARLCP method is simple yet effective, adaptively balancing reflection and reasoning length.  
4. The experiments on multiple benchmarks demonstrate clear improvements in both accuracy and efficiency.

### Weaknesses
typo line 448: analysis -> analyze

Weaknesses:  
1. The method appears potentially sensitive to the hyperparameters n₁ and n₂. It would be helpful if the authors could analyze how these values affect stability and performance.  
2. The paper lacks clear guidance or heuristics for selecting n₁ and n₂, which seem crucial to the method’s effectiveness.  
3. The work assumes that longer reasoning tends to produce incorrect answers, yet the analysis mainly shows correlation rather than causation. It remains possible that more difficult problems naturally require longer reasoning.  
4. The paper could better position itself relative to prior studies on difficulty dynamics in reinforcement learning for efficient reasoning, such as [1].  

[1] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping. https://arxiv.org/abs/2505.15612

### Questions
See weakness

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
This paper addresses the excessive and unnecessary reflection problem in Large Reasoning Models by proposing an adaptive reflection and length coordinated penalty (ARLCP). The method first estimates the problem's complexity by counting the Reflection Token Counts in a generated response, then assigns a penalty weight based on whether the problem is categorized as "simple," "moderate," or "hard." During RL training, a second penalty is applied to the overall response length to discourage general verbosity. The method achieves a superior efficiency-accuracy trade-off, simultaneously reducing length and improving accuracy.

### Strengths
- The work directly addresses the "over-reflection" in Large Reasoning Models (LRMs), a key issue that leads to high token consumption, computational overhead, and latency without improving accuracy.
- The core idea of an adaptive reflection penalty is a key innovation.
- The thorough analysis and ablation show that the method improves both efficiency and accuracy.

### Weaknesses
- The core "Adaptive Reflection Penalty" relies entirely on a manually curated list of "reflection-trigger keywords" like "wait", "hmm", "alternatively", etc. This relies on a specific model response style and is not generalizable to other distributions or domains.
- The method introduces several new and important hyperparameters: the complexity thresholds ($n_1=40, n_2=80$) and the penalty weights ($\lambda_1=0.05, \lambda_2=0.1, \lambda_3=0.15, \alpha=0.2$). The paper presents these as fixed values but does not include a sensitivity analysis.
- The method uses the output Reflection Token Count (RTC) as a proxy for the input problem's complexity. The paper itself shows that incorrect answers (i.e., failed reasoning) lead to higher reflection. This means the model might be penalizing "hard" for a problem that is actually "simple" but on which the model got stuck and "over-reflected." It's penalizing the symptom (high RTC) rather than the cause (intrinsic problem difficulty).

### Questions
- Have you tried directly using RLOO on the training dataset? I'm curious why this method achieves short outputs but still has the best performance.
- Did you face any instability during training? Does RLOO truly prove to be more stable?

### Soundness
3

### Presentation
3

### Contribution
2
