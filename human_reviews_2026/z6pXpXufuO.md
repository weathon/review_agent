# REAR: Scalable Test-time Preference Realignment through Reward Decomposition

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Aligning large language models (LLMs) with diverse user preferences is a critical yet challenging task. While post-training methods can adapt models to specific needs, they often require costly data curation and additional training. Test-time scaling (TTS) presents an efficient, training-free alternative, but its application has been largely limited to verifiable domains like mathematics and coding, where response correctness is easily judged. To extend TTS to the domain of preference alignment, we introduce a novel framework that models the task as a realignment problem, as the base model often fails to sufficiently align with the preference. Our key insight is to decompose the underlying reward function into two components: one related to the question and the other to user preference. This allows us to derive a REAlignment Reward (REAR) that selectively rescales the preference-related reward while preserving the question-related reward. We show that REAR can be formulated as a linear combination of policy probabilities, making it computationally efficient and easy to integrate with existing TTS algorithms like best-of-N sampling and tree-search algorithms. Experiments on various preference alignment and role-playing benchmarks demonstrate that TTS with REAR enables scalable and effective test-time realignment with superior performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces REAR, a framework for aligning with user preferences at test time, specifically targeting subjective and open-ended tasks. REAR decomposes the reward function into question-related and preference-related components, enabling dynamic re-weighting of user preference without retraining. The authors show that REAR can be efficiently formulated as a linear combination of policy probabilities, making it tractable and compatible with TTS approaches such as best-of-N (BoN) sampling and Diverse Verifier Tree Search (DVTS).

### Strengths
1. REAR is plug-and-play, and requires no external models or retraining, making it attractive for deployment.
2. REAR outperforms both token-level preference-alignment baselines (e.g., Amulet, Linear Alignment) and TTS methods.
3. The method is grounded in a reinforcement learning formulation and provides detailed proofs.

### Weaknesses
1. Performance of REAR depends on the choice of the λ parameter, which may require  tuning for different questions or preferences.
2. If user preferences are not expressed in words (for example, are only implicit, behavioral, or external), the REAR method as proposed would not function without modification.
3. Heavy reliance on LLM-based evaluation for some tasks raises concerns about evaluation robustness and objectivity.

### Questions
1. Hyperparameter selection of λ: How robust is the method to the choice of λ across different tasks and datasets? Is there a principled or automated way to select λ at inference time, possibly in the absence of validation data?
2. It it insufficient to use only the “helpfulness” score to measure the general response quality in the Section "Analysis on Generated Responses", and also "helpfulness" can be one of the preferences to consider.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes REAR, a test-time method for aligning large language models with user preferences without further training. The key idea is to decompose the implicit reward into question-related and preference-related components and to rescale the preference term at inference time. The authors show that REAR can be computed as a linear combination of log-probabilities and integrated into existing test-time scaling (TTS) algorithms such as best-of-N sampling and diverse verifier tree search (DVTS). Experiments on several preference-alignment benchmarks (PrefEval, Multifaceted, PingPong) demonstrate modest gains compared to existing TTS and test-time alignment baselines.

### Strengths
- The paper is well written and clearly structured, with theoretical derivations and reasonable experimental design.

- The proposed formulation is efficient and lightweight, requiring no extra training or reward models.

### Weaknesses
- The idea of using log-probabilities or policy scores for implicit reward shaping at test time has been explored in prior work. The contribution mainly reformulates known ideas in a slightly different analytical framing. The proposed reward decomposition essentially feeds the reward model (or policy probability) with different segments of the same input (question vs. question + preference) to derive a rescaled score. This approach, while intuitive, lacks genuine novelty or theoretical depth.

- In the second paragraph of the introduction part, the authors state that "However, existing TTS research has predominantly focused on domains such as mathematics and coding". However, test-time alignment has been long studied, even before the prevalence of TTS, including papers such as:

1. Args: Alignment as reward-guided search.

2. Inference-time language model alignment via integrated value guidance.

3. Fast Best-of-N Decoding via Speculative Rejection.


- Performance gains over baselines are modest, often within small margins and without strong qualitative differentiation.

- The method remains heuristic: while mathematically presented as a decomposition, it does not provide clear theoretical or empirical evidence that the “preference” and “question” components can be distinctly separated in practice.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an idea of reward decomposition that is used in test-time inference for the personalized preference task. The idea is simple, decompose the whole reward into question-related reward and preference-related reward. Then, the authors use the policy probability $\pi(a|s)$ as a proxy of the reward $r(s,a)$. 

Based on these reward decomposition, the authors use them in two test-time algorithms, Best-of-N and DVTS. Experiments demonstrate the effectiveness/efficiency of their reward design.

### Strengths
The reward decomposition strategy is simple and easy to understand, while the effectiveness is demonstrated by the experimental results.

### Weaknesses
1. The authors tried to use some mathematical derivation to show the depth of their approach, however, I feel these components were not well stated. For example: From (5) to (7), it feels like nothing substantial is explained. Lemma 3.1 also appears quite abrupt — it introduces a seemingly fancy formula, but its meaning is unclear, and in the end, everything just circles back to (7). Note that Lemma 3.1 is merely an intermediate result and does not offer any theoretical guarantee. It only shows that your policy is an optimizer of a certain expression, making the interpretation of this relation crucial. However, I don't find it particularly insightful or helpful for understanding your algorithm’s design. In fact, it ultimately leads to equation (7), which is equivalent to (5). 

2. What truly matters is Lemma 3.2, as it informs the reader how to compute the reward. However, the authors merely state that the reward can somehow be replaced by the policy probability. I believe this is a critical step in the algorithm’s design, yet the paper provides no intuition or explanation for this substitution. Although proofs are included in the appendix, they also fail to offer any meaningful intuition.

2. The presentation could be improved for better clarity and coherence. For example:
 (a) What is the REAR score, and how is it defined? (I can roughly infer its meaning, but the paper should state it explicitly.)
 (b) What is $\hat{r}_{REAR}$ in Lemma 3.2?
 (c) In Proposition 2.1 and several other places, you should distinguish between the symbols '=' (equality) and ':= / \triangleq' (definition). This is especially important when your notation deviates from standard conventions. For instance, in Equation (3), your Q function is not the one that I learned. If you want to define a new Q function (or the soft Q-function, as you called), you should use a definition symbol rather than the equality sign.

3. After analyzing the paper, I find the novelty is quite limited. It decompose the reward into two reward terms, then replaces the two reward terms by two policy probability terms, and use it as guidance in two common test-time inference methods. The mathematical derivations make little sense and sometimes disrupt the reading flow of the paper.

### Questions
see weakness.

### Soundness
2

### Presentation
1

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
This paper presents a test-time scaling method for preference alignment in LLMs, in case of non-verifiable rewards. This paper makes the assumption that preference is specified in-context. The authors decompose rewards into question-related and preference-related components, then derive a score based on policy probabilities that can be integrated with best-of-N sampling and DVTS. While the core idea has merit, the paper suffers from significant theoretical gaps, questionable experimental design, and overclaimed contributions.

### Strengths
1. The "realignment" framing is intuitive: base models have implicit preferences from training that may not match specific user needs.

2. The paper correctly identifies that test-time scaling (TTS) has been limited to verifiable domains (math, coding) and extending it to subjective preference alignment is a worthwhile research direction.

### Weaknesses
1. What is α? Is it:
  - Task specific constant
  - A property of how the model was trained?

2. This paper would have been much easier to understand if it were presented as "Test-Time Preference Alignment via Policy Interpolation". Overcomplicating a simple method doesn't add value to the paper.

3. Experimental Design Lacks Statistical Rigor. Statistical significance of the results is not reported.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
