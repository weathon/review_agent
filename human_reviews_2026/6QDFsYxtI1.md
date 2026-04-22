# When More is Less: Understanding Chain-of-Thought Length in LLMs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 2

## Abstract
Large Language Models (LLMs) increasingly rely on Chain-of-Thought (CoT) reasoning to solve complex problems. Contrary to the common belief that longer CoTs always improve performance, we demonstrate that **longer is not always better**. Across both real-world LLMs and theoretical models, task accuracy follows an inverted U-shaped curve with respect to CoT length: performance rises initially but declines once reasoning chains become too long. Through controlled experiments, we uncover **scaling behaviors of the optimal CoT length**: it increases with task difficulty but decreases with model capability. This exposes a significant mismatch with current practice, where supervised training often reuses the same CoT data across models and tasks without adaptivity. We further show that Reinforcement Learning (RL) can mitigate this gap by dynamically calibrating CoT length, thereby improving accuracy and offering a new perspective on differences between supervised fine-tuning and RL training. To explain these phenomena, we introduce an error-accumulation analysis that characterizes how reasoning errors propagate across steps and derives the scaling behaviors of CoT length observed empirically. Building on these insights, we show that training with optimally sized CoTs and applying length-aware filtering during inference yields substantial improvements in performance. Taken together, these findings establish a principled explanation of the ''overthinking'' effect and yield practical guidelines for calibrating CoT length in accordance with task complexity and model capability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies how CoT length affects accuracy and cost. It reports an inverted U-shaped relationship between CoT length and task accuracy: performance improves up to an optimal CoT length, then degrades due to error accumulation; the optimal length increases with task difficulty and decreases with model capability. The authors present (i) evidence on real LLMs (Qwen2.5 series) and synthetic setups (arithmetic, DP), (ii) an error-accumulation theory that yields a closed-form for the optimal length and scaling laws, and (iii) two applications: training with optimal-length CoTs and an inference-time Length-Filtered Vote that selects answers from near-optimal-length samples. They also show that RL can calibrate CoT length toward the optimum.

### Strengths
This paper studies an important problem about CoT length and the empirical analysis provides clear patterns: Multiple figures show the inverted-U accuracy vs. length and the stated scaling behaviors (harder tasks → longer optimal length; larger models → shorter optimal length). The well-controlled synthesis experiments are insightful.

The theoretical study strengthens the insight of this work: A compact error-accumulation model derives an explicit optimal length and recovers the empirical scaling laws.

The provided practical implication provides actionable takeaways and can inspire future study.

Overall, this is a comprehensive and insightful paper combining both analysis and practical usage.

### Weaknesses
1. Missing experimental details, especially in Section 2.

a) The definition of CoT length is very vague, and the authors do not provide details of how to extract the length, such as whether use another model or parsing based on some rules.

b) The definition of task difficulty also requires proper thought as the accuracy can be biased. For example, if the model favors more CoT steps on one harder question (the difficulty from a human's perspective) and achieves higher accuracy, while generating shorter steps for another simpler question and achieving lower accuracy. This can lead to the wrong problem difficulty. More proper ways include leveraging the dataset's own difficulty level, human judgment, advanced LLM's judgment and etc. 

2. There exists a concern about whether the optimal CoT length phenomenon on synthetic data can scale up to larger models. 

a) The Arithmetic Problem seems to be too simple for advanced mdoels (maybe even for small-size models in the Qwen2.5 family), and I suspect that advanced models can solve these questions in one step and therefore make this dataset trivial. I suggest that the author extend the Arithmetic Problem from only addition to include more complex operators.

b) The DP problem is interesting, and it is worth testing on more advanced models.

c) The claim of per-step difficulty can benefit from more experiments. There may also exist an optimal per-step difficulty as the model can not handle sub-tasks with infinitely increasing difficulty (otherwise, one step is already optimal). This is crucial as it can guide how to define sub-tasks based on the difficulty.

3. While the theoretical analysis is intuitive, it relies on strong assumptions that may only hold for the Arithmetic Problem. This is not a very critical problem, as I understand the difficulty in modeling CoT errors. However, the authors should explicitly mention the assumptions in the main paper; otherwise, people may be confused about why $\sigma(T)$ and $E(N,M,T)$ are independent of the step index, and the sub-question error only relies on the total difficulty (in proposition 4.1). This concern arises because in practice, steps are generated autoregressively and later step depends on previous steps. Please make these clear.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the question of how long chain of thought (CoT) reasoning should be to achieve optimal performance in LLMs. They find that the the ideal CoT length generally increases as tasks get harder, and generally decreases as models become larger. They develop synthetic experiments to support their hypothesis, and further demonstrate their findings on real-world tasks for large models.

### Strengths
* This paper is well-written. 
* As mentioned, the paper provides both synthetic and real-world tasks where the findings are observed. 
* The authors meaningfully point out that reasoning traces typically become longer (most prominently demonstrated by the Deepseek-R1 paper), yet they show evidence that in fact this does not always true (e.g. on Leetcode-2k). 
* The authors provide one important application of their work, which is to propose a novel majority voting mechanism weighted by reasoning length, which outperforms standard majority voting.

Overall I think that this paper makes important contributions toward addressing the fundamental question of how long reasoning should be in LLMs, and has important implications for how to design LLMs with a specific reasoning budget.

### Weaknesses
My main concern is that I do not think the experimental setup adequately discusses how important reflection/backtracking is to the reasoning process, which I think the authors rightfully point out is present in "real-world CoTs" (Section 2, Appendix A.3). To frame it another way, the main question I am asking is: given that real-world models do self-correct, **how does self-correction/backtracking play a role in influencing the total length of the CoT**?
* To my understanding, none of the synthetic setups (addition/dynamic programming) include some variant of verification/self-correction. The conclusion from the synthetic experiments seem to suggest more that  "LLMs can perform multiple calculations in one step", rather than "LLMs _require_ only X steps to solve the full problem". 
* I wonder if it would be possible to augment the synthetic setup with some "mistake" in the reasoning process. For example in the addition case, one could artificially inject a wrong addition step (1+2+3+4=_11_), then follow-up with some self-reflection token in the next step and provide the corrected step. I think it would be interesting to see how much noise you add to the trace influences the final optimal length. 
* I find Corollary 4.4 to be a little weak. It simply states that RL training optimizes for reward $A(N)$, which should be straightforward to see given that this is just the REINFORCE objective. The empirical analyses (Figure 2c, Fig 4) seem to suggest that CoT length keeps going down - I am wondering if the theory would say anything further about when CoT length goes up or down?

### Questions
* I am missing some intuition on why RL is causing CoT length to go down (Figure 2c). What is happening to the model rollouts? Is it learning to merge steps, or do less backtracking, etc? 
* Similar question for Figure 4: what happens as the model goes from CoT length 24->5 - is it combining lots of steps together? 
* Figure 1b: for task difficulty 24, seems to only get better with more CoT length. does this level off?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper reveals that the performance of Chain-of-Thought (CoT) reasoning does not uniformly improve with length; rather, accuracy follows an inverted U-shaped curve. The authors demonstrate that an optimal CoT length exists, which is dependent on several factors:
- Task Difficulty: More difficult problems require longer optimal CoT lengths to solve effectively.
- Model Capability: The optimal length varies inversely with model capability. More capable models tend to favor shorter, more efficient reasoning paths.
- Training Dynamics: Reinforcement Learning (RL) training can also induce this simplicity bias, guiding models to gravitate towards shorter, more optimal CoT lengths as their accuracy improves.
Based on these insights, the authors propose a theoretical framework that models the trade-off between task decomposition and error accumulation. They demonstrate that this insight can be practically applied to achieve better accuracy by aligning models with this optimal length during both training and inference.

### Strengths
1. The paper is well-written, clearly organized, and easy to follow. The authors articulate their core argument effectively.
2. The core finding is "an optimal Chain-of-Thought (CoT) length exists". This conclusion is convincingly demonstrated. The results of most experiments are very clear.
3. This study not only analyzes the results, but also provides guidance for the reasoning process of practical models, which has also shown positive effects in experiments.

### Weaknesses
The controlled experiments described in Section 3 may be problematic because factors other than CoT length were altered. For instance, the long and short CoT solutions differ not only in total length but also in how they approach problem-solving: short CoTs take fewer but longer steps, whereas long CoTs take more but shorter steps. Since the paper measures CoT length by the number of steps and controls this variable in the experiments, variations in step length and the more complex operations used in shorter CoTs could act as confounding factors, weakening the validity of the claimed causal link between length and performance.

### Questions
1. The sampling of model sizes in Figure 2(a) is sparse, making the "simplicity bias" trend less conclusive (e.g., the 72B model's optimal length is slightly longer than the 32B's). Adding intermediate models, like the 14B, would help create a more convincing trend line. Additionally, adding the MOE model would be better.

2. In Section 2, regarding the impact of RL on CoT length, an additional set of experiments using models of other scales should be included. The current results only demonstrate that RL can reduce CoT length. (While other studies have concluded that RL can increase CoT length, your intended proof here is not merely about whether CoT length increases or decreases, but rather that it converges to the optimal length—this requires verification through different trends in CoT length under different conditions.)

3. In the current experiments, most only examine the model's performance at a specific CoT length by manually controlling the length (see Fig. 4). However, I am more curious about the difference between the CoT length independently chosen by the RL-trained model during inference (both before and after RL) and the optimal CoT length. Can it be experimentally demonstrated that sufficient RL training enables the CoT to converge to the true optimal length?

4. Inconsistency in Figure 2: The caption for Figure 2(b) states it is "with the 7B model," whereas the main text (line 152) references the "Qwen1.5B-Instruct model" when discussing this result. 

5. The definitions for symbols in the theoretical model could be strengthened. For instance, the constant 'C' introduced in Equation (1) (and later in Theorem 4.3) is defined somewhat informally.

6. Recent related work about theoretical understanding of CoT: "How Likely Do LLMs with CoT Mimic Human Reasoning?" and "Correlation or Causation: Analyzing the Causal Structures of LLM and LRM Reasoning Process".

### Soundness
4

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
This paper presents several insights about the length of reasoning chains within long chain-of-thought reasoning.

### Strengths
1. Overall the paper is pretty well written and I was able to follow the main points.
2. The insight regarding step-wise computation also increasing for difficult instances is an interesting one that as far as I know as not covered extensively in previous papers.

### Weaknesses
The paper is mainly insight-driven, but as far as I can tell many of the insights presented in the paper have already been uncovered by previous work that was not cited:

1. Insights related to long CoTs not being better are covered by Jiang et al., which was not cited.
2. The idea of error accumulation in CoTs being responsible for long chains of thought being less successful was published in Schaeffer et al. 2023.
3. Adaptive length-filtered voting was examined by Fu et al., which was cited in the paper but not cited in the appropriate section.

References:
- What Makes a Good Reasoning Chain? Uncovering Structural Patterns in Long Chain-of-Thought Reasoning. Jiang et al. EMNLP 2025.
- Are Emergent Abilities of Large Language Models a Mirage? Schaeffer et al. NeurIPS 2023.
- Complexity-Based Prompting for Multi-Step Reasoning. Fu et al. ICLR 2023.

### Questions
1. If you were to refine the main claims of the paper based on what has already presented in the previous work that I cited, what do you think are the remaining most valuable insights?
2. I'm not sure if the term "scaling law" is appropriate in this case. This is a bit of a nuance, but usually when we talk about "large scale" it's about things like data, compute, etc. and not some 10s of thousands of tokens in a reasoning trace. Maybe you could consider a different term here?

### Soundness
3

### Presentation
3

### Contribution
2
