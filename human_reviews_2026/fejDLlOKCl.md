# Learning to Reason for Factuality

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Reasoning Large Language Models (R-LLMs) have significantly advanced complex reasoning tasks but often struggle with factuality, generating substantially more hallucinations than their non-reasoning counterparts on long-form factuality benchmarks. However, extending online Reinforcement Learning (RL), a key component in recent R-LLM advancements, to the long-form factuality setting poses several unique challenges due to the lack of reliable verification methods. Previous work has utilized automatic factuality evaluation frameworks such as FActScore to curate preference data in the offline RL setting, yet we find that directly leveraging such methods as the reward in online RL leads to reward hacking in multiple ways, such as producing less detailed or relevant responses. We propose a novel reward function that simultaneously considers the factual precision, response detail level, and answer relevance, and applies online RL to learn high quality factual reasoning. Evaluated on six long-form factuality benchmarks, our factual reasoning model achieves an average reduction of 23.1 percentage points in hallucination rate, a 23% increase in answer detail level, and no degradation in the overall response helpfulness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles hallucinations in large reasoning R-LLMs. The authors observe that standard online RL methods underperform on long-form factuality when they directly use existing evaluation metrics (e.g., VeriScore) as rewards, because these metrics are susceptible to reward hacking (e.g., producing shorter, less detailed, or irrelevant yet factually correct content). To address this, the authors propose a composite reward that integrates: (i) factual precision, (ii) response detail (the number of correct claims), and (iii) answer relevance assessed by an LLM-as-a-judge. Experiments on six long-form factuality benchmarks indicate that this reward design improves performance.

### Strengths
1. The paper introduces a fine-grained reward function to mitigate hallucinations in R-LLMs, addressing the limitations of relying solely on automatic evaluation methods (e.g., VeriScore), which can incentivize less detailed or less relevant responses.
2. The authors present extensive evaluations across diverse datasets, along with ablation studies that isolate and validate the contribution of each reward component.

### Weaknesses
1. The answer-relevance reward appears closely tied to the quality of responses produced by the reference model, yet the paper provides insufficient discussions about the chosen reference model. Please elaborate on this dependency and its impact on results.
2. The contribution may be incremental relative to prior factuality-alignment work. Clarifying the conceptual novelty and quantifying gains over closely related methods would help.
3. The paper lacks comparisons with prior alignment methods. Many factuality-alignment approaches developed for base LLMs can be readily adapted to reasoning LLMs and should be included as baselines.

### Questions
1. Could you include case studies and an error analysis of the proposed approach? In particular, does introducing the answer-relevance reward lead to new forms of reward hacking? For example: Producing overly long responses that mix on-topic content with irrelevant material to inflate relevance scores; Repeating informative paragraphs to appear more detailed or consistent.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies how to improve long-form factuality in reasoning-style LLMs through online reinforcement learning.
The authors design a composite reward that balances factual precision, detail level (number of supported facts), and answer relevance judged by an LLM-as-a-judge.
They implement an online VeriScore system that enables claim-level factual verification within a few seconds, making it practical for RL training.
Using GRPO optimization on Llama-3.1-8B-Instruct, the model significantly increases factual precision and factual detail while maintaining helpfulness.
Ablation experiments show that removing the relevance term causes reward hacking, while combining all three rewards yields more balanced outputs.
Overall, the work demonstrates an effective way to train LLMs for factual reasoning without sacrificing answer quality.

### Strengths
- The paper clearly identifies the challenge of long-form factuality in reasoning LLMs and motivates the need for online RL training.

- The authors implement an efficient online version of VeriScore, reducing verification time to a few seconds per response.

- The experiments cover six factuality benchmarks, showing consistent gains in both factual precision and supported facts.

### Weaknesses
The answer relevance reward depends on another LLM’s judgment, which may introduce bias from the judge model.

- The results may also be sensitive to the choice of the reward LLM, yet the paper does not clearly specify which model or size was used as the judge.

- The authors mention that FactScore leads to less detailed answers, but the paper provides limited explanation of how detail level is precisely measured or how the model avoids generating irrelevant but correct statements.

- Because both relevance and detail are evaluated by an LLM, and the RL objective directly optimizes those same LLM-based rewards, human evaluation would be necessary to validate helpfulness and factuality.

- I think the title "Learning to Reason" may be misleading, as the approach uses outcome-based reinforcement learning with rewards applied only to final answers, making it unclear whether the model actually learns better reasoning chains rather than optimizing end results.

- Minor issue: Figure and table references should be checked. Figure 1 is not clearly cited in the text.

### Questions
- The answer relevance reward relies on another LLM's judgment. Could this introduce bias from the judge model? Did the authors observe or mitigate such bias? 

- How sensitive are the results to the choice of the reward LLM? What exact model and size were used as the judge? 

- The paper mentions that FActScore leads to less detailed answers. How is detail level precisely measured? How do the authors ensure the model is not producing irrelevant but correct statements? 

- If relevance and detail are both evaluated by an LLM, and the RL objective optimizes those same LLM-based rewards, should the evaluation include human assessment to validate helpfulness and factuality? 

- The paper claims the method avoids reward hacking, but could it simply shift the exploitation to new composite metrics? Are current evaluations sufficient to confirm this claim? 

- The approach appears to use outcome-based RL, with rewards only on final answers. Does the model actually learn better reasoning chains, or just optimize for end results?

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
5

### Summary
This paper proposes an online reinforcement learning approach to improve factuality in reasoning LLMs while maintaining response detail and relevance. The method introduces a three-component reward function that simultaneously optimizes factual precision, response detail level, and answer relevance (via LLM-as-a-judge), addressing reward hacking issues where models generate overly short or off-topic responses. The authors implement a scalable VeriScore variant (24x speedup) to enable real-time factuality evaluation during GRPO training. Evaluated on six long-form factuality benchmarks, the trained model achieves 23.1% higher precision and 23% more factual statements compared to the base model. The approach maintains overall response quality with >50% win rate against the base model.

### Strengths
Overall, I appreciate the research question and motivation of this paper, which tackles an important and timely problem with clear significance. The authors observe that state-of-the-art reasoning models (DeepSeek-R1, QwQ-32B) exhibit significantly higher hallucination rates than their non-reasoning counterparts (10-13 percentage points worse on average, Table 1) is both surprising and concerning. This finding challenges the implicit assumption that "more reasoning=better quality" and highlights a genuine problem that the community needs to solve.

### Weaknesses
1. The motivation of the paper is appealing in that it aims to address the issue that previous methods for long-form factuality evaluation have not considered the relevance between the question and the corresponding answer. However, the implementation is somewhat disappointing: it merely compares whether the optimized model's responses are better than those of the base model. But what if the base model's answer is itself irrelevant to the question? This approach does not directly solve the stated problem. Rather, it implicitly assumes that outperforming the base model automatically means being closer to the correct answer. The motivation suggests designing an absolute judgment of relevance, *i.e.*, whether an answer is relevant. But the actual method degenerates into a relative comparison of whether an answer is better than that of the base model.

2. The hyperparameters in Equation (2), such as $\lambda$ and $\mu$, have a substantial impact on model performance, yet the paper does not include a sensitivity analysis. Only three discrete values (0, 0.01, 0.1) are tested. Among the three factors, including factual accuracy, factual detail, and factual relevance, it remains unclear which one is the dominant factor influencing the results.

3. Equation (2) appears to disregard the role of $y_{\text{cot}}$, although I believe the factuality of long CoT is central to reasoning models. The factual consistency within these reasoning steps directly affects the factual correctness of the final answer. The paper does not explain why it considers the content of $y_{\text{cot}}$ negligible. Could this omission exacerbate factuality issues in reasoning models, creating situations where the final answer is correct but the reasoning process is factually flawed?

4. Earlier work has explored similar ideas to accelerate existing factuality evaluation methods, such as VeriFastScore[1].
      
      [1] VeriFastScore: Speeding up long-form factuality evaluation.

### Questions
Please address my concerns in the above Weaknesses section, and then I will accordingly improve my score.

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
4

### Summary
This paper investigates the factuality problem in Reasoning LLMs (R-LLMs), models trained via reinforcement learning to produce long chain-of-thought (Long CoT) reasoning traces (e.g., OpenAI-o1, DeepSeek-R1). The authors observe that such models, despite excelling at math/coding reasoning, hallucinate significantly more than non-reasoning LLMs on long-form factual benchmarks. They propose a RL framework for factual reasoning, centered on a novel reward function combining Factual precision (verified via a scalable variant of VeriScore), Response detail level, and Answer relevance (using an LLM-as-a-judge). They optimize this composite reward using Group Relative Policy Optimization (GRPO). Experiments on six long-form factuality benchmarks (LongFact, FAVA, AlpacaFact, Biography, FactBench, FACTORY) show the effectiveness of the method.

### Strengths
1. The paper tackles an important issue: factuality in reasoning LLMs, extending the RL-for-reasoning paradigm beyond purely verifiable domains.

2. The authors systematically diagnose failure modes (over-precision → short answers; spurious detail → irrelevant verbosity) and explicitly design against them.

### Weaknesses
1. The overlap with concurrent factual-RL papers (Li & Ng (2025) and Ren et al. (2025)) could be better delineated; distinguishing features beyond “long-form” need clearer articulation.

2. The reward is empirically motivated but lacks formal justification or convergence discussion, e.g., how the multi-objective reward interacts with GRPO stability.

3. Given that the core claim is reduction in hallucination, human-verified factuality on a subset would strengthen the argument considerably.

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
1
