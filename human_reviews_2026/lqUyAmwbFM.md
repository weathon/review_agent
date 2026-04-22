# What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Large reasoning models (LRMs) spend substantial test-time compute on long chain-of-thought (CoT) traces, but what *characterizes* an effective CoT remains unclear. While prior work reports gains from lengthening CoTs and increasing review (revisiting earlier steps) via appended *wait* tokens, recent studies suggest that shorter thinking can outperform longer traces. We therefore conduct a systematic evaluation across ten LRMs on math and scientific reasoning. Contrary to the “longer-is-better” narrative, we find that both naive CoT lengthening and increased review are associated with *lower* accuracy. 

As CoT unfolds step by step, token-level metrics can conflate verbosity with process quality. We introduce a graph view of CoT to extract structure and identify a single statistic—the *Failed-Step Fraction (FSF)*, the fraction of steps in abandoned branches—that consistently outpredicts length and review ratio for correctness across models. To probe causality, we design two interventions. First, we rank candidate CoTs by each metric at test time, where FSF yields the largest pass@1 gains; second, we edit CoTs to remove failed branches, which significantly improves accuracy, indicating that failed branches bias subsequent reasoning. Taken together, these results characterize effective CoTs as those that *fail less* and support *structure-aware* test-time scaling over indiscriminately generating long CoTs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to explore the factors that may affect the effective reasoning of LLMs. In specific, the authors mainly study on three factors" response length, review ratio, and failed-step fraction.  By conducting experiments on ten large reasoning models (LRMs) and two reasoning datasets, the authors highlight that: **given a same question, longer response length, higher review ratio, or higher failed-step fraction brings negative impact to the reasoning accuracy**. Then, the authors show that re-ranking and selecting responses based on the values of these three metrics can improve the reasoning accuracy, furthering validating the relationship between the metrics and reasoning performance.

### Strengths
1. The studied problem, effective reasoning, is important. The motivation to study the effects of response length, review ratio, and failed-step fraction is clear and straightforward.

2. The paper is well-written. The experimental analysis validates the claim of the paper.

### Weaknesses
1. **The contribution is very limited, and the findings do not provide new insights.** The potential negative effects of longer CoT length and more failed steps are already discovered in previous studies such as [1], the fact that selecting shorter responses under different samplings to a same problem leads to higher accuracy has also been revealed in [2]. The findings in this paper are not surprising to the community.

2. The proposed method that using failed-step fraction as an indicator to achieve more effective test-time scaling is equivalent as introducing an extra LLM critic. Introducing LLM critic or external reward models is a common practice on improve model's inference-time scaling performance [3,4]. 

3. The evaluated datasets are limited. The authors only conduct experiments on a subset of HARP problems and 198 GPQA-Diamond problems. 


[1] Yang, Wenkai, et al. "Towards thinking-optimal scaling of test-time compute for llm reasoning." arxiv 2025

[2] Hassid, Michael, et al. "Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning." arxiv 2025

[3] Madaan, Aman, et al. "Self-refine: Iterative refinement with self-feedback." NeurIPS 2023

[4] Gao, Bofei, et al. "Llm critics help catch bugs in mathematics: Towards a better mathematical verifier with natural language feedback." arxiv 2024

### Questions
Regarding the definition of review ratio in Line 202, why is the ratio calculated based on the proportion of all characters in the review chunk rather than the proportion of tokens?

### Soundness
2

### Presentation
3

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
**Motivation**: Large reasoning models often produce very long chain-of-thought traces to gain accuracy, but what actually makes a good trace is unclear and long outputs raise cost and latency.  The paper revisits whether making thoughts longer or adding more review really helps, and asks if structural traits of the trace explain success across models and tasks.

**Approach**: The authors study ten models on math and science, measure two lexical metrics at the character level (Length and Review Ratio from LLM-labeled chunks), and extract a reasoning graph to compute the Failed-Step Fraction (FSF). 

**Key Results**: Within the same question, shorter traces and lower Review Ratio are linked with higher accuracy, and FSF is the strongest and most consistent predictor across models and difficulty levels. For causal tests, picking answers by lowest FSF gives the largest pass@1 gains, up to about 10% on AIME, and removing failed branches raises continuation accuracy by roughly 8–14%.

### Strengths
**Motivation:** The paper tackles core questions about how LLMs reason. It states clear, testable questions and uses simple metrics at the character level and the graph level, which allows fair model comparisons while controlling for question difficulty.

**Evidence:** The evidence is strong and not just correlational. FSF predicts correctness across tasks, models, and difficulty bands; choosing traces with lower FSF boosts pass@1 more than length or review, and trimming failed branches improves continuation accuracy.

**Message:** The take-home message is practical. Do not default to very long chains of thought; use structure-aware selection and light pruning to get higher accuracy with lower cost and latency.

### Weaknesses
**Novelty of Finding:** The paper’s main claims feel close to earlier results about CoT length and review. Prior works (cited in the submitted paper) have already shown that adding “wait” can help at first but can also hurt, and that shorter traces can match or beat longer thinking, which raises the bar for what is new here. Also, regarding FSF and recovery, the paper itself notes that models often do not fully “unsee” early mistakes when they backtrack, and related work on self-verification probes hidden states without guaranteeing reliable recovery. This aligns with reports of over-verification effects and limited self-correction, which suggests the failure to recover can persist even with verification steps [1,2,3]. Clearer positioning against these threads would help.

[1] Training Language Models to Self-Correct via Reinforcement Learning

[2] Large Language Models Cannot Self-Correct Reasoning Yet

[3] When Can LLMs Actually Correct Their Own Mistakes? A Checklist

**Reliance on LLM-as-judge and extraction prompts:** Several steps depend on LLMs judging other LLMs. Review spans are labeled with the Llama 4 Maverick judge, and the paper shows a confusion matrix against human labels, but the approach could still inherit judge or prompt bias. Robustness checks like judge swaps, alternative prompts, and more human spot checks across tasks would strengthen the claims. Graph extraction also uses Claude 3.7, which introduces another source of dependency that would benefit from ablations or agreement analyses.

**CoT Intervention and missing baselines:** The strongest gains come from re-ranking many samples per question. The test-time selection setup generates 64 candidates per problem, then picks the top one, which is costly even if each trace is shorter. It would be useful to see guidance methods that push the model toward low-FSF traces during decoding, or lower-sample selectors with similar gains. Comparisons against structured search methods like Tree-of-Thoughts and common baselines such as self-consistency would also clarify cost-accuracy tradeoffs.

### Questions
1. How sensitive are FSF estimates to the specific graph extraction and failed-node labeling prompts? Would a different judge shift the results?
2. Can we reduce FSF during generation with lightweight controls, so we need fewer samples for selection?
3. Do these findings hold for tasks like coding, or tool use where long context and retries are common?

### Soundness
3

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
The authors introduce a graph view of Chain-of-Thought (CoT) to extract its structure and identify a single statistic—the Failed-Step Fraction (FSF), defined as the fraction of steps in abandoned branches—that consistently outperforms length and review ratio in predicting correctness across various models. To probe causality, they design two interventions. First, they rank candidate CoTs by each metric at test time, finding that FSF yields the largest pass@1 gains. Second, they edit CoTs to remove failed branches, which significantly improves accuracy, indicating that these failed branches bias subsequent reasoning. Taken together, these results characterize effective CoTs as those that fail less and support structure-aware test-time scaling over indiscriminately generating long CoTs.

### Strengths
The authors' methodology is logically sound, supported by ample experimentation, and the conclusions drawn contribute to enhancing model performance.

### Weaknesses
I acknowledge that the conclusions drawn by the authors are valuable, hence the positive score. However, approaches based on Prompting and Chain-of-Thought (CoT) do not fundamentally enhance the capabilities of LLMs. The authors' approach is akin to reviewing model outputs and If this idea could be integrated into the model's reinforcement learning training or data construction process, it would further enhance the value of this work.

### Questions
Same as above

### Soundness
3

### Presentation
3

### Contribution
3
