# VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Large reasoning models such as OpenAI o1 and DeepSeek-R1 have demonstrated remarkable performance in complex reasoning tasks. A critical component of their training is the incorporation of reference-based reward systems within reinforcement learning (RL), where model outputs are evaluated against ground truth references.  However, existing reward benchmarks focus on preference comparisons between responses rather than evaluating verification against ground truth references, leaving a critical gap in our ability to evaluate verification systems used in reasoning model training. In this paper, we introduce VerifyBench and its challenging variant VerifyBench-Hard, two benchmarks specifically designed to assess reference-based reward systems. These benchmarks are constructed through meticulous data collection and curation, followed by careful human annotation to ensure high quality. Our comprehensive evaluation reveals that while larger model-based verifiers show promise on standard cases, all current systems demonstrate substantial room for improvement on challenging instances. Through systematic analysis of performance patterns across reasoning tasks and error categories, we provide insights for advancing reference-based reward systems. These benchmarks establish a standardized framework for improving verification accuracy, ultimately enhancing reasoning capabilities in models trained via RL.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces VerifyBench and VerifyBench-Hard, two benchmarks for evaluating reference-based reward systems in reinforcement learning for large language model reasoning. Unlike previous reward benchmarks that rely on pairwise preference judgments, VerifyBench assesses whether the reward models are able to identify if a model-generated response is correct or not, given reference answers. The datasets are built through large-scale curation and careful human annotation.

### Strengths
- The work is timely, filling a clear gap in the evaluation landscape as the community shifts from pairwise preference to reference-based reward evaluation in RL for reasoning models.

- The pipeline ensures high diversity (spanning general, logical, and mathematical reasoning), and the use of human annotators for verification improves reliability and validity.

- The RL correlation experiment convincingly demonstrates that verifier quality (as measured by VerifyBench) directly impacts downstream RL performance, emphasizing the benchmark’s practical utility.

### Weaknesses
- Despite the diversity of question types, coding-related tasks (e.g., code generation, debugging) are notably absent. 

- The benchmark does not include regex-based verifiers, which are widely used in practice.

### Questions
- Could you clarify the difference between LLM-as-a-judge and Model-based Verifier? Since both involve LLMs producing binary verification signals, what distinctions make the latter uniquely "model-based"?

- In the RL correlation results, the VerifyBench-Hard scores suggest that stronger verifiers (e.g., gpt-oss-20b) outperform weaker ones (e.g., Qwen3-4B), yet this does not translate linearly into downstream RL gains (Figure 3). What do you think are the sources of this mismatch?

- Do you plan to maintain a public and continuously updated leaderboard (similar to RewardBench) to track verifier progress over time? Such a platform could help make VerifyBench the de facto standard for reference-based reward evaluation.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VerifyBench and its more challenging variant, VerifyBench-Hard, as two new benchmarks designed for the systematic evaluation of Reference-Based Verifiers. The two benchmarks cover four distinct answer formats and three different reasoning task types. Through extensive experimentation involving Rule-Based Verifiers, LLM-as-a-Judge methods, and Model-Based Verifiers, this study reveals that while current state-of-the-art model-based approaches perform well in general testing, VerifyBench-Hard still poses a significant challenge to them. The paper further explores the performance of traditional ORMs and experimentally confirms that a higher score on VerifyBench positively correlates with the greater performance improvement the verifier provides during RL training.

### Strengths
1. The benchmark provides a validated evaluation framework that enables quantitative assessment of verifiers.

2. The test setting is diverse and challenging, and it is able to distinguish even among SOTA models.

3. The experiments empirically demonstrate a correlation between higher benchmark scores and better training performance.

### Weaknesses
1. The paper does not appear to include examples of difficult samples. Showing more complex and challenging instances from the benchmark would be convicing.

2. The benchmark is mainly focused on mathematics, so it may not reflect performance on verifiable tasks across the broader domain, such as evaluating physics expressions or chemical equations.

### Questions
1.RL training with Qwen-4B as the verifier clearly outperforms training with weaker models, while OSS does not yield additional gains. Does this suggest that there is in fact some “threshold” that determines whether a verifier is basically reasonable, and that once this threshold is reached, further gains become minor?

2.In the RFT experiments, were models stronger than Qwen-4B (e.g., OSS used in the RL experiments) also evaluated? If so, how did they perform?

3.Can you provide some challenging examples from VerifyBench and illustrate how the models judged them correctly or incorrectly? In particular, for the OSS model, examining its reasoning traces may clarify why it makes incorrect decisions. This could in turn inform future verifier improvements.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents VerifyBench, a benchmark designed to evaluate the accuracy of reference-based reward systems. Specifically, given a question, reference answer and a generated completion, the benchmark tests if models are able to correctly classify the generated completion as correct/incorrect. A well-designed pipeline consisting of human annotation and large-scale model generations is used to curate VerifyBench and VerifyBench-Hard, a harder variant of the same task. Evaluations on the benchmark show that current models (especially smaller ones), have significant room for improvement on the task.

### Strengths
- The paper is well-written, smooth, and easy to follow, with clear structure and presentation.

- The data curation pipelines are comprehensive and well-designed. 

- The benchmark tackles a challenge that is not addressed in existing benchmarks.

### Weaknesses
- The final source distribution of completions may be biased toward specific models, which could undermine benchmark diversity. More analysis on this would address this. 

- Some more clarifications on the data curation pipeline will further increase clarity of the paper.

### Questions
1. What is the final source (model) distribution of completions? Models have a bias to prefer their own completions, so all completions coming from a small number of models will be undesirable. 

2. Do completions include long reasoning traces (chains of thought) or only the final summarized solutions? Reasoning models generate long thinking traces involving complex behavior such as self-reflection/verification and backtracking. I personally do not think these traces should be part of the completion, instead the completion should be a final summary of the solution (Eg. Reasoning traces of GPT5 are not visible, only the final solution is presented to the user). I am curious to hear how authors approached this.

3. What does the length distribution of the completions in the benchmark look like?

4.  I liked the experiment of training with the best verifiers on the benchmark. Another experiment, which is much easier to run, is to use verifiers for rejection sampling (best-of-N). This can be compared to pass@N (using a perfect verifier). The difference in performance between the two methods would be another indicator of the strength of a verifier (an ideal verifier should closely match the pass@N curve). I believe this is also a cleaner experiment, as training might have second-order effects.   

5. My main doubt is on the performance of reference answers. If reference answers are provided to the model, then isn’t the task simply just to extract the final answer from the completion and check equivalence to the reference answer. In domains like Math, this is exactly what a symbolic parser (e.g., Math-verify) does. This might be a harder task in non-math domains, so I am curious to understand the benefits of this in tasks like Math. In general, if one of the applications of these verifiers is RL training in domains where ground truth answers are unavailable, then a more practical evaluation is one without reference answers. Similarly, rolling out long traces from verifiers is also not practical for RL training, especially if slightly worse but significantly cheaper symbolic parsers are available. 

6. Adding on to the previous point, if verifiers are to be used for inference scaling (best-of-N), then reference answers will not be available. Perhaps, the best thing to do is to expand the benchmark to have a reference and non-reference setting? This would be able to get the best of both worlds. 

7. How exactly is controlled downsampling performed, and does it distort the difficulty/accuracy distribution of the dataset?

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
This paper presents VerifyBench and a more challenging version VerfiyBench-Hard to assess reference-based reward systems used in training large reasoning models (LRMs). The authors develop a step-by-step pipeline to create these benchmarks. The authors evaluate a large pool of large language models (LLMs) and reward models (RMs) on these benchmarks.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed benchmarks, VerifyBench and VerifyBench-Hard, are well-designed to evaluate the capabilities of reward models in assessing the correctness of reasoning steps in LLM-generated outputs. These benchmarks could be valuable resources for the research community.

### Weaknesses
1. The motivation of the paper is somewhat questionable. Typically, the correctness of responses can be directly verified against ground truth answers without the need for reward models. Using a reward model to verify correctness can achieve higher accuracy compared to directly comparing with ground truth answers, as demonstrated in Figure 12, 13, and 14. However, it is unclear whether the added complexity of using reward models is justified given the performance gain.
2. The benchmark appears to be relatively easy for current state-of-the-art LLMs and RMs. As shown in Table 2, the best-performing LLMs and RMs achieve over 90% accuracy on VerifyBench, leaving limited room for improvement. In the evaluation, The verifier models are esstentially tasked to compare the consistency between the reference answer and the model-generated answer, which should be relatively straightforward for current LLMs.
3. In Section 5.2 and Appendix H, the authors demonstrate that better models on VerifyBench is a better verifier in RL training. However, it is unclear whether these models are better than directly matching with ground truth answers under the same computation budget.

### Questions
1. The authors demonstrate that better verifier models lead to better LLMs when used in RL training. What is the relationship between the verifier model's performance and the resulting LLM's performance? I am wondering whether weaker/smaller LLMs typically require stronger verifiers while stronger/larger LLMs can work well with weaker verifiers, such as direct matching with ground truth answers. Is this true?

### Soundness
3

### Presentation
3

### Contribution
2
