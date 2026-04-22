# Inverse IFEval: Can LLMs Unlearn Stubborn Training Conventions to Follow Real Instructions?

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Large Language Models (LLMs) achieve strong performance on diverse tasks but
often exhibit cognitive inertia, struggling to follow instructions that conflict with
the standardized patterns learned during supervised fine-tuning (SFT). To evaluate
this limitation, we propose Inverse IFEval, a benchmark that measures models’
Counter-intuitive Ability—their capacity to override training-induced biases and
comply with adversarial instructions. Inverse IFEval introduces eight types of
such challenges, including Question Correction, Intentional Textual Flaws, Code
without Comments, and Counterfactual Answering. Using a human-in-the-loop
pipeline, we construct a dataset of 1012 high-quality Chinese and English questions across 23 domains, evaluated under an optimized LLM-as-a-Judge framework. Experiments on existing
leading LLMs demonstrate the necessity of our proposed Inverse IFEval benchmark.
Our findings emphasize that future alignment efforts should not only pursue fluency and factual correctness but also account for adaptability under unconventional contexts. We hope that Inverse IFEval serves as both a diagnostic tool and a foundation for developing methods that mitigate cognitive inertia, reduce overfitting to narrow patterns, and ultimately enhance the instruction-following reliability of LLMs in diverse and unpredictable real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the Inverse IFEval benchmark to measure models’ capability to generalize to OOD instructions. It includes evaluation results of different models on the benchmark tasks and some analysis of why models perform better or worse.

### Strengths
The paper has a strong motivation – language models may struggle to follow instructions that fall outside their training distribution, which can constrain their use cases. The authors develop a benchmark to evaluate this shortcoming over 8 different instruction categories and evaluate several open and closed source models. The paper is also structured and written clearly.

### Weaknesses
While the benchmark has a reasonable motivation (measuring OOD instruction-following), it is not clear that the tasks in the benchmark would provide a useful real-world signal of progress on this failure case. Including more tasks such as “question correction” which can have real-world applicability would strengthen the contribution. The paper benchmarks many models on the proposed tasks, but the analysis feels too succinct and lacking insight. There could be a more thorough discussion of the impact of instruction tuning and why reasoning models are better than non-reasoning models. Following OOD instructions can also lead to harmful responses which we seek to prevent through alignment. A discussion of this concern and how to balance model alignment w/ OOD instruction following feels lacking.

### Questions
- To better motivate the benchmark, would be good to show at least one compelling example of a real-world problem where a model fails OOD due to instruction-following.

- How were the categories for this benchmark selected? Please provide more detailed motivation.

- While the authors mention that the benchmark is similar to an “IQ-test” and doesn’t directly have relevance in terms of task utility, it would be valuable to include a few tasks that are more meaningful or demonstrate that performance on these tasks is correlated with performance on more meaningful tasks so that practitioners are more motivated to use the benchmark.

- Responding to out-of-distribution prompts or instructions is not always desirable and can raise concerns of safety and model alignment. It would be good for the paper to discuss this concern and describe a way to balance the need for model alignment (e.g. not responding to toxic or harmful queries) and model flexibility (responding in OOD formats). How can we measure that a model strikes this balance?

- section 2.5: Please clarify how the 88% and 98% accuracy is computed? and on what dataset and sample size?

- section 2.5: Using LLM-as-a-judge for evaluation on the benchmark is useful, but getting human annotations on a subset of tasks/questions would make these results stronger.

- section 3.1: Overall these findings are not too surprising or insightful. It is intuitive that thinking models or larger models would tend to perform better with OOD formats. It is a bit surprising that instruction-tuned models tend to perform worse as this stage should make the model more prompt or formatting agnostic. Including a discussion / analysis on why this is + results on a base model and its corresponding instruction tuned version would be helpful.

- section 3.2.1: Some more analysis on why thinking improves performance on the benchmark would be valuable. 

- section 3.2.2: This comparison of performance across instruction types also feels cursory. Why are some categories harder than others?

### Soundness
3

### Presentation
3

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
This paper introduces Inverse IFEval, a benchmark that probes whether LLMs can follow counterintuitive instructions rather than defaulting to habits learned during SFT. The authors motivate a new evaluation dimension called Counterintuitive Ability and organize 8 instruction types such as question correction, intentional textual flaws, code without comments, counter-conventional formatting, deliberately incorrect answers, instructional induction, mid-turn instruction modification, and counterfactual answering. The dataset contains 1,012 items in Chinese and English across 23 domains, built through an expert-seeded, human-in-the-loop pipeline with LLM generation and careful filtering.

Experimentally, the study evaluates many closed and open models and finds that an o3 model leads the results, that thinking variants outperform non-thinking ones, and that reduced thinking budget models trail their full budget counterparts. Fine-tuned models perform worse on this benchmark, consistent with its goal of testing out-of-distribution instruction following. The authors also analyze test-time compute and show a clear Best-of-N effect, where increasing N from 1 to 16 to 32 raises scores, and several models approach or pass 90 at N equals 32. Together, these results suggest the benchmark exposes failure modes not captured by prior instruction following tests while leaving room for gains with better post-training.

### Strengths
1. Counterintuitive instructions provide a strong stress test of IF capability, revealing habitual biases and mode locking that standard prompts miss. This perspective fills a gap in existing evals.

2. The benchmark is carefully curated and well-documented, with bilingual coverage and diverse task types that the community can readily reuse. Its transparent construction and release improve reproducibility.

3. The authors evaluate a broad range of models and add thoughtful analyses of thinking modes and test-time compute. The comparisons yield actionable insights on trade-offs and failure modes.

### Weaknesses
1. **Insufficient annotation detail.** The authors lack a detailed description of the annotation process. For example, what is the detailed background of the "experts," and are they affiliated with the authors' team? What is the specific inter-rater agreement rate? Were there multiple rounds of iteration, and what was the proportion of data modified in each round? Also, clarify whether a written guideline existed and, if so, summarize key rules and provide examples of borderline cases and resolution criteria. Adding these details would give us a clearer understanding of the data quality.

2. **Unexplained outliers in results.** Some results appear to be strong outliers, for example Qwen3-235B-A22B-Thinking on QC English at 6.53 and DeepSeek-V3.1 on DIA English at 0.00. An analysis that probes these anomalies and discusses possible causes would be helpful.

3. **Limited error analysis.** The error analysis is mostly a few appendix case studies. A more systematic treatment that categorizes error types, quantifies their frequencies, and relates them to model families and test-time settings would make the findings more actionable.

### Questions
See the above weaknesses.

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
4

### Summary
This paper addresses the cognitive inertia of Large Language Models (LLMs)—their struggle to follow counterintuitive instructions conflicting with training conventions—by proposing Inverse IFEval, a novel benchmark. It introduces "Counterintuitive Ability" to measure LLMs’ capacity to override training-induced biases.
Inverse IFEval includes 8 challenging task types (e.g., Question Correction, Deliberately Incorrect Answers, Counter-Conventional Formatting) and a 1012-sample dataset , built via a human-in-the-loop pipeline (observation→seed construction→large-scale generation→automatic filtering→human verification). An optimized "LLM-as-a-Judge" framework, with task-specific judge models and refined prompts, achieves 98% evaluation accuracy.
Experiments on various LLMs (closed-source like o3-high, open-source like Qwen3) reveal: thinking models outperform non-thinking ones, larger models perform better, and fine-tuned models lack flexibility. The work fills gaps in LLM evaluation, serving as a diagnostic tool and guiding future alignment to enhance LLMs’ reliability in real-world unconventional scenarios.

### Strengths
The idea is neat and interesting. The paper’s emphasis on Large Language Models (LLMs)’ ability to comply with adversarial instructions that conflict with training inertia fills the gap in the existing LLM evaluation landscape regarding the "unconventional instruction-following" dimension. It further complements instruction-following benchmarks of the IFEval category, and to some extent, provides a `lower bound' for instruction-following capability assessment. Additionally, several experimental findings are particularly insightful: for instance, instruction-finetuned models exhibit a decline in long-tail instruction-following capability, which can be attributed to their overfitting to the paradigms of Supervised Fine-Tuning (SFT).

### Weaknesses
Even though the paper provides a critical extension to instruction following evaluation, there are some aspects that I think the paper can further improve on. 

Firstly, the relevance between the benchmark’s tasks and real-world scenarios requires enhancement. Some counterintuitive instructions designed in the work are overly extreme and not aligned with actual user long-tail needs (e.g., deliberately generating texts with a fixed number of typos), which may lead to a discrepancy where models performing well on the benchmark still struggle to address practical unconventional requirements.

Then, the analysis of the root causes of models’ cognitive inertia is insufficient. The paper only observes surface phenomena—such as the underperformance of fine-tuned models and the superiority of thinking models—but fails to delve into how core factors (e.g., supervised fine-tuning data scale, reinforcement learning with human feedback (RLHF) feedback types, or model architectural characteristics) shape cognitive inertia, making it difficult to fundamentally explain the mechanisms behind the observed performance differences.

Thirdly, the error analysis could be improved with in-depth analysis. While the paper presents isolated error cases of several representative models (e.g., Claude-4-Opus, Doubao-1.6-Thinking), it does not summarize common error patterns across different instruction types or explore cross-linguistic error regularities, limiting the insights into models’ systematic weaknesses in counterintuitive instruction-following.

Additionally, if the pos training phase integrates counterintuitive instruction training into their development pipelines, would the current 8 task types in Inverse IFEval lose their ability to effectively discriminate model capabilities? Do you have any idea how to avoid this problem?

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3
