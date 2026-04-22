# Do LLM Evaluators Prefer Themselves for a Reason?

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Large language models (LLMs) are increasingly used as automatic evaluators in applications such as benchmarking, reward modeling, and self-refinement.
Prior work highlights a potential *self-preference* bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability.
This raises a critical question: 
Is self-preference harmful, or does it simply reflect the genuinely higher-quality outputs of stronger models?
Answering this has been difficult because previous studies relied primarily on subjective tasks. 
These tasks lack an objective ground truth, meaning that either preference can be reasonably justified.
To address this ambiguity, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. 
This enables us to distinguish *harmful* self-preference (favoring objectively worse responses) from *legitimate* self-preference (favoring genuinely superior ones). 
We conduct large-scale experiments under controlled evaluation conditions across diverse model families (*e.g.*, Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek).
Our findings reveal three key insights:
(1)
While stronger models exhibit greater self-preference, much of this preference aligns with objectively superior performance, indicating stronger models prefer themselves mostly legitimately.
(2) Harmful self-preference persists when evaluator models err as generators, and stronger models display more pronounced harmful self-preference bias when they do err.
This suggests stronger models struggle more to recognize when they are wrong.
(3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce the harmful self-preference.
These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses a timely topic: the self-preference of LLMs when acting as judges. Unlike prior studies, the authors conduct experiments on verifiable tasks (mathematical reasoning, factual question answering, and code generation), which allows them to distinguish whether an LLM’s self-preference is legitimate or harmful. The authors perform cross-family and cross-scale evaluations, where each model serves as both evaluator and evaluatee. They define three metric, SPR, LSPR , and HSPP, to quantify the magnitude and type (legitimate vs harmful) of self-preference. Empirical results show that stronger models exhibit higher self-preference, which is mostly legitimate. While harmful bias does exist, longer reasoning reduces them. Based on these findings, the authors offer several practical suggestions for improving the reliability of LLM-as-judge systems.

### Strengths
1. This paper studies a timely and relevant topic, the self-preference of LLM-as-judge, and provides practical suggestions for improving their reliability.
2. By focusing on verifiable tasks (math reasoning, factual QA, and code generation), the paper fills a gap left by previous works by allowing for subjective evaluation and comparison that distinguishes legitimate from harmful self-preference.
3. The conducts a systematic and large-scale evaluation across multiple model families and sizes. The comparisons are comprehensive, the experiments are well-organized and reproducible, and the presentation of results is clear and visually effective.

### Weaknesses
1. Some claims in the paper are subjective or speculative, not supported by empirical evidence. For example, in Line 252-256, the authors interpret correlations between model accuracy and self-preference as causal evidence that stronger models “recognize correct answers” or “identify bugs,” yet no qualitative analyses or case studies are provided to support these assertions. At minimum, the paper should include some examples showing how the LLM-as-judge identifies or explains specific errors in outputs to substantiate this claim.

2. This results have limited generalizability. While verifiable tasks allow for clear evaluation of LLM-as-judge behavior, most real-world applications of LLM-as-judge systems involve subjective tasks. It remains unclear whether the findings on self-preference in verifiable domains transfer to subjective evaluation settings, which substantially limits the paper’s practical contribution.

3. The evaluation is somewhat incomprehensive. The authors compute SPR over the entire $\mathcal D$ and analyze LSPR and HSPP on the $\mathcal D_{diff}$. However, the subset where evaluatees are both correct or both wrong, say $\mathcal D_{agree}$, is not analyzed in detail. I think this subset is particularly important for understanding the model’s self-preference that comes out of recognizing their own production when correctness provides no differentiating signals. Examining behavior on $\mathcal D_{agree}$ would offer a more complete view of self-preference behavior beyond what can be inffered from $\mathcal D_{diff}$ alone.

4. My biggest concern is that the claim in Line 269, “the degree of self-preference in stronger models is primarily driven by the objective quality of their outputs (evident by their higher judge accuracy)”, is an overclaim. What the paper actually demonstrates is a correlation, but this relationship may be affected by multiple confounding factors. The authors aim to conduct a systematic cross-family and cross-scale evaluation, which is a good point. But this design inevitably confound the relationship between generation accuracy and evaluation accuracy, making it unclear whether the upward trend in SPR arises from higher generation quality (a legitimate bias), or from identity/stylistic effects, where a model may recognize its own outputs. The latter one can increase SPR either legitimately or harmfully depending on the competitor’s quality. Alsom the authors state that stronger models tend to exhibit both greater harmful self-preference (Line 359) and mostly legitimate self-preference (Line 269). What I can derive from these phenomenon is that stronger models simply show greater self-preference overall, and their higher legitimacy ratio is possibily a byproduct of their higher task accuracy. Since they are correct often, their self-preference *appears* justified, which weakens the causal interpretation of these findings.
A more convincing design would involve comparing models of similar parameter scales or conducting analysis after controlling for generation accuracy to disentangle these effects. 

5. Although this work is the first to explicitly measure self-preference on verifiable tasks, filling a gap in the existing literature, its contributions remain largely observational. I also note that several recent studies have already begun to mitigate or correct harmful self-preference, such as “Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators” and “Limits to Scalable Evaluation at the Frontier: LLM-as-Judge Won’t Beat Twice the Data.” It would strengthen the paper to review and discuss these works in the related literature.

### Questions
Seek weaknesses.

### Soundness
2

### Presentation
3

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
This paper studies self-preference (LLMs favoring their own outputs when acting as evaluators) on verifiable benchmarks so the authors can distinguish legitimate self-preference (preferring objectively better self-outputs) from harmful self-preference (preferring objectively worse self-outputs). They run a controlled, large-scale pairwise evaluation across many model families and sizes (Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek distilled models) on three verifiable tasks: MATH500 (math reasoning), MMLU (factual knowledge), and MBPP+ (code generation). They define metrics—SPR (self-preference ratio), LSPR (legitimate self-preference ratio), and HSPP (harmful self-preference propensity)—and show: (1) stronger models prefer themselves more, but much of that preference is legitimate (Figures 2–4); (2) when stronger models are wrong as generators they more often still prefer their own incorrect outputs (HSPP increases with capability; Figure 5); (3) generating reasoning (CoT / long CoT) before making a verdict substantially reduces harmful self-preference (Figure 6). The paper gives practical recommendations for safer LLM-as-judge use (model pre-assessment, inference-time scaling, routing to specialized evaluators).

### Strengths
* Clear, useful framing. Distinguishing legitimate vs harmful self-preference is a simple idea but crucial for interpreting prior findings; the paper operationalizes this clearly (SPR / LSPR / HSPP).
* Controlled experiments. Same evaluatee set across judges, multiple model families and scales, and three distinct verifiable domains (math, factual, code) improves generality.
* Concrete empirical findings. Strong models are better evaluators and often prefer their own (legitimate), but they also exhibit higher harmful self-preference when they err, a non-obvious safety concern emphasized with numbers and per-model results.

### Weaknesses
* Further interpretation and insight for self-preference. The paper would benefit from a deeper exploration of why self-preference arises and under what conditions it becomes harmful versus legitimate. For instance, is harmful self-preference linked to model confidence calibration, output style similarity, or internal alignment mechanisms? Providing diagnostic analyses (e.g., correlation with log-probabilities or similarity metrics) could make the findings more interpretable and actionable.

* Self-preference in subjective tasks. While using verifiable benchmarks is a strength, which allows clean separation between legitimate and harmful self-preference, it remains important to validate whether the observed patterns hold in subjective or open-ended tasks, where evaluation bias is more subtle and consequential. Such tasks are inherently more prone to hidden self-preference because ground-truth correctness is ambiguous. The authors could test robustness by leveraging annotated pairwise samples from LMArena or similar datasets containing subjective human judgments. This would help assess whether the same mitigation strategies (e.g., CoT reasoning) generalize to evaluators on creative or conversational domains. I would be open to raising my overall score if such experiments or discussions were added.

### Questions
See the weakness section.

### Soundness
3

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
- Investigates whether LLM judges' self-preference reflects genuine quality or harmful bias using verifiable math, factual QA, and code tasks, formalizing position-robust aggregation J^* and a self-preference ratio.
- Finds that stronger models tend to prefer their own outputs largely in line with objective correctness, while harmful self-preference is captured by HSPP.
- Shows that longer CoT before verdicts mitigates HSPP and offers practical guidance for LLM-as-judge pipelines.

### Strengths
- Comprehensive coverage across evaluator models and benchmarks with careful handling of position bias and ties.
- Clear visualizations and definitions that show relationships between capability and self-preference.
- Actionable mitigation signal: longer Chain-of-Thought reduces HSPP with a known token-budget tradeoff.
- Evidence aligns across model families and scales, supporting robust empirical patterns.

### Weaknesses
- Limited external validity to subjective or multi-criteria evaluations.
- Possible coupling when a judge evaluates its own generations, suggesting a withheld-generator check.
- Narrow decoding settings; robustness to sampling choices is not fully explored.
- Limited theoretical analysis and few mitigation avenues beyond longer reasoning, plus missing statistical significance reporting.

### Questions
- How do results extend to subjective or multi-attribute domains and to formats like scalar scoring or ranking?
- How do sampling-based decoding choices affect SPR and HSPP, and does longer CoT still help under higher response diversity?
- Can harmful self-preference be isolated by fixing the judge and evaluating on responses from withheld generators?
- What linguistic or family-level style correlates predict harmful self-preference, and what mechanisms drive it?

### Soundness
3

### Presentation
3

### Contribution
2
