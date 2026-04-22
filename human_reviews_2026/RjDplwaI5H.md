# Eye of Judgement: Dissecting the Evaluation  of Russian-speaking LLMs with POLLUX

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Evaluating open-ended generation remains a highly non-trivial challenge, as responses vary in style, quality, and correctness, making reliable assessment difficult. To address this, we introduce POLLUX, an open-source framework for evaluating Russian-speaking large language models (LLMs). Its novelty lies in a criteria-based methodology that improves interpretability by combining a structured benchmark with a family of LLM-as-a-Judge evaluators. For each task type, we define explicit criteria and a scoring protocol in which models not only rate responses but also justify their judgments, offering a transparent alternative to resource-intensive human comparisons. The benchmark spans 35 task types across domains such as code generation, creative writing, and assistant-style interactions, supported by 2,115 expert-authored prompts stratified by difficulty. In addition, we release specialized evaluators (7B and 32B) trained for fine-grained assessment of generative outputs. By uniting a comprehensive taxonomy with automated judges, POLLUX provides scalable and interpretable evaluation tools that move beyond the costs and inconsistencies of human annotation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces POLLUX, an open-source framework for Russian-speaking LLM evaluation. The framework provides 35 task categories along with 22 criteria, and 2115 manually-designed prompts with expert annotations. Finally, 2 LLMs of size 7B and 32B are released to provide automated assessment.

### Strengths
The work contributes to Russian LLM evaluation by crafting a high-quality manually crafted dataset, covering a wide-range of tasks and detailed evaluation criteria.

### Weaknesses
1. The released models (7B and 32B) do not have high correlations compared with baselines in Table 1. In addition, on many categories the correlation is only around 0.1, so the LLM-as-a-judge score in Table 5 & 6 is not reliable.
2. The benchmark contains many task categories (35), but only around 2K prompts. This can lead to high evaluation variance on some categories with very few prompts.

### Questions
How does POLLUX 7B and 32B compare with open-weight models of similar size?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a new benchmark for evaluating LLM capabilities in the Russian language, using tasks derived from the Russian-speaking portion of WildChat-1M. The authors have performed a large annotation effort, and trained LLM-as-judges to try to mimic expert ratings in an automatic metric setting. The authors report on current LLM performance on the benchmark.

### Strengths
The strongest contribution of this paper is a new measure of performance specifically for Russian speaking capabilities, which they use to measure Russian language capabilities of several current models. The measure is the result of a large-scale annotation effort.

### Weaknesses
The paper does not make a convincing case as to why it makes a significant contribution to the field of AI.
While the authors frame the benchmark as helping to address the general challenge of evaluating generative tasks (inappropriately, e.g. in the first 2 sentences of the abstract), the paper does not make a significant contribution to new methodology there---which would require deeply surveying existing methodology and demonstrating why any new methodology improves evaluation. However, the paper only superficially covers related work on evaluation methodology.
Even in the domain of Russian language, I would expect a crisp characterization of what are the limitations of current measures of Russian speaking capabilities, and how does this benchmark address those limitations---but again, the related work was quite weak there.
I am also suspicious of the quality of the LLM-as-judge models, which say reference-free is a benefit; however, I would expect that item-specific criteria (e.g., a rubric) would help to align answers rather than relying fully on parametric knowledge.

### Questions
How would the authors characterize the limitations of current work on evaluation methodology generally and evaluation artifacts for Russian-speaking capabilities, and how does this work address those limitations?
Can the authors justify the quality of the LLM-as-judge models given their reference-free design?

### Soundness
2

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
3

### Summary
This paper proposes POLLUX, a Russian-based framework for LLM-as-a-judge evaluation with more clearly defined judging criteria. The paper evaluates POLLUX against typical LLM-as-a-judge approaches, in contrast to human expert evaluation.

### Strengths
POLLUX focuses on evaluation research for the Russian language, which is less commonly studied compared to English. The proposed method is interesting and demonstrates greater consistency, as it provides explicit judging criteria rather than allowing the model to rank outputs without guidance. This paper contrasts its findings with human expert judgments and evaluates several LLM-as-a-judge baselines. In addition, the provided benchmark resources, including prompts across various difficulty levels and domains, are highly valuable.

### Weaknesses
My main concern lies in the gap within the comparison. The models used for POLLUX differ from those used as the baseline LLM-as-a-judge. Therefore, it is unclear whether the differences in results are caused by the framework itself or by the differences in the underlying LLMs. I suggest including an experimental setting where the same models are used, varying only one factor at a time, to ensure a fair comparison.

The benchmark is expert-created. But it is unclear on what categorizes as expert. Who are they and what are the requirement/demography to be considered as 'expert'. Furthermore, no details on how the expert/annotators were hired or paid.

### Questions
-

### Soundness
2

### Presentation
3

### Contribution
3
