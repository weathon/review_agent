# JudgeBench: A Benchmark for Evaluating LLM-Based Judges

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
LLM-based judges have emerged as a scalable alternative to human evaluation and are increasingly used to assess, compare, and improve models. However, the reliability of LLM-based judges themselves is rarely scrutinized. As LLMs become more advanced, their responses grow more sophisticated, requiring stronger judges to evaluate them. Existing benchmarks primarily focus on a judge’s alignment with human preferences, but often fail to account for more challenging tasks where crowdsourced human preference is a poor indicator of factual and logical correctness. To address this, we propose a novel evaluation framework to objectively evaluate LLM-based judges. Based on this framework, we propose JudgeBench, a benchmark for evaluating LLM-based judges on challenging response pairs spanning knowledge, reasoning, math, and coding. JudgeBench leverages a novel pipeline for converting existing difficult datasets into challenging response pairs with preference labels reflecting objective correctness. Our comprehensive evaluation on a collection of prompted judges, fine-tuned judges, multi-agent judges, and reward models shows that JudgeBench poses a significantly greater challenge than previous benchmarks, with many strong models (e.g. GPT-4o) performing just slightly better than random guessing. Overall, JudgeBench offers a reliable platform for assessing increasingly advanced LLM-based judges. Data and code are available at \url{https://github.com/ScalerLab/JudgeBench}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Currently, many approaches try to employ LLM-Based judges as a cheap and scalable alternative to human judges. However, their limitations remain unknown. While most previous works regard crowdsourced human annotations as ground-truth answers, these evaluations may have mistakes if the questions grow complex in reasoning. 
Therefore, the authors present JudgeBench, a benchmark to evaluate LLM-based judges, prioritizing factual and logical correctness. Firstly, the authors propose a pipeline for constructing such a dataset: Given a dataset with ground truth answer labels, they sample k responses from a strong model (e.g., GPT-4o). Then, they filter out questions where all k response are either all correct or all incorrect. They construct response pairs (correct and incorrect) from the remaining questions. Secondly, the pipeline is used to collect a dataset in four distinct categories: knowledge, reasoning, mathematics, and coding. A total of 350 questions are collected.
The resulting dataset is the most challenging for evaluating LLM-based judges. The authors evaluated a number of LLM-based judges on the dataset, ranging from vanilla judges to fine-tuned judges. The best model has a 57.43 performance overall.

### Strengths
1. The authors include extensive baselines of critical models and methods. The results lead to intriguing insights on how LLM-based judges fall short in evaluating challenging questions.
2. The motivation is very clear, which is creating a challenging benchmark to assess LLM judge's abilities in evaluating complex and reasoning-heavy problems.

### Weaknesses
While the experiments generally show that the resulting benchmark is challenging and has good separability, I believe there are more points that could be further dug into. The limited analysis makes the paper's contribution less significant.

1. The "key insight" mentioned in L229-231 is not verified carefully. The authors state: "if a model struggles to consistently generate correct, coherent responses to a challenging question, it will find it difficult to differentiate between those responses." From Table 4, this doesn't seem to be the case for all setups, especially for reasoning, math, and coding, where we see that stronger models can be weaker judges (Claude-3.5 and Llama-3.1 in reasoning).
2. Also, in Table 1, we see that many LLMs-judges perform less than random-chance. For example, JudgeLM gets a 11.96 accuracy in evaluating coding questions. Does this mean it always predicts the opposite answer? Because that would be another type of bias. There is no failure case analysis or type analysis for these LLM-based judges.
3. In paper [1], the authors notice that reasoning-heavy questions are hard for LLM-based judges to evaluate. Therefore, it is better for them to first construct its own reference answer, and then evaluate the candidate answer. They call this the reference-based judge. Have you considered similar simple prompting strategies to improve performance? How would they perform?
4. In the section on L426, the authors simply state that advancing the reasoning ability of LLM-based judges could help increase performance. This is a rather generic conclusion. It would be nice to have more contributions in producing a better LLM judge that could score higher on JudgeBench, using the insights that the authors have discovered. If not so, this paper stands only on the analysis level. And the analysis is not so deep.
5. The amount of questions collected (350) is small. Could you show the confidence intervals of the resulting estimations?

Reference:
[1] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

### Questions
1. For your description in L286-298 on position bias: as this is a benchmarking dataset, you also allow room for ambiguity and position bias in LLM-based judges. Wouldn't this also be an aspect to evaluate? I.e., if the judgment results change, the judges should be discredited on their discrepancy. There could be a metric to quantify such position bias.

see weaknesses.

### Soundness
2

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
2

### Summary
This article introduces JudgeBench, a novel benchmark for evaluating LLM judges, focusing specifically on their ability to assess factual and logical correctness in complex responses. They developed a pipeline to convert existing datasets with real labels into challenging response pairs, creating a benchmark of 350 pairs in the domains of knowledge, reasoning, mathematics, and coding. The key innovation is focusing on cases where even strong LLMs struggle to consistently generate correct answers, making the benchmark particularly challenging.

### Strengths
1. This paper aims to address the pressing need for a reliable method/benchmark to evaluate LLM judges as LLMs become more advanced and task complexity continues to scale up. 
2. This work pinpoints a critical gap in existing benchmark frameworks for LLM judges by concentrating on factual/logical correctness instead of just instruction following ability or human preference alignment.
3. The proposed benchmark is significantly more challenging than existing benchmarks on LLM judges.
4. The paper proposes a novel pipeline to generate challenging response pairs tailored for LLM judges.

### Weaknesses
1. The benchmark only has 350 response pairs, and it's unclear if this size is sufficient to obtain robust conclusions about LLM judge performance. Will increasing the size to 500 or 1000 change the performance rank of different LLM judges and reward models?

2. Table 1 shows that the proposed benchmark is quite challenging, with 10 out of 14 evaluated models/judges performing far below 50% accuracy. It is counterintuitive to me that most evaluated models/judges perform much worse than random guessing (50% accuracy). Are there any superficial artifacts in this benchmark that might lead these LLM judges to incorrect answers? Please correct me if there is any misunderstanding.

3. The paper would benefit from more extensive validation to ensure that the selected response pairs truly represent challenging cases.

4. The key insight—"If a model struggles to consistently generate correct, coherent responses to a challenging question, it will find it difficult to differentiate between those responses"—utilized in this paper is not empirically verified.

### Questions
Please see the weakness section above. 

Additionally:

When the data collected in this paper is released online, there is a possibility that it may be included in future LLM judge training, raising concerns about data contamination. This issue could impact the benchmark’s long-term effectiveness. Are there practices in place to prevent this and ensure the benchmark remains reliable over time?

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
- The paper proposes a new benchmark, *Judge Bench*, for evaluating large language models (LLMs) acting as judges, with a focus on knowledge, reasoning, math, and coding skills.
- Existing LLM-as-judge evaluations prioritize human preferences, assuming these preferences are reliable indicators of quality. However, this assumption can be weak for complex tasks, such as verifying mathematical proofs, where crowd-sourced annotators may lack the necessary domain expertise.
    - These existing benchmarks tend to emphasize stylistic preferences or instruction-following behaviors rather than logical accuracy and factual correctness.
- *Judge Bench* argues that effective evaluations should consider three core principles:
    - **Principle 1:** Responses must follow human instructions accurately.
    - **Principle 2:** Responses must be factually and logically correct.
    - **Principle 3:** Responses should stylistically align with human preferences.
- **Key contributions:**
    - A straightforward approach to generating evaluation data for LLM-as-judge models.
    - *Judge Bench*, a benchmark consisting of 360 challenging data points.

### Strengths
- Highlights the overemphasis on stylistic preferences in existing LLM-as-judge benchmarks, often at the expense of true task completion.
- Significant finding that many current LLM-as-judge models perform close to a random baseline on *Judge Bench*, revealing the difficulty of the prompts and the limitations of existing models.

### Weaknesses
- The benchmark includes only 360 examples, much smaller than many existing benchmarks such as Reward Bench.
- Beyond accuracy metrics, failure cases would add valuable insights into judge performance.
- Although the paper claims to apply a hierarchical approach, it appears that Principles 1 and 3 are largely overlooked in constructing *Judge Bench*.

### Questions
- It is interesting that the accuracy of judges drops when they are tasked with self-evaluation, which seems to contradict the prevailing research advocating for self-assessment or self-critique in LLMs. A deeper discussion on this aspect would be beneficial.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces JudgeBench, a new benchmark designed to evaluate LLM-based judges on complex tasks across knowledge, reasoning, math, and coding domains. While LLM-based judges are widely used as scalable alternatives to human evaluation, their reliability is often taken for granted. JudgeBench addresses this gap by objectively evaluating LLM-based judges, particularly in tasks where human preference alone may not accurately indicate factual correctness. The benchmark employs a novel pipeline to convert challenging datasets into response pairs with preference labels for objective evaluation. Experiments show that JudgeBench is significantly more challenging than existing benchmarks, with many strong models performing only slightly better than random guessing, thus providing a reliable framework for assessing advanced LLM-based judges.

### Strengths
- This paper is well-written and well-motivated, with a strong emphasis on the motivation, which is crucial in the LLM-as-a-Judge domain.
- The paper proposes a new benchmark dataset for LLM-based judging, called JudgeBench, and demonstrates the effectiveness and challenging nature of their proposed dataset compared to other similar judging benchmarks, which is a significant contribution in the field of new dataset development.
- The authors conduct extensive experiments to demonstrate JudgeBench's effectiveness, testing various baseline methods and comparing it with different judging benchmarks. Additionally, they perform comparisons with reward models and RewardBench and investigate potential biases commonly recognized in the LLM-as-a-Judge research field.

### Weaknesses
- There is no technical novelty; however, considering that this paper proposes a novel benchmark dataset, this is acceptable, as the paper makes a significant contribution to this field.
- The authors conduct extensive experiments; however, I have a few suggestions/questions that do not impact the overall rating score:
  - The authors investigate various biases, but I am curious about length bias, a well-documented issue in this field where LLM-based judging models tend to prefer longer responses over shorter ones. My question is: when constructing the dataset, did the authors consider the length difference between "correct" and "incorrect" answers?
  - In Tables 1 and 2, the performance in the "Math" domain is relatively higher than in other domains (e.g., Knowledge, Coding). I am concerned that the "Math" domain samples in JudgeBench may be less challenging compared to other domains. Additionally, does the JudgeBench dataset include subcategories? For example, within "Math," are there various categories based on subjects (e.g., geometry), and within "Coding," are different programming languages and problem difficulties represented? If so, could the authors present detailed experimental results based on these finer-grained dataset samples?

### Questions
Please refer to Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4
