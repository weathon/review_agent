# Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
While Test-Time Scaling (TTS) effectively enhances the reasoning capabilities of Large Language Models (LLMs), its potential is often bottlenecked by low output diversity. This limitation raises questions about the standard one problem, one solution (1P1S) fine-tuning paradigm, which, by rewarding a single canonical answer, may encourage models to overfit to specific reasoning paths. To address this, we argue that adopting a one problem, multiple solutions (1PNS) training paradigm is crucial for cultivating reasoning diversity and unlocking the full potential of LLM reasoning. However, a central challenge of this paradigm lies in quantifying the semantic difference between complex, multi-step reasoning paths. To address this, we introduce Reasoning Path Divergence (RPD), a novel, fine-grained metric that operates at the step-level of Long Chain-of-Thought solutions. Using RPD, we curate a training set composed of maximally diverse solutions for each problem. Experiments with Qwen3-4B-Base demonstrate that training on our RPD-curated data significantly enhances output diversity and yields substantial gains in pass@k performance. Specifically, our 1PNS approach surpasses the 1P1S baseline by an average of 2.80\% on pass@16 across challenging math benchmarks, with the improvement reaching 4.99\% on AIME24, making Test-Time Scaling more effective.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the low diversity issue of traditional one-problem-one-solution fine-tuning (1P1S) paradigm, this paper proposes an effective one-problem-multiple-solutions fine-tuning method based on a newly proposed metric-Reasoning Path Divergence (RPD). Specifically, after obtaining multiple solutions to a problem, the authors first leverage a LLM to decompose each solution into several summarized core steps. Then, for each two solutions, the authors calculate the semantic distance between each step in one solution to all steps in another solution. The RPD score is calculated as the average calculated semantic distances over all steps in that solution. Selecting solutions with higher RPD scores with other solutions can bring more diversity to the dataset, and fine-tuning the model on this dataset helps to improve the Pass@K metric during inference.

### Strengths
(1) The paper is generally well-written, and the structure is clear.

(2) The proposed metric, Reasoning Path Divergence, although somewhat complex and intricate, remains quite reasonable and intuitive.

(3) The authors try to conduct comprehensive ablations to study the effectiveness of the method.

### Weaknesses
1. First, the calculation process of RPD is quite complex and unstable. Its computation heavily relies on decomposing the original solutions using an external LLM, a process that is not always accurate and reproducible, and is highly dependent on the LLM's instruction-following capabilities and designed prompts.

2. The experiments are conducted on a very small sample size (300 samples), which makes the results and findings potentially unreliable. The authors should perform experiments on a larger-scale dataset.

3. The experiments are only conducted on Qwen models, lacking the validation of the method's generalizability on other model architectures, such as Llama.

4. The number of evaluation sets is limited. The sample sizes of AIME24 and MATH500-Level are limited. The authors are encouraged to perform thorough evaluations on more test sets.

5. The included baselines are naive and weak. A strong baseline should be training the model on all the original, unfiltered data. If similar or better response diversity and performance can be achieved without filtering, then the need for extra data filtering seems less justified.

6. I am quite interested in whether the proposed method or trained model in this paper can bring improvements in subsequent RL training, leveraging the higher Pass@K results.

### Questions
1. The authors compared the performance of various methods after implementing different data filtering measures. However, a direct baseline would involve using the original, unfiltered data for SFT without any additional data screening. If similar or better response diversity and performance can be achieved without filtering, then the need for extra data filtering seems less justified.

2. Why choose LoRA fine-tuning instead of full parameter fine-tuning?

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
4

### Summary
The paper proposes a technique for selecting the SFT training dataset for better output diversity. In the context of LLM reasoning with math, the paper first posits that the current models suffer from a diversity issue. Then, it introduces a two-stage process that selects a subset of SFT data. They show that after tuning on this SFT data, the generated solutions on the test set achieve high accuracy under majority voting.

### Strengths
* clear writing
* identifies an important problem
* method description + experiment execution is sound

### Weaknesses
* I don't think diversity is a serious issue for math&code reasoning problems. They tend to be a problem for more subjective tasks. The problem domain selected by the author seems contrived --  i.e. since we have readily available benchmarks and datasets in math, let's do math
* There should be a temperature scaling for the majority vote. It's unclear why the authors stop at T=1 (Table 6). Also, even given the results in Table 6, it's clear that the marginal benefit of RPD diminishes as temperatures rise. Even DS-R1 (https://api-docs.deepseek.com/quick_start/parameter_settings) suggests production temperature up to 1.5. I think the author should sweep T much higher.
* The experiment setup is very contributed. OpenThought is a distillation dataset. Nobody uses it in actual production. The real question regarding diversity is how we should preserve output diversity during RLVR? The paper identifies the right question, but provides a solution in a setting (SFT) where the question is nonexistent.
* Even in the distillation setting, a natural baseline is the following: for each question in OpenThought, distill multiple solutions at a higher temperature, and then SFT on the new synthetic data.

### Questions
See weakness

### Soundness
2

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
3

### Summary
This paper introduces Reasoning Path Divergence (RPD), a step-level semantic distance metric designed to quantify diversity among long chain-of-thought (CoT) reasoning paths in LLMs. Using RPD, the authors curate a one-problem-multiple-solutions (1PNS) training set from the OpenThought3 dataset and fine-tune a 4B-parameter model (Qwen3-Base). The proposed method improves pass@16 performance by 2.80% on average across three math benchmarks, with a peak gain of 4.99% on AIME24. The paper argues that training with diverse reasoning paths mitigates output homogenization and enhances test-time scaling (TTS).

### Strengths
### 1. Novel and Well-Motivated Metric (RPD):

RPD is a creative and principled approach to measuring semantic diversity at the step level, addressing a key limitation of embedding-based methods that conflate surface-level differences with strategic divergence. The asymmetric design is particularly insightful for handling summarization granularity.


###  2. Thorough Experimental Design:

The paper includes extensive ablations, scalability tests, and diversity analyses. The authors also validate their LLM judge against human annotations (78% accuracy), lending credibility to their automated evaluation pipeline.

### Weaknesses
### 1. Limited Generalization Beyond Math

All experiments are conducted on math reasoning tasks (AIME24, MATH500, Olympiad Bench). While the gains are convincing, it is unclear whether RPD and 1PNS generalize to other reasoning domains (e.g., logic, science, coding), limiting the broader impact of the work.

### 2. Scalability and Compute Overhead

RPD relies on LLM-based summarization and embedding computation for every solution pair, which is compute-intensive and may not scale well to larger datasets or real-time curation. The paper does not discuss the computational cost of RPD compared to simpler baselines (e.g., raw embeddings).

###  3. Lack of Theoretical Justification

While RPD is intuitively appealing, the paper does not provide theoretical grounding for why step-level asymmetric distance correlates with strategic diversity or why 1PNS improves TTS. The connection between training data diversity and inference-time exploration remains largely empirical.

### Questions
see weakness.

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
This paper addresses low output diversity by introducing Reasoning Path Divergence (RPD), a step-level metric for quantifying semantic diversity between reasoning paths. Using RPD, the authors curate a one-problem–multiple-solutions (1PNS) training dataset that maximizes strategic variation across Long-CoT solutions. Fine-tuning a Qwen3-4B-Base model on this RPD-curated dataset yields consistent gains over standard one-problem–one-solution (1P1S) baselines. The method increases reasoning diversity without requiring architectural or inference-time changes.

### Strengths
- The metric design is clear and reasonable. RPD introduces a fine-grained, asymmetric step-level comparison that captures strategic rather than superficial differences. 
- Consistent improvements across AIME24, MATH500, and OlympiadBench, with robust ablations on number of solutions, problem selection, and temperature scaling.

### Weaknesses
- Evaluation limited to math reasoning: The study focuses exclusively on quantitative tasks (AIME, MATH, Olympiad); generalization to open-ended or commonsense reasoning remains unclear.

- The curation pipeline—step summarization, embedding, and pairwise distance computation—may be costly for larger datasets.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3
