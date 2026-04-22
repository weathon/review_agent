# AlphaBench: Benchmarking Large Language Models in Formulaic Alpha Factor Mining

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Formulaic alpha factor mining (FAFM) is a central problem in quantitative investment, where interpretable formulas are designed to extract predictive signals from historical financial series. With the emergence of large language models (LLMs), recent studies have begun to explore their roles in FAFM, yet their capabilities across different tasks and configurations remain unclear. In this work, we introduce AlphaBench, the first systematic benchmark for evaluating LLMs in FAFM. AlphaBench covers three core tasks, including factor generation, factor evaluation, and factor searching, which are all popular tasks integrated in the workflow of quantitative researchers. Beyond task-level evaluation, we further analyze how different LLM settings, including model type, prompting paradigm, and reasoning strategy, influence performance. Our experiments on a range of open-source and closed-source models reveal that LLMs hold strong potential in automating factor mining, while also facing persistent challenges in robustness, search efficiency, and practical usability. The project is available at: https://alphabench.cc/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a benchmark to evaluate how large language models perform in quantitative finance tasks involving the generation, evaluation, and searching of interpretable factors. Using the CSI300 market data and Qlib framework, the benchmark tests models across metrics such as reliability, stability, accuracy, and cost. Results show that while LLMs can reliably produce syntactically valid and intuitive factor expressions, they struggle to assess factor quality without data, and reasoning strategies offer limited benefit.

### Strengths
The paper is well-written and clearly organized. I commend the authors for the scope of their analyses. As a potential user of AlphaBench, I believe that the benchmark dataset will be of interest to many researchers.

It is good to see that frontier models already achieved a supreme performance in generations tasks. The models also achieved a moderate performance in search tasks.

### Weaknesses
My concerns are primarily related to the evaluation task. Given its statistical nature, probably LLMs are not well-suited for factor evaluation. LLMs might help human quant researchers to construct creative new factors but it might be better to outsource quantitative assessment to traditional backtesting models.

Section 4.3 reports very weak zero-shot evaluation performance, which is near random. The discussion (Section 5) attributes this mainly to missing supervision and execution context. Although this explanation is convincing, the paper can further test the following:
(1) Provide a controlled ablation showing how much of this failure is due to lack of numeric context versus model scale or prompt design.
(2) Demonstrate even a small fine-tuning (e.g., adding example factor–IC pairs) to show whether performance improves. This would make the benchmark actionable for future model development.

The authors implicitly make two foundational assumptions:
 (1) LLMs can “understand” factor expressions semantically, and
 (2) they can reason about predictive strength without data
These assumptions are not empirically grounded. Alpha factors embed statistical relations, not linguistic ones; their performance often depends on noise sensitivity, normalization, market characteristics, time-specific regimes, and data-specific effects that linguistic/symbolic interpretation cannot capture. Hence, the current design may be more a test of syntactic familiarity (e.g., knowing that Mean + Std are stable factors) rather than genuine financial reasoning.
This can only be tested if the authors can show that human experts (quant researchers) can perform the same ranking/scoring from formulas alone (without access to numerical data). This would establish an upper bound for LLM expectations.

Currently, if I did not misunderstand, ground truth labels are derived from backtests on the same CSI300 data used to generate the factors. If the LLM learned statistical patterns from similar data distributions (e.g., public quant research corpora), then its apparent understanding could partially reflect memorization, not reasoning. A stronger design would use out-of-sample or cross-market validation (e.g., use CSI300 for training labels and SP500 for testing).

Another potential concern is that the paper’s contributions may appeal to a financial AI audience rather than the broader general AI community. It is difficult to identify insights that generalize beyond the domain of quantitative finance. The authors could strengthen the paper’s broader relevance by adding a sentence or two in the conclusion to highlight how AlphaBench’s methodology can inform LLM benchmarking or reasoning research in other structured domains.

### Questions
Please see above (weaknesses).

### Soundness
3

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
This paper introduces AlphaBench, the first benchmark for evaluating LLMs in Alpha Factor Mining, where it's an important task in quantitative finance focused on discovering interpretable mathematical expressions that predict asset returns. 

The benchmark includes: 687 generation prompts, 1170 evaluation instructions, 27 search tasks, 10 LLMs, and different prompting strategies. AlphaBench covers comprehensive evaluation metrics for Generation, Evaluation, and Searching.

### Strengths
1. The benchmark focuses on the factor mining task, but is comprehensive. The paper provides lots of insights: a) LLMs show high reliability in generating syntactically valid factors, but accuracy drops significantly for complex instructions. b) Factor evaluation remains a major bottleneck. 3) LLM-guided search improves factor quality at a reasonable computational cost; d) CoT sometimes doesn't work on large models. That's important for the FinTech domain.

2. The authors build on a real-market dataset using Qlib-compatible backtesting. They also provide detailed documentation for the curated dataset.

### Weaknesses
1. I have some confusion about the factor evaluation task. The factor evaluation (ranking/scoring) task sometimes outputs near-random results across all models. Could the author provide more explanation on that?

2. The dataset covers only daily equity factors on the CSI-300 (China) with ~1,700 factors. 

3. The search step is helpful, and search quality is measured via IC improvement from backtesting. But this could vary due to noise in short-term financial returns.

### Questions
See in Weaknesses

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
3

### Summary
1. The paper presents - AlphaBench which introduces first benchmark for evaluating LLMs in (FAFM) which is pretty noval and insightful
2. Three main pillars at which model measures are- generation, evaluation, and searching tasks for financial factors and their discovery.
3. A far I could evaluate in manuscript - GPT-5 performs best overall; 
4. Chain-of-Thought (CoT) yields minimal gains which was also expected, sometimes reduces stability.
5. For all models, evaluation or judgement remains a big challenge
6. Gemini models are competitive; open-source models lag behind.

### Strengths
1. A very detailed and comprehensive coverage of FAFM lifecycle.
2. Very good benchmarking with quantitative metrics (IC etc..)
3. The paper also presents a very strong validation on real financial datasets (CSI300, 2020–2025).
4. Manuscript also rightly highlights trade-offs between model size, cost.

### Weaknesses
1. LLMs struggle in factor evaluation — low accuracy in ranking/scoring that was observed on some experiments
2. CoT prompting often hurts large-model performance.
3. Lack of supervised data limits evaluation reliability. Some strong large scale data would be helpful
4. Benchmark restricted to daily equity factors; excludes intraday or multi-asset tests.

### Questions
1. How can supervised or weakly labeled data be created for training factor evaluators?
2. Can structured representations improve evaluation interpretability and what experiments can be done more on it?
5. What is the role of specific domain knowledge in guiding LLM outputs?

### Soundness
3

### Presentation
3

### Contribution
3
