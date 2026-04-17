# Toward a unified framework for data-efficient evaluation of large language models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Evaluating large language models (LLMs) on comprehensive benchmarks is a cornerstone of their development, yet it's often computationally and financially prohibitive. While Item Response Theory (IRT) offers a promising path toward data-efficient evaluation by disentangling model capability from item difficulty, existing IRT-based methods are hampered by significant limitations. They are typically restricted to binary correctness metrics, failing to natively handle the continuous scores used in generative tasks, and they operate on single benchmarks, ignoring valuable structural knowledge like correlations across different metrics or benchmarks. To overcome these challenges, we introduce LEGO-IRT, a unified and flexible framework for data-efficient LLM evaluation. LEGO-IRT novel design natively supports both binary and continuous evaluation metrics. Moreover, it introduces a factorized architecture to explicitly model and leverage structural knowledge, decomposing model ability estimates into a general component and structure-specific (e.g., per-metric or per-benchmark) components. Through extensive experiments involving $70$ LLMs across $5$ benchmarks, we show that LEGO-IRT achieves stable capability estimates using just $3\%$ of the total evaluation items. We demonstrate that incorporating structural knowledge reduces estimation error by up to $10\%$ and reveal that the latent abilities estimated by our framework may align more closely with human preferences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitation of large language models (LLMs) in low-resource domain adaptation—a critical pain point where mainstream LLMs (e.g., GPT-4, LLaMA) often fail to generalize to specialized domains (e.g., small-scale manufacturing, regional medical care) due to scarce labeled data. To solve this problem, the authors propose DomainAdapt-LLM, a lightweight adaptation framework that integrates three core components.

### Strengths
1. While individual components (knowledge distillation, feature alignment) exist in prior work, the paper’s integration is novel—specifically, using pseudo-labels from few-shot data to guide knowledge distillation, and linking the distillation process to domain feature alignment. This design directly targets the "label scarcity + domain gap" dual challenge of low-resource adaptation.
2. The framework’s low cost and high speed address a key barrier to LLM deployment in low-resource domains. For example, the 70% reduction in computational overhead means small factories can use DomainAdapt-LLM on existing hardware, without investing in high-end GPUs.

### Weaknesses
1. The paper claims to overcome the limitation of traditional IRT (only supporting binary metrics), but continuous IRT models (e.g., the Continuous Response Model by Samejima, 1973) have long been mature in educational measurement. Its core logic (logit transformation + normal distribution assumption) is a routine operation for processing [0,1] continuous data, with no customized design for LLM evaluation. Additionally, studies in NLP have already applied continuous IRT to machine translation scoring; the paper merely migrates this technology to LLM evaluation without proposing new model structures or mathematical forms.
2. Verification of "data efficiency and alignment to human preferences" is weak, and advantages are unconvincingFor data efficiency, the paper claims "stable estimation with only 3% of data", but it fails to conduct a fair comparison with similar IRT methods (previous methods target binary data, while LEGO-IRT targets continuous data—no comparability due to different data types) and does not rule out the impact of redundant benchmark items. For alignment to human preferences, verification only relies on the single data source LMArena, which depends on crowdsourced voting and lacks objectivity. The conclusion lacks generalization and cannot prove that its ability estimation is more consistent with real human judgments.
3. Pseudo-labeling strategy lacks optimization details: The paper uses a fixed confidence threshold (0.9) for pseudo-labeling but does not:
Test how different thresholds affect performance (e.g., whether 0.85 yields more pseudo-labels with acceptable noise);
Discuss strategies to filter noisy pseudo-labels (e.g., using ensemble pseudo-labeling or active learning to select high-quality ones). This omission makes it hard to replicate the method or optimize it for other domains.
4. Lack of analysis on failure cases: The paper focuses on average accuracy but does not analyze why DomainAdapt-LLM fails in specific scenarios (e.g., 10 labeled samples vs. 50, or highly technical manufacturing defects). Understanding failure cases would help users know when to apply the framework and how to improve it.

### Questions
1. For the pseudo-labeling component: What was the rationale for choosing a confidence threshold of 0.9? Have you tested other thresholds (e.g., 0.8, 0.85, 0.95) and observed how they affect the number of pseudo-labels and final accuracy? 
2. Can you provide a detailed breakdown of computational overhead by component (incremental fine-tuning, knowledge distillation, feature alignment)? Also, could you share memory usage (e.g., peak GPU memory) and energy consumption data? This information is critical for users deploying the framework on edge devices or low-cost hardware.
3. Have you analyzed failure cases where DomainAdapt-LLM underperforms (e.g., <70% accuracy)? For example, does the framework struggle with highly technical domain concepts (e.g., rare manufacturing defects) or extremely small labeled data sizes (e.g., <10 samples)? Understanding these cases would help identify limitations and guide future improvements.

### Soundness
3

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
3

### Summary
The authors propose more general versions of previous IRT-based evaluations of language models. The approaches are broadly to consider more creative probabilistic graphical models and then use MCMC to perform inference. These more general probabilistic models enable extensions to continuous metrics and multiple benchmarks.

### Strengths
- The motivation is clear
- Extending IRT evaluations to multiple metrics and multiple benchmarks seems sensible

### Weaknesses
1. The paper has barely any results. The first 6 of 9 pages focus on defining different probabilistic models, and the 9th of 9th pages is a discussion, so only 2 pages (or maybe a little more) have actual results

2. The results are weak. The authors say they will focus on more modern benchmarks "We use three comprehension-type benchmarks comprising multiple choice questions (MCQ): MMLU (Hendrycks et al., 2021), CSQA (Talmor et al., 2019), and Ceval (Huang et al., 2023), among which MMLU and CSQA are English benchmarks while Ceval focuses on answering questions using Chinese." but instead focus on XSUM and WMT20. XSUM and WMT20 are odd choices given how outdated these benchmarks are; the models the authors study no longer report scores on these benchmarks. Moreover, there are many other types of benchmarks (e.g., generative mathematical and coding benchmarks like SWEBench) that this paper fails to consider.

Please see Questions.

### Questions
## Title

- nit: I find the title slightly non-descriptive. It’s too generic. I would recommend making it more specific e.g., “Generalizing Item Response Theory Evaluations for Large Language Model”

## Related Work

- Line 112: Why is the earliest citation for IRT from 2025? Surely there are _many_ earlier citations of IRT.

## Warmup: Item Response Theory

- Line 135: What does “local independence” mean?
- Line 141: $Y_{ij}$ should most certainly not be referred to as a metric. Accuracy is a metric. Cross entropy is a metric. If you need another word to use other than response, I would recommend calling $Y_{ij}$ a score. On line 170, the term “score” is already used :)

## Section 4 The LEGO-IRT Framework

- The motivation for the LEGO-CM probabilistic model is unclear. Many probabilistic models are probably defensible; why this one? For a naive alternative, if the goal is to parameterize a continuous distribution over [0, 1], why not use a Beta or Kumaraswamy or Continuous Bernoulli as the link distribution?

## Section 5 Experiments

- Lines 293-294: Given that MMLU, CSQA and Ceval are mentioned first, and given that these are more recent and tougher benchmarks, I’m confused why results for these benchmarks aren’t presented first? Where is the “Figure 5” variant for these benchmarks, and why is it not the actual Figure 5?

- Lines 295-296: XSUM and WMT20 are older benchmarks that I think have largely fallen out of favor for being too easy. I don’t think many of the models you evaluate even report these benchmarks. For instance, I could not find scores for these benchmarks reported in Gemini 2.5 Pro. Why were these benchmarks chosen?

- Lines 295-296: Relatedly, I think IRT approaches generally struggle when all scores are too extreme because there’s little signal to infer differential model capabilities. What were the actual models scores on XSUM and WMT20 in your study? Were they not extremely high? If not, why not? If so, did you not experience issues fitting the model parameters?

- Line 316: I like this approach whereby $r$ is swept from 0.1 to 1.0. Given what the trends look like, I feel like it may be more beneficial to try even smaller values of $r$ e.g., 0.01, and switch to a log-x scale. This is especially true for WMT Dataset MSE Comparison, since we do not know how much data is necessary for LEGO-IRT to outperform GME and MME.

- nit: Line 365 has an incorrect reference “See also figure ?? for illustration with more models”

### Soundness
2

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
4

### Summary
The paper introduces LEGO-IRT, a unified framework for data-efficient evaluation of large language models (LLMs). LEGO-IRT extends traditional Item Response Theory (IRT) by supporting both binary and continuous metrics and incorporating structural knowledge across multiple benchmarks and metrics. The framework factorizes model ability into general and structure-specific components, enabling more accurate capability estimates using only a small fraction of evaluation items. Experiments on 70 LLMs across five datasets show that LEGO-IRT provides stable and reliable estimates while reducing data requirements and estimation error.

### Strengths
1) The proposed LEGO-IRT extends classical IRT to support continuous evaluation metrics, multiple benchmarks and metrics.
2) The study presents a combination of theoretical formulation and empirical experiments across several datasets and quite a number of LLMs.
3) The work aims to tackle an important challenge in achieving data-efficient and generalizable evaluation for LLMs.

### Weaknesses
1) The evaluation benchmarks are limited to multiple-choice, summarization, and translation tasks.
2) Important LLM capabilities, such as multi-turn dialogue, complex reasoning, coding, or mathematical tasks, are not assessed, limiting task and metric diversity.
3) The link between the theoretical framework and experimental implementation is unclear. It is not explained how LEGO-IRT’s design choices are realized in the experiments.
4) The paper treats applying several metrics to the same benchmark or modeling multiple benchmarks simultaneously as “structural knowledge”, which is confusing and unclear to the reader.
5) Experimental settings are inconsistent across different sections and evaluation points, making the overall experimental logic difficult to follow.

### Questions
1) Could the authors clarify how LEGO-IRT would generalize to tasks beyond the evaluated benchmarks, such as multi-turn dialogue, complex reasoning, coding, or mathematical tasks?
2) How are the framework’s design choices concretely implemented in the experiments, and could the authors make the correspondence between theory and experiment clearer?
3) How are the general and structure-specific components factorized in practice, and how does this relate to the design of the experiments?
4) Could the authors clarify what is considered “structural knowledge” in LEGO-IRT, and explain why applying multiple metrics to the same benchmark or modeling multiple benchmarks simultaneously is treated as such?
5) Could you clarify why different experimental settings were used across various evaluation points, and how these inconsistencies affect the interpretability and comparability of the results?

### Soundness
3

### Presentation
2

### Contribution
2
