# Do LLMs Align with My Task? Evaluating Text-to-SQL via Dataset Alignment

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Supervised Fine-Tuning (SFT) is an effective method for adapting Large Language Models (LLMs) on down-stream tasks. However, variability in training data can hinder a model's ability to generalize across domains. This paper studies the problem of \textit{dataset alignment} for Natural Language to SQL (NL2SQL or text-to-SQL), examining how well SFT training data matches the structural characteristics of target queries and how this alignment impacts model performance.
We propose the Alignment Ratio (AR), a metric that quantifies structural alignment and can serve as a predictive, actionable decision criterion to select or filter fine-tuning datasets.
We hypothesize that alignment can be accurately estimated by comparing the distributions of structural SQL features across the training set, target data, and the model’s predictions prior to SFT. 
Through comprehensive experiments on three large cross-domain NL2SQL benchmarks and multiple model families, we show that structural alignment is a strong predictor of fine-tuning success. When alignment is high, SFT yields substantial gains in accuracy and SQL generation quality; when alignment is low, improvements are marginal or absent. These findings highlight the importance of alignment-aware data selection for effective fine-tuning and generalization in NL2SQL tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how dataset alignment affects supervised fine-tuning for Text-to-SQL models. The authors propose a KL-based metric to quantify the structural similarity between SFT training data and target SQL queries, and show that this alignment strongly predicts post-SFT performance. Extensive experiments across multiple NL2SQL benchmarks and model families confirm that high structural alignment leads to substantial gains in accuracy and SQL generation quality, while low alignment yields limited improvements or even performance degradation.

### Strengths
1. The paper provides extensive experiments across diverse NL2SQL benchmarks and multiple LLM families, offering convincing evidence that the proposed KL-based alignment metric is strongly correlated with post-SFT improvements. 

2. The narrative is well structured and easy to follow, with a clear motivation, carefully articulated methodology, and thorough analysis.

### Weaknesses
1. The central conclusion — that structurally aligned fine-tuning data yield better SFT performance — is fairly straightforward and offers limited deeper insight beyond confirming the intuitive notion that “more similar data helps”.

2. The methodology of applying the Kullback‑Leibler divergence (KL-divergence) to measure dataset alignment or support data selection is not particularly novel. Prior literature has used KL divergence for distributional comparison and subset selection [1, 2], which somewhat diminishes the originality of the methodological contribution.

3. The reliance on SQL skeletons (query templates) for computing structural statistics restricts the generality of the approach: by design the method is tightly coupled to the SQL domain and may not extend easily to tasks with less rigid templated structure or to downstream applications beyond NL2SQL.

[1] Everaert D, Potts C. Gio: Gradient information optimization for training dataset selection[J]. arXiv preprint arXiv:2306.11670, 2023.

[2] Kurian J F, Allali M. Detecting drifts in data streams using Kullback-Leibler (KL) divergence measure for data engineering applications[J]. Journal of Data, Information and Management, 2024, 6(3): 207-216.

### Questions
None.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the role of dataset alignment in supervised fine-tuning (SFT) for text-to-SQL tasks. The authors propose a KL-alignment metric based on structural SQL features to measure how well training data matches target query distributions. Through extensive experiments on multiple benchmarks and model families, the authors demonstrate that alignment strongly predicts SFT success and generalization.

### Strengths
1. The observation of this paper has certain value for subsequent post-training.
2. The evaluation model is comprehensive.

### Weaknesses
1. Whether the proposed alignment prediction framework can directly improve the SFT performance, there is a lack of a clear method to directly improve the performance of SFT.
2. A large amount of related work published in 2025 was not discussed.
3. No automated data selection or valid tuning method is proposed—only a diagnostic metric. I believe alignment will be effective, but I cannot verify it at this stage.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

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
This paper explores the problem of dataset alignment in supervised fine-tuning (SFT) for Natural Language to SQL (NL2SQL) tasks. Essentially, the authors ask: how well do the characteristics of the training data align with those of the target queries? They hypothesize and empirically show that when SFT data is well-aligned with the structural patterns in the target data, the resulting fine-tuned models perform much better. The paper formalizes a KL-alignment metric and an alignment ratio (based on distributions of SQL n-grams/templates) and demonstrates that these predict the success of fine-tuning. Through comprehensive experiments on several cross-domain NL2SQL benchmarks (BIRD, Spider, Gretel) and a range of LLM families (Qwen, CodeLlama, Deepseek), they show high alignment correlates with strong gains in execution and exact match accuracy, while low alignment can yield little to no improvement (sometimes even degrading performance). The authors also propose a simple framework for predicting post-SFT performance before actually fine-tuning, which can help practitioners select training datasets more strategically.

### Strengths
Identifies and systematically formalizes the effect of dataset alignment in NL2SQL fine-tuning. Introduces alignment metrics (KL-alignment, ratio) that not only measure but also predict transfer learning success or failure before SFT. Comprehensive empirical study spanning a wide model and dataset range; clear, robust trends. Shows practical use: enables practitioners to avoid wasted effort/failures due to misaligned data. Readily reusable ideas and framework could be adapted to other semi-structured outputs, e.g., code generation.

### Weaknesses
KL-alignment is focused on syntactic distribution. it may not capture semantic nuances or correctness of the generated SQL, so has limits as a universal proxy. Statistical trends could be more explicit e.g., when/how often does high alignment fail to predict actual gains?. Limited discussion on how much alignment is enough for different problem scales or domains, and what to do when no dataset aligns well. Technical explanations  like calculation of features, practical computation of large n-gram sets might be too heavy for non-experts. While the approach is generalizable, the actual experiments only show text-to-SQL. extension to other structured seq2seq tasks is not explored or discussed.

### Questions
Do you have plans or suggestions for alignment metrics that could also measure semantic or functional compatibility e.g., for queries with equivalent meaning but different syntax? How would you recommend users act when no candidate training set aligns well, is few-shot prompting viable, or is new data collection unavoidable? Can the alignment prediction/generalization story be extended to very large LLMs with more “universal” prior coverage? Did you find any real-world settings outside the chosen benchmarks where KL-alignment failed to track actual SFT performance?

### Soundness
4

### Presentation
3

### Contribution
2
