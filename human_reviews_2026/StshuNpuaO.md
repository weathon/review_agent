# LayerMix Law: Scaling Law for Large Language Models on Quality-Weighted Mixture Data with Repetition

- Decision: Reject
- Scores: 6, 2, 8, 4

## Abstract
Upweighting high-quality data in large language model (LLM) pretraining typically improves performance. However, the limited availability of high-quality data—particularly in overtrained regimes—means that stronger upweighting often increases repetition, which can degrade performance. This creates a fundamental trade-off between data quality and data repetition. In this paper, we systematically investigate how varying data quality and repetition affects models across different scales. Concretely, we partition the source corpus into buckets based on quality scores and sample from each bucket with different weights, thereby constructing training sets with diverse scales, quality distributions, and repetition levels. We then train a family of models on these datasets to measure performance across conditions. Building on these observations, we introduce a theoretical framework analogous to scaling laws, which we call \textbf{LayerMix Law}. LayerMix Law predicts model loss as a function of consumed tokens, model size, sampling weights, and repetition levels. The key intuition is to view training as the accumulation of information from data, where the amount of information is governed by data quality, while model scale and repetition determine the information gained per training step. We show that LayerMix Law accurately predicts the model performance on unseen data recipes at larger computation scale (up to 7B parameter run with 425B token, each x2 invest compute), with 0.15\% average absolute error and 0.96\% maximum absolute error, which enables efficient search for optimal data recipes without costly additional experiments. Moreover, LayerMix Law extrapolates reliably to different degrees of overtraining, providing a efficient tool for selecting data recipes under varying computational budgets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LayerMix Law, a new scaling law framework designed to model large language model (LLM) performance when training data varies in quality and repetition. LayerMix Law explicitly captures the trade-off between using limited high-quality data (which may require repetition) and including lower-quality data to reduce overfitting. They propose an Information Quantity metric to quantify total learned information and demonstrate a power-law relationship between this metric and model loss. Empirically, across 27 training runs (252M–1.2B models) and extrapolations up to 7B parameters, LayerMix Law accurately predicts model loss under unseen mixtures and overtrain degrees with <1% error, providing a principled tool for data recipe optimization under data constraints.

### Strengths
1. Good Motivation: The paper tackles a critical and realistic problem in LLM training. It clearly explains why traditional scaling laws fail under such conditions and motivates the need for a data-aware scaling framework.
2. Comprehensive Experiments: Experiments are extensive and well-structured, covering multiple model sizes (252M–7B), data mixtures, and overtrain ratios. The consistent results across interpolation and extrapolation settings strongly support the proposed LayerMix Law.
3. Problem Formalization: The formulation of Information Quantity elegantly links data quality, repetition, and compute into a unified model. It provides a clear theoretical basis for understanding and predicting LLM performance across heterogeneous data mixtures.

### Weaknesses
1. Lack of Optimization Guidance: Although the law connects information gain to model loss, it does not analyze how to derive optimal mixture ratios or repetition levels under fixed compute. This limits its practical applicability for data recipe design.
2. Format and Clarity Issues:
Naming conventions like Q1_V1 and Q1_V2 are unintuitive, and equation formatting could be improved for readability.

### Questions
While the paper successfully establishes a theoretical connection between information gain and model performance, it stops short of providing actionable guidance on how to optimize data mixtures in practice. The LayerMix Law defines how loss depends on parameters like sampling weights $w_d$, token scale K, and repetition $R_d$, yet it does not analyze how to choose or optimize these variables to achieve the best results under a given compute budget. For example, once the relationship $L = \alpha \cdot \text{info}^{-\beta}$ is known, a natural next step would be to derive the optimal allocation of quality buckets $w_d$ that minimizes loss for fixed compute or token count. However, the paper does not explore this direction. There is no gradient-based or analytical discussion of how $w_d$, K, or S interact to yield an optimal configuration. Moreover, since the practical value of a scaling law lies in guiding future training strategies without brute-force grid search, the absence of such optimization analysis weakens the applicability of LayerMix Law for real-world data curation or scaling decisions. Including even a preliminary sensitivity or optimization study would make the framework much more actionable and impactful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces LayerMix Law, a data-aware scaling law for large language models (LLMs) that explicitly accounts for data quality and data repetition—two important factors often ignored by traditional scaling laws.
Existing scaling laws assume an unlimited supply of diverse, high-quality data and tend to fail when datasets are upsampled or repeated. 
However, real-world LLM pretraining frequently requires upweighting scarce high-quality data or reusing existing data, leading to a fundamental trade-off between data quality and diversity.
The authors study this trade-off and propose both a theoretical formulation and empirical validation of the resulting “LayerMix Law.” 
The framework extends classical scaling laws by embedding data-quality weights and repetition dynamics into performance prediction, offering a practical tool for selecting and optimizing pretraining datasets under limited compute and data availability.

### Strengths
1. The paper proposes a promising extension of scaling laws by incorporating data-quality weighting and repetition effects, which are crucial in today’s data-constrained LLM training.
2. The authors effectively combine ideas from scaling theory, data attribution, and mixture modeling into a single predictive law that connects information accumulation with model loss.
3. The paper reports comprehensive experiments—27 controlled pretraining runs (252M–1.2B parameters) with systematically varied data mixtures and repetition levels—and further validates extrapolation on 7B-parameter models. The resulting predictions achieve an average absolute error of only 0.15%, demonstrating impressive fit accuracy.
4. The paper is well-organized, clearly motivated, and easy to follow. Figures and explanations illustrate the intuition behind the proposed law, and the experimental design is well-documented.

### Weaknesses
1. The proposed information-theoretic formulation appears heuristic rather than theoretically derived. The framework would be more convincing if grounded in established information theory or accompanied by ablations comparing alternative functional forms.
2. All experiments are conducted on English Common Crawl data and relatively small models (≤7B). It remains unclear whether the fitted parameters generalize across domains (e.g., code, multilingual), which limits the broader applicability of the proposed law.
3. Although the paper cites related work (e.g., Data Mixing Laws (Ye et al., 2025), CMR Scaling Law (Gu et al., 2024)), it does not provide quantitative comparisons against these or traditional scaling-law baselines. This omission makes it difficult to assess the true improvement brought by LayerMix Law.
4. The analysis focuses exclusively on validation loss and perplexity. It would strengthen the paper to show whether the predicted improvements in pretraining loss translate into downstream task gains (e.g., MMLU, HellaSwag accuracy).

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work studies the scaling law taking data quality and repeats into consideration. Concretely, authors split the source data into different buckets with different quality scores and then model a function of consumed tokens, model size, sampling weights, and repetition levels by  treating the training as the accumulation of information from data. And the final scaling law shows a reliable prediction across different settings.

### Strengths
1) It is a very important problem to fit a scaling law in terms of data quality and repeats. We are running out of data very soon. For both small and large models. And we are also looking for better compute efficiency from using more times of high quality data, but also getting struggle with the overfitting effects and diminishing return from that.
2) The insight of treating the learning as a process of information accumulation is neat and intuitively sound.

### Weaknesses
1) I think one assumption of this work is the definition of quality is very reliable. But how can we trust that so much? 
2) Viewing the learning as a process of information accumulation is good. However, it cannot explain some aggressive overfitting -> If we repeat much more times, the model would get worse and worse when training for more steps. It is not an accumulation of info for sure.

Minor: Line 428: LayerMix Law also generate well .... "generate" -> "generalise"?
Missing reference: https://arxiv.org/abs/2305.13230 -> A very early work studying the repeat data problem also considered how the data quality matters.

### Questions
1) I don't understand why the name is "layer mix law". It is confusing because of the "layer" is widely used in model architecture. At the first glance, I thought this work is a model arch paper. How about "quality mix law"?
2) If we treat the learning process as a func of information accumulation, and we also take forgetting into consideration, are there any insights to order/shuffle the data for a smarter data schedule? For example, shall we repeat early more aggressively and then repeat a few more times again at the very late stage of the training to leverage the forgetting effect? No exps needed here. Just want to hear more insights.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The central focus of this paper is to investigate the effects of data quality and data repetition on the pre-training of large models. To this end, the authors introduce a novel metric, termed "Information Quantity," and establish its relationship with both data quality and data repetition. Subsequently, by modeling a power-law relationship between this "Information Quantity" and the val loss, the paper indirectly examines how data quality and duplication impact the model's training dynamics.

### Strengths
The power-law relationship established between "Information Quantity" and validation loss is robustly validated by the experiments presented in this work, demonstrating a commendable degree of originality. Furthermore, the overall experimental procedure of the paper is relatively comprehensive and complete.

### Weaknesses
To establish quantitative relationships, the paper hypothesizes several formulations (e.g., Equations 1, 6, 8, and 9). However, these assumptions lack sufficient justification, and their credibility is questionable.

The definitions of key concepts in the manuscript are ambiguous and appear arbitrary. Specifically, in Section 3.1 (Training Data Sampling, lines 168-174), within the formulation H(w, K, S, B), the meaning of the key variable 'S' is not introduced. Furthermore, the relationship between the associated variables 'w' and 'S' is not elucidated. The specific referent for 'B' also remains unclear. Additionally, the substitution of 'I' with f_{d} \cdot M_{d} in Equation 5, as well as the definition of f_{d} itself, requires substantial further justification or supporting evidence.

The justification for the preliminary experiments lacks rigor. Regarding the persistence of the Loss-C scaling law (line 206), demonstrating a good fit (fitting) is insufficient evidence on its own. To claim empirical validity, the authors should have used the fitted law to make further predictions; the accuracy of these predictions would then serve to validate the law. Concurrently, the "traditional law" (i.T., the OpenAI law) relates to (C-min), not simply 'C'. The authors must clarify this distinction and elaborate on the relationship between their findings and this established principle.

The Related Work section is missing several representative recent publications that are relevant to the scope of this study:

-Observational Scaling Laws and the Predictability of Langauge Model Performance

-Capability Salience Vector: Fine-grained Alignment of Loss and Capabilities for Downstream Task Scaling Law

-RegMix:Data Mixture as Regression for Language Model Pre-training

### Questions
1. please refer to 'Weakness'; 

2.How about providing some specific traindata proportion/ratio suggestions?

### Soundness
3

### Presentation
3

### Contribution
3
