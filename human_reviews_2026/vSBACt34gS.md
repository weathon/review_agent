# The data-quality illusion: Rethinking Classifier-based Quality Filtering for LLM Pretraining

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large-scale models are pretrained on massive web-crawled datasets containing documents of mixed quality, making data filtering essential.
A popular method is Classifier-based Quality Filtering (CQF), which trains a binary classifier to distinguish between pretraining data and a small, high-quality set. It assigns each pretraining document a quality score defined as the classifier's score and retains only the top-scoring ones.
We provide an in-depth analysis of CQF. 
We show that while CQF improves downstream task performance, it does not necessarily enhance language modeling on the high-quality dataset.
We explain this paradox by the fact that CQF implicitly filters the high-quality dataset as well. 
We further compare the behavior of models trained with CQF to those trained on synthetic data of increasing quality, obtained via random token permutations, and find starkly different trends. 
Our results challenge the view that CQF captures a meaningful notion of data quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper critically examines classifier-based quality filtering, a widely used method for pretraining data selection. It shows that CQF improves downstream task performance but paradoxically doesn’t improve language modeling on the high-quality data. The authors argue that CQF implicitly filters the HQ set itself and captures stylistic similarity rather than true data quality. They introduce "data conditioning" as a new lens to evaluate whether filtering improves optimization dynamics.

### Strengths
- Excellent empirical and theoretical dissection of a widely used but poorly understood technique.
- Timely and relevant analysis of a core assumption in LLM data engineering.
- Clear exposition of the paradox and insightful explanation linking CQF to implicit HQ filtering.
- Strong experimental design and visualizations that clarify complex effects.

### Weaknesses
- This work is largely diagnostic with limited actionable guidance on improving CQF beyond critique.
- Data conditioning concept, while elegant, remains somewhat abstract and untested in real large-scale settings.
- All experiments are conducted at modest scale which may not generalize.
- Relies heavily on pretraining proxies (ARC, MMLU) rather than practical LLM evaluation.

### Questions
- Can the proposed data conditioning principle be used to design better filtering methods?
- How robust are the findings when using multilingual datasets?
- Does the illusion persist when HQ sets are human-curated instruction data versus web data?
- Could the implicit HQ filtering effect be leveraged deliberately (e.g., via adaptive weighting)?

Missing citations:
- When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale, Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, Sara Hooker
- Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs, Kyle O'Brien, Stephen Casper, Quentin Anthony, Tomek Korbak, Robert Kirk, Xander Davies, Ishan Mishra, Geoffrey Irving, Yarin Gal, Stella Biderman

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper argues that the perceived benefits of Classifier-based Quality Filtering (CQF) in large language model (LLM) pretraining are largely illusory.

Although CQF improves downstream benchmark performance, it does not actually select data that better resemble “high-quality” corpora, nor does it improve language modeling performance on such data.

Instead, CQF works primarily by removing obvious low-quality samples and by reweighting the pretraining distribution toward benchmark-style text.

### Strengths
1. Elegant theoretical framing (density-ratio view) that unifies prior empirical quirks.

2. Strong empirical design across multiple HQ datasets.

3. Clear visualizations demonstrating domain drift.

### Weaknesses
1. The filtering process also reduces token count; loss differences may stem from fewer tokens, not “quality”.

2. Circular validation: The “HQ-decile” analysis (Fig. 5) reuses the same CQF score to both partition and evaluate HQ samples, effectively validating the classifier with itself rather than demonstrating genuine quality correlation.


2. The scaling-law fitting is not convincing.

    - The experiments vary both data size and distribution with k, violating the assumption of a fixed task distribution.

    - Only three data points (1 %, 10 %, 100 %) are used, with uncontrolled compute budgets, making the fitted β unreliable.

    - No residuals or confidence intervals are reported.

I recommend removing or reframing this section; at present, it does not provide meaningful evidence of scaling behavior.

### Questions
see weaknesses.

### Soundness
3

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
This paper revisits Classifier-based Quality Filtering (CQF) — a standard data curation method in LLM pretraining. The authors find a paradox: CQF boosts downstream task performance but doesn’t improve language modeling on the high-quality (HQ) dataset. They explain this by showing CQF implicitly reweights the HQ set itself, emphasizing samples far from the low-quality (LQ) data. They also contrast CQF with importance sampling and introduce data conditioning as a new, optimization-based notion of data quality.

### Strengths
- The writing is clear and easy to follow.


- The paper identifies a previously overlooked paradox in CQF, showing that it may not improve language modeling on high-quality data even when it enhances downstream task performance — a perspective that challenges long-standing assumptions in data filtering.


- Introducing data conditioning as an optimization-based definition of data quality represents a significant conceptual advance, offering a new lens to evaluate dataset usefulness beyond static classifier scores.

### Weaknesses
- While the data conditioning notion is conceptually sound but too ideal, it is impractical for real-world data filtering — evaluating it requires repeated model training and loss computation across datasets, which is computationally expensive and cannot be efficiently estimated per sample. Thus, it serves more as a diagnostic concept rather than a usable filtering metric. From this perspective, the original CQF-style “remove-the-bad” metric can be viewed as a more conservative yet pragmatic strategy for large-scale data curation. Moreover, this paper fail to provide an alternative practical metric to 


- The experimental analysis, while extensive, is conducted on relatively limited model scales (≤1.3B) and specific datasets (RedPajama-V2, OpenOrca, KnowledgePile). It is unclear whether the same paradoxical behavior of CQF would persist under larger-scale models (such as 7B) or more diverse corpora with heterogeneous noise patterns.

### Questions
1. The data conditioning notion is theoretically appealing but computationally impractical. How could it be approximated or operationalized for large-scale data filtering — e.g., via small-model proxies, gradient statistics, or early-training dynamics?

2. The paper analyzes how the CQF selection fraction (k) affects downstream and HQ-set performance, showing that smaller k values emphasize samples farther from the LQ distribution. However, the analysis does not explicitly quantify how the relative size of the selected subset influences this implicit reweighting. When k becomes large—approaching the size of the HQ set itself—does CQF recover most HQ-like data, or does it still inevitably include LQ-like regions? Could the authors clarify how the composition (HQ-aligned vs LQ-aligned) of the filtered dataset evolves with k and whether this proportion can be estimated empirically?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to explore the mechanism behind classifier-based quality filtering (CQF) that is commonly used for large-scale pretraining data. The authors question whether CQF selects high-quality (HQ set) data. The paper finds that although CQF improved the downstream metrics, it does not necessarily lead to better language modeling on the HQ set. Finally, the authors propose a framework to "evaluate" CQF using "data conditioning," where they find CQF captures properties more closely related to stylistic or domain similarities.

### Strengths
1. CQF is a commonly used method to curate a pre-training dataset for many state-of-the-art LLMs. Works aims to understand CQF is highly relevant and important. 

2. The paper includes a large set of experiments ranging over multiple datasets and settings.

### Weaknesses
1. The paper challenges that CQF correlates with the data property of "universal quality"; however, it is unclear what "universal quality" means. There are no actionable findings pointed out from the paper, rather than showing that CQF works to better align with the downstream tasks. Is this not expected when HQ data is used from downstream targets?

I would be interested in creating a universally high-quality dataset -- without using directly downstream task datasets, and maybe using human/LLM annotations -- and exploring CQF using this data. 

2. Evaluation tasks do not necessarily correlate with HQ data used in terms of the paper's analysis. Specifically, OpenOrca and OpenHermes datasets are instruction datasets, and OpenWebMath is a math domain data; however, there is no instruction-style or math evaluation. Supporting this, the downstream metric vs loss relationship is only different for ARC Eacy, where the evaluation and HQ set closely match. 

3. Training details (token budget, model size, hparams) are not present in the main pages of the paper, which is important because pretraining dynamics are highly dependent on these factors. I would like to see a baseline where the model is trained with the base dataset and see if it performs above random in downstream evaluations.

### Questions
Please see `Weaknesses` for my questions.

### Soundness
2

### Presentation
2

### Contribution
2
