# Anchor-Based Conformal Prediction Under Noisy Annotations in Single-Cell Data

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Conformal prediction provides a flexible framework for quantifying
prediction uncertainty and has attracted extensive interest. However, most
existing methods are designed to handle clean data and may fail to perform
satisfactorily when  labels are noisy. In this work, we
consider the setting where the ground-truth  labels are unobserved but crowdsourced noisy
labels are available. We introduce an anchor-based conformal prediction
method that provides  uncertainty quantification.
 Our method identifies anchor points by selecting samples
with strong agreement across annotators. These anchors points are used to train a base predictor
that is calibrated to construct a conformal prediction set with a desired coverage rate.
Meanwhile, we provide a theoretical analysis of anchor--point identification and
provide associated conditions that have been importantly overlooked in the literature.
We apply the proposed method to analyze two single-cell datasets to demonstrate its utility and promise.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper suggests a conformal prediction scheme for the case of a calibration set with multiple annotators.
The method is applied to single-set data.

### Strengths
The paper's topic, applying conformal prediction in the presence of label noise, is an important problem that often appears in real-world situations.

### Weaknesses
There are many recent studies on conformal prediction with label noise. See e.g. Conformal prediction of classifiers with many classes based on noisy labels, COPA 2025 and the references (from previous years) in that paper. 
I find the assumption that we can find anchor points unrealistic in many cases.
The proposed method is not tailored to single-cell data, and it can be validated on other types of datasets.
 The assumptions in Theorem 1  are very strong. What happens if the assumptions are not fulfilled?

### Questions
see weeknesses

### Soundness
2

### Presentation
2

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
This paper proposes an anchor-based conformal prediction framework for learning predictive models under noisy annotations from multiple annotators, with a focus on single-cell transcriptomics data. The method identifies (pseudo-)anchors—samples with high annotator agreement—to train a base predictor that models annotator-specific noise transitions using deep neural networks. It then calibrates top-$k$ prediction sets to provide distribution-free uncertainty guarantees, ensuring marginal coverage while producing compact sets. Contributions include theoretical guarantees on anchor identification and coverage, as well as empirical validation on two scRNA-seq datasets , demonstrating robustness to label noise.

### Strengths
The paper is clear to follow and well-written. The application to single-cell data is of practical importance.

### Weaknesses
Dealing with label noise via anchor points and the CP method seems disconnected. The optimization of the model in the first stage does not seem to directly influence CP results.
In addition, calibration based on top-k is not standard and leads to fixed-size sets that does not reflect uncertainty.

### Questions
1. Seems that the related work section does not contain a comprehensive discussion on previous CP methods that deal with label noise. Can you add elaborate on that?
2. Can you explain why APS appears to be overly conservative?
3. Can you compare to other scores, such as softmax-based scores, RAPS and SAPS?
4. To strengthen the generality of the proposed method, consider including results on additional commonly used noisy labeled datasets from diverse domains.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an anchor-based conformal prediction framework for classification with noisy annotations, motivated by multi-annotator single-cell data. The key idea is to identify anchor samples—instances where multiple annotators strongly agree—then train an annotator-dependent transition model to learn label noise patterns, and finally apply top-k conformal calibration on these anchors to generate uncertainty-calibrated prediction sets. The paper provides theoretical results on anchor identifiability (Theorems 1–2) and standard conformal validity (Theorem 3), and demonstrates empirical results on two single-cell RNA-seq datasets.

### Strengths
1. Using multi-annotator agreement to define “anchors” and calibrate predictions is conceptually appealing and relevant for biological or crowdsourced settings.
2. The method achieves coverage close to nominal levels with smaller prediction sets compared to APS baselines.
3. The paper is clear and readable; theoretical statements are mathematically consistent.
4. The anchor-based identifiability concept could inspire broader label-noise research.

### Weaknesses
1. The conformal component is entirely standard (split conformal with top-k quantile calibration). There is no novel conformity score, no modified calibration rule, and no new theoretical insight about conformal validity under noisy labels. The main contribution is anchor identifiability, which belongs more to noisy-label learning or multi-annotator modeling rather than to conformal prediction.
2. The paper assumes that coverage guarantees derived on anchor subsets transfer to general noisy test samples, but this is never justified.
3. Exchangeability may not hold once anchors are selected via annotator agreement, so the claimed “distribution-free validity under noisy labels” is overstated.
4. The anchor identifiability theorems are essentially restatements of existing results in label-noise learning (e.g., Xia et al., NeurIPS 2019) under multi-annotator independence assumptions. The conformal theorem is a trivial re-use of standard results from Shafer & Vovk (2008).
5. Experiments are limited to two single-cell RNA-seq datasets; no controlled noise experiments, no cross-domain evaluation (e.g., image or NLP), and no comparison with other noise-aware or conformal baselines. Thus, the empirical evidence is insufficient to support claims of general “noise robustness”.

### Questions
1. The conformal component seems standard (split-conformal with top-k quantile calibration). What is the genuine novelty on the conformal prediction side, beyond applying CP to anchors identified from noisy labels?
2. You claim “distribution-free coverage under noisy annotations”. Does this guarantee hold only for the anchor (or pseudo-anchor) subset, or for all test samples? Please clarify the assumed exchangeability after anchor selection.
3. Theorems 1–2 resemble known identifiability results (e.g., Xia et al., NeurIPS 2019). Can you explain what is new in your formulation, and whether the “better-than-random” and independence assumptions can be relaxed or empirically verified?
4. Experiments are limited to two single-cell datasets. Have you tested robustness under controlled synthetic noise or on non-biological datasets to show generality?
5. How does the number or quality of anchors influence coverage and set size?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work uses anchor points to develop a conformal prediction framework that produces valid prediction sets in the presence of label noise and multiple noisy annotators. The framework identifies class-specific anchors and models annotator behavior and class dependence using feedforward neural networks. Conformal prediction is used to generate prediction sets with marginal coverage. The paper provides guarantees on existence and identification of anchor points and validates empirical performance on two single-cell RNA-seq datasets.

### Strengths
The problem and considered application is interesting and of significance. The experimental setup is interesting, practically relevant, and detailed. The authors perform experiments on single-cell RNA-seq datasets and identify anchors across cell types. The comparison with APS also demonstrates the benefits of the approach.

### Weaknesses
1. The paper is not well-written in its current form and needs improvement. The paper mentions concepts without formally introducing them e.g, ‘top-k’ in the abstract and after that as well without explaining the notation; the definition of ‘anchor’ appears late in the paper. Additionally, the introduction is written as related work without appropriately motivating the paper, making the paper not as accessible generally.
2. The paper has hallucinated citations e.g., ‘Anastasios N Angelopoulos et al. Conformal prediction for multi-label classification, ICML 2022’ – I don’t believe there exists any such paper. Additionally, the paper has cited incorrect papers on some instances e.g., p1 l49 ‘split conformal prediction’ – the correct citations for these include Lei et al. (2015); Papadopoulos et al. (2002) as can be seen from Lei et al. (2018). The references are also formatted inconsistently (Anastasios N Angelopoulos et al., 2022; Yaniv Romano et al., 2020 – some references are in this format which is unusual, while others mention all authors). If LLMs were used for generating citations and references, no such usage has been disclosed in the paper and I would like to flag this.
3. Missing/insufficient baselines: The paper compares only with APS which is not expected to perform well in this setup. Not only do there exist score functions which produce smaller sets e.g., RAPS but comparison with methods that are more geared toward similar applications is required to establish benefits of the method.
4. The paper doesn’t study pseudo-anchors or class imbalance and its implications in detail e.g., Plasma cell has 0 anchors (Table 1).

### Questions
Missing discussion and comparison with some relevant work:

David Stutz, Abhijit Guha Roy, Tatiana Matejovicova, Patricia Strachan, Ali Taylan Cemgil, Arnaud Doucet. Conformal prediction under ambiguous ground truth. TMLR.

Michele Caprio, David Stutz, Shuo Li, Arnaud Doucet, Conformalized Credal Regions for Classification with Ambiguous Ground Truth, 2024.

### Soundness
2

### Presentation
2

### Contribution
2
