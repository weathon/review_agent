# From Misclassification to Outliers: Joint Reliability Assessment in Classification

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Building reliable classifiers is a fundamental challenge for deploying machine learning in real-world applications. A reliable system should not only detect out-of-distribution (OOD) inputs but also anticipate in-distribution (ID) errors by assigning low confidence to potentially misclassified samples. Yet, most prior work treats OOD detection and failure prediction as separated problems, overlooking their closed connection. We argue that reliability requires evaluating them jointly. To this end, we propose a unified evaluation framework that integrates OOD detection and failure prediction, quantified by our new metrics DS-F1 and DS-AURC, where DS denotes double scoring functions. Experiments on the OpenOOD benchmark show that double scoring functions yield classifiers that are substantially more reliable than traditional single scoring approaches. Our analysis further reveals that OOD-based approaches provide notable gains under simple or far-OOD shifts, but only marginal benefits under more challenging near-OOD conditions. Beyond evaluation, we extend the reliable classifier SURE and introduce SURE+, a new approach that significantly improves reliability across diverse scenarios. Together, our framework, metrics, and method establish a new benchmark for trustworthy classification and offer practical guidance for deploying robust models in real-world settings. Code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces two metrics to evaluate the model in the joint scenario of OOD detection and failure prediction. The metrics -- DS-F1 and DS-AURC are derived directly from F1 and AURC scores, by defining different cases in the joint evaluation setting. The paper further improves the SURE methods with several techniques, like RegPixMix and F-SAM, to improve both the ID acc and OOD detection performance.

### Strengths
1. New evaluation metrics: the paper studied the joint setting of OOD detection and failure prediction, and proposes two straightforward metrics -- DS-F1 and DS-AURC to evaluate the model.
2. New training methods: the paper improves SURE by integrating several techniques and achieves better performance on both OOD detection and ID acc.

### Weaknesses
1. The significance of double-scoring needs further justification: while separate scores and be adopted to evaluate the model's performance on failure prediction and OOD detection, the significance of calculating a joint metric remains unclear.
2. Lacking soundness of the proposed SURE+: the authors propose to adopt several off-the-shelf techniques to improve the baseline SURE. Though achieving higher performance, this method is not well related to the paper's main contribution and claims, more like an engineering combination, not a new method.
3. The experiments are not sufficient to validate the effectiveness of the metrics: the authors only use the MSP score as the ID score, which is also a kind of OOD score, making it a special case. By definition, the ID scores can be any scores that measure the failure likelihood. Therefore, adopting only the MSP as the ID score doesn't thoroughly examine the proposed new metrics.
4. The motivation for adding each technique to SURE is not presented.

### Questions
1. At around line 348, why do the two metrics never worsen the evaluation by producing scores that are at least as high as F1 score and as low as AURC. I don't see a direct correlation between the score scale and evaluation quality.

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
4

### Summary
This paper argues that real-world deployment of machine learning requires classifiers that can not only detect OOD inputs but also misclassifications within the in-distribution (ID) data. Since prior work often treats these two problems separately, the paper proposes a unified evaluation framework.

### Strengths
The paper clearly demonstrate the necessity of jointly evaluating OOD detection and misclassification prediction for real-world reliability, which is a crucial practical concern.

### Weaknesses
1. The paper's primary claim of proposing a unified evaluation for OOD detection and failure prediction is not entirely novel. Several prior works [1-5] have already addressed this problem. The paper's distinction rests mainly on the double scoring mechanism rather than the concept of joint evaluation, which significantly weakens the framework's overall contribution.

2. The proposed metrics are viewed as a trivial extension of existing single-scoring metrics (F1 and AURC) to a two-dimensional threshold space $(\tau_{OOD}, \tau_{ID})$. They do not introduce new theoretical insights into risk modeling. Furthermore, similar to their single-scoring counterparts, these metrics are still heavily influenced by the absolute number of mispredicted ID samples and the number of OOD samples. This reliance can obscure the true effectiveness of the underlying detection mechanism when the class distributions are highly imbalanced, which is a known limitation in failure detection benchmarking.

3. The key experimental observation that OOD-based methods provide only marginal benefits under challenging near-OOD conditions is a widely recognized limitation [1-5]. Simply re-confirming this known challenge does not constitute a significant contribution.

4. The proposed SURE+ method appears to be an engineering modication of several established regularization techniques integrated into the existing SURE framework.

References

[1] A call to reflect on evaluation practices for failure detection in image classification

[2] Failure detection in medical image classification: A reality check and benchmarking testbed

[3] Learning to reject meets ood detection: Are all abstentions created equal

[4] A unified benchmark for the unknown detection capability of deep neural networks

[5] Plugin estimators for selective classification with out-of-distribution detection

### Questions
Please refer to Weaknesses

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
3

### Summary
This paper presents SURE+, an improved training recipe and evaluation framework aiming to build more reliable classifiers by jointly considering OOD detection and failure prediction. The authors further introduce two new evaluation metrics, DS-F1 and DS-AURC, to assess reliability in a unified manner. The method modifies the SURE baseline (Li et al., 2024b) by replacing several components—such as CRL loss, CSC head, SWA, and data augmentation—with simpler or alternative choices (e.g., EMA, linear classifier, RegPixMix, and F-SAM). Experiments on the OpenOOD benchmark demonstrate consistent improvements over prior methods.

### Strengths
The paper addresses an important and timely topic, reliable classification that integrates OOD detection and failure prediction.
The joint evaluation framework is conceptually reasonable and could potentially help bridge two often-separated research directions.
The paper provides comprehensive experimental results on standard benchmarks, showing the consistency of improvements.
The writing is generally clear, and the experimental setup is reproducible.

### Weaknesses
Limited novelty of the proposed method (SURE+).
The modifications over SURE are mainly component replacements using existing methods (e.g., EMA, F-SAM, RegPixMix), without introducing fundamentally new ideas. The resulting method reads more like a collection of known techniques rather than a coherent new approach.

Lack of clear methodological focus.
The framework mixes metric design, pipeline tweaks, and augmentation choices, making it hard to identify the core contribution. The work feels somewhat “mixed and unfocused.”

Unclear motivation and limited effectiveness of new metrics (DS-F1 and DS-AURC).
The motivation behind these metrics is not fully convincing—why a double scoring setup is inherently better than existing reliability measures (e.g., AURC, AUROC, ECE). From Table 1, the observed gains appear marginal.

Insufficient theoretical or conceptual justification.
The paper would benefit from a deeper analysis or theoretical discussion showing why the proposed double scoring better reflects model reliability or uncertainty.

Benchmarking vs. contribution gap.
While the authors claim to establish a new benchmark, the contribution seems incremental and largely empirical, with little conceptual advancement.

### Questions
1. Can the authors clarify the conceptual novelty of SURE+ beyond being an ensemble of existing training tricks?
2. How sensitive are the results to the specific component choices (e.g., EMA vs. SWA, RegPixMix vs. RegMixup)?
3. For DS-F1 and DS-AURC, what is the precise intuition or mathematical rationale that supports their superiority over existing reliability metrics?

### Soundness
2

### Presentation
3

### Contribution
3
