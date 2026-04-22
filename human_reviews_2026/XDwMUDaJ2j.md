# Agreement with the Ensemble for Zero-Shot Vision-Language Model Selection

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Pretrained Vision-Language Models (VLMs) such as CLIP are well known for enabling zero-shot classification with category names. The rapid growth of open-access variants has led to a diverse VLM zoo, where selecting the most suitable model can yield superior zero-shot performance, yet the optimal choice is often dataset-dependent. At the same time, selecting VLMs for zero-shot tasks is challenging, since only category names are available and target images are absent. Prior approaches rely on text-only evaluation, which suffers from the modality gap inherent to VLMs. To address this issue, we propose SAGE (Selection via AGreement-with-the-Ensemble), which leverages in-the-wild images to bridge the modality gap. Specifically, SAGE quantifies the agreement between individual VLMs and their ensemble counterparts in terms of prediction behavior on in-the-wild images. Experiments demonstrate that SAGE consistently outperforms state-of-the-art zero-shot VLM selection methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SAGE (Selection via Agreement-with-the-Ensemble), a method for zero-shot vision–language model (VLM) selection. Instead of relying on text-only proxies (e.g., LLM-generated captions), SAGE computes an Agreement-with-Ensemble (AGE) metric using in-the-wild images and measures how consistent each model’s predictions are with those of a model ensemble. Experiments on 35 VLMs across 23 datasets show improved correlation with true accuracy and outperform existing baselines (ModelGPT, SWAB).

### Strengths
1. Addresses a practically relevant and underexplored problem: zero-shot VLM selection without target images or labels.
2. The AGE metric is simple, intuitive, and easy to compute without LLMs.
3. Extensive experiments with consistent improvements over baselines.

### Weaknesses
1. The effectiveness of the SAGE method largely relies on a key assumption: that the ensemble of vision–language models can provide high quality pseudo labels for in-the-wild images. However, the reliability of such ensemble generated pseudo labels is not always guaranteed.
2. Using ImageNet images (even unlabeled) introduces domain leakage: most target datasets are natural images, making results less convincing.
3. The idea of using agreement or disagreement as a proxy for generalization has strong precedents—e.g., “Assessing Generalization of SGD via Disagreement” (Jiang et al., ICLR 2022) and AETTA (Li et al., 2024), yet these are not cited or compared.
4. Only empirical correlations are shown, no formal analysis of why agreement implies accuracy.

### Questions
1. How does SAGE perform when the in-the-wild images come from a very different domain (e.g., satellite, medical)?
2. Would the performance drop significantly if ImageNet were replaced with a more diverse or less curated dataset such as LAION or COCO?
3. Have you evaluated whether including the tested model in the ensemble biases the AGE score?

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
3

### Summary
The paper tackles the problem of zero-shot VLM selection, where the goal is to choose the most suitable pretrained VLM for a downstream classification task without access to target images or labels, using only the category names.Existing approaches such as ModelGPT and SWAB rely on text-only proxy evaluations using LLM-generated captions, but these methods suffer from the modality gap between text and image domains.
To address this, the authors propose SAGE (Selection via AGreement-with-the-Ensemble), which introduces the AGE metric which is a measure of how much an individual VLM’s prediction agrees with the predictions of an ensemble of VLMs. The key insight is that ensemble consensus provides high-quality pseudo-labels that correlate strongly with actual performance.
SAGE extends AGE to the zero-shot setting by:
1) Using in-the-wild images (e.g., ImageNet samples) to approximate unseen target data
2) Employing semantic retrieval to select or weight images semantically relevant to the task categories

Incorporating robust similarity measures (class ranking correlation, KL divergence, total variation) to improve stability
The paper conducts experiments across 23 benchmark datasets and a zoo of 35 VLMs which show that SAGE outperforms baselines like ModelGPT and SWAB.

### Strengths
1) Introduces a novel ensemble-agreement metric (AGE) for model evaluation without labels. It moves away from reliance on LLMs, reducing computational cost and potential biases.
2) The paper has Conducts thorough ablations showing the importance of each component: semantic retrieval, composed similarity, and regularized regression.
3) The method demonstrates robustness to random image sampling and insensitivity to model pool composition.

### Weaknesses
1) In Eq 7, the semantic retrieval weighting depends on a reference VLM f_0. The choice of f_0 may inference results, but this sensitivity is not analyzed. 
2) Regarding the usage of “in-the-wild” ImageNet images, their domain proximity to training data may bias the results. How does the SAGE performance vary on natural images and non-natural image domains (e.g. satellite, medical, sketch, etc.)? 
3) The ensemble-based metric may implicitly leak information about overall VLM capacity. The paper could benefit from analysis on whether SAGE generalizes to unseen architectures or novel domains.

### Questions
1) The pseudo-label assumption assumes that ensemble consensus correlates with accuracy. Could this break down if all models share the same bias? Since AGE relies on ensemble agreement, could a biased ensemble (e.g., models trained on similar data) lead to misleading pseudo-labels?
2) For the combination with text-based scores (SAGE + ModelGPT), how much of the improvement comes from complementarity vs. redundancy?

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
This paper proposes SAGE, a method to select top-performing vision-language models (VLMs) for zero-shot downstream inference from a VLM model zoo. The proposed method leverages in-the-wild images with pseudo-labels generated by VLM ensembles to estimate the performance for each individual VLM. SAGE further conducts semantic retrieval to reduce the semantic gap between the in-the-wild image dataset and the target task. Compared with state-of-the-art approaches, SAGE achieves higher Recall 5 and Weighted Kendall's coefficient on 23 downstream datasets.

### Strengths
- The problem being investigated, i.e., how to select proper models from a model zoo, is becoming more critical at this time. Research towards this direction should be encouraged.
- The paper is well-written and easy to follow, and the idea is simple yet effective on the evaluated 23 downstream datasets.

### Weaknesses
- The effectiveness of the proposed method relies heavily on the constructed in-the-wild image dataset. The weight estimation in eq. (7) could become unreliable if the semantic gap between the in-the-wild dataset (e.g., ImageNet) and the target task (e.g., medical images) is huge. To improve robustness, the authors should consider developing a principled method to construct the proxy dataset. For instance, by curating web images relevant to the target categories, rather than using a random subset of ImageNet.
- The result in table 4 is insufficient to demonstrate that the VLM ensemble used in SAGE is insensitive to the construction of the model zoo. The relationship between model selection performance and the size of the ensemble should be discussed.

### Questions
- Will the selected model still performs consistently better than other models in the zoo after few-shot finetuning?

### Soundness
3

### Presentation
3

### Contribution
2
