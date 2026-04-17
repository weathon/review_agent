# Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Vision-Language models (VLMs), such as CLIP, achieve powerful zero-shot classification. However, their predictions remain highly sensitive to spurious correlations, where common background or contextual cues dominate predictions over semantic content. Earlier solutions typically rely on fine-tuning, but this undermines the advantages of pre-trained models. Others depend on prompt engineering, which is prone to hallucination issues. In addition, most approaches are limited to a single modality, increasing the risk of misalignment between text and images. In this work, we propose Density-Aware Translation (DAT) that refines image-text similarity scores using a local geometric density term derived from group reference sets. Our approach is motivated by the phenomenon that CLIP embeddings exhibit a modality gap and lie on an anisotropic shell in the feature space: common patterns cluster near the mean, while rare patterns are pushed outward. This geometry creates uneven alignment, where spurious correlations are amplified while semantically meaningful but rare cues are marginalised. To address this, we employ a relative measure that rescales similarities based on embedding density, suppressing overconfident scores in diffuse regions while preserving dense, semantically consistent matches. Experimental results on benchmark datasets demonstrate consistent improvements in worst-group and average accuracy, highlighting density-aware translation as a simple and effective calibration mechanism for reliable zero-shot classification using multimodal models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of spurious correlations in zero-shot Vision-Language Models (VLMs) like CLIP, where models over-rely on frequent but semantically irrelevant cues rather than meaningful content. The authors propose Density-Aware Translation (DAT), a method that refines image-text similarity scores by incorporating a local geometric density term derived from group reference sets. DAT rescales similarities to suppress overconfident scores in sparse regions while preserving dense, semantically consistent matches. The approach operates in a zero-shot regime without fine-tuning or prompt engineering. Theoretical analysis shows that DAT corrects biases in cosine similarity under anisotropic embeddings, aligning scores with Bayes-optimal decisions. Experiments on benchmarks demonstrate consistent improvements in worst-group and average accuracy across multiple VLMs.

### Strengths
1)	DAT consistently improves worst-group and average accuracy across multiple benchmarks and model architectures.
2)	Formal proofs show DAT reinstates anisotropy-sensitive terms ignored by cosine similarity, aligning with Bayes-optimal decisions.
3)	DAT operates in a zero-shot setting, requires no model parameters, and is computationally efficient compared to baselines like TIE.

### Weaknesses
1)	Limited discussion on the sensitivity of DAT to the quality and diversity of reference sets, especially in datasets with high class or attribute complexity (e.g., FMoW).
2)	The theoretical analysis relies on the Kent distribution and log-SLOF fidelity assumption, which may not fully capture real-world embedding geometries.
3)	No comparison to methods that explicitly model uncertainty or use generative approaches for debiasing, which could provide additional context for DAT’s advantages.

### Questions
1)	Could the authors provide more intuition or visualization for how SLOF captures density in high-dimensional embedding spaces, especially for non-experts?
2)	How does DAT scale to datasets with a large number of spurious attributes or fine-grained classes, and are there computational trade-offs as group diversity increases?
3)	Beyond accuracy, has DAT been evaluated on other fairness metrics (e.g., Equality of Opportunity, Demographic Parity)? This is important for assessing its utility in sensitive applications.
4)	The experiments use fixed prompt templates. How would the relative advantage of DAT change if more powerful or domain-specific prompt engineering (e.g., using LLMs) were employed?

### Soundness
3

### Presentation
3

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
The objective of this paper is to address spurious correlations in zero-shot VLMs. This work builds on existing research showing that spurious-correlated embeddings exhibit an anisotropic ellipsoidal structure, with such samples located near the mean of the geometry. To mitigate spurious correlations, the authors propose density-aware translation (DAT), which adjusts the scores of samples to break the spurious relationship. To further improve performance, they introduce an aggregation method over predictions. The visualization in Figure 4 demonstrates the effectiveness of DAT. The study is evaluated on widely accepted benchmark datasets and outperforms existing methods in improving robustness to spurious correlations.

### Strengths
I find this paper quite interesting for the following reasons:

1. It leverages previous research showing that spurious-correlated embeddings are located near the mean of an ellipsoidal geometry. The proposed DAT method addresses spurious correlations from a geometric perspective, which I find particularly interesting and novel.

2. It provides a theoretical result showing that DAT yields a Bayes-aligned decision boundary, offering a theoretical justification for its effectiveness.

3. The paper is well written. The background information is clearly presented, and the overall flow of the paper is smooth and easy to follow.

4. The experimental results are solid. The authors evaluate their method across different scenarios within multi-class classification settings. I agree that FMOW is a very challenging dataset, and an improvement of nearly 7% on WG is convincing evidence of the strength of their approach.

### Weaknesses
1. The aggregation step lacks justification. In Equation (4), both components could serve as prediction logits. I don’t quite understand the difference between using the average group-specific score (second term) and the max score (first term). Could you clarify why the chosen combination is better?

2. A straightforward baseline would be to use SLOF to detect outliers, i.e., samples that break the spurious correlation. Since CLIP embeddings are often dominated by spurious features (e.g., background), if we set a threshold to flag outliers and then flip the zero-shot prediction, would that work? And compared with the translation results, how close does this simple baseline get?

3. What is the numerical range of  $D_{y,a}(a)$, and how does it vary across groups? Would dividing by this value cause disproportionately large shifts for certain groups? Could you provide a bar plot per dataset to illustrate the distribution across groups?

### Questions
1. Could DAT be used to diagnose spurious concepts or discover new spurious features, similar to the first-step procedure in [1]?

2. In Section 3.2, the authors only evaluate one prompt. Can the evaluation generalize to different prompts and different datasets?

[1] Wu, Shirley, et al. "Discover and cure: Concept-aware mitigation of spurious correlation." International Conference on Machine Learning. PMLR, 2023.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the problem of spurious correlations in vision-language models (VLMs). It proposes Density-Aware Translation (DAT), a method that rescales CLIP similarity scores based on data density, leveraging group reference sets to capture the local geometry of the data.

### Strengths
The idea of addressing the spurious correlation problem in VLMs from a density-based perspective is novel.

### Weaknesses
The debiasing approach proposed in this paper (DAT) uses training or validation data as a reference set to capture the geometry of the data. This design contradicts the “zero-shot” claim made by the authors, as it requires access to target images. And the hyperparameters also seem to require target data for tuning. In contrast, competing methods such as ROBOSHOT, TIE, and Perception CLIP do not rely on any training or validation images, making them genuinely zero-shot debiasing approaches. Therefore, the comparison between DAT and these baselines is not fair. 

Moreover, the paper introduces a large number of notations, but they are not organized clearly. Many complex equations are left unlabeled, making them difficult for readers to reference. Some equations also lack clarity—for example, it is unclear what the variable w represents in the definition of TMD.

### Questions
In Equation (2), embeddings from sparse regions have larger SLOF values, resulting in lower similarity scores. This mechanism appears to encourage similarities among dense regions while penalizing those from sparse regions. According to the abstract, this design aims to “preserve dense, semantically consistent matches.” However, as illustrated in Figure 1(a), frequent but spuriously correlated samples tend to cluster near the mean, whereas rare yet semantically meaningful samples lie in sparser regions. This seems paradoxical. Could the authors please clarify this point?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a training-free method, Density-Aware Translation (DAT), for mitigating spurious correlations in zero-shot VLMs. The authors propose to rescale image–text cosine similarities using a local density term estimated from small, balanced group reference sets, motivated by the observation that CLIP embeddings are anisotropic and lie on an ellipsoidal shell where frequent cues cluster near the mean while rarer, semantically meaningful cues lie in sparser regions. The authors model group distributions with the Fisher–Bingham distribution and show DAT restores anisotropy-sensitive log-likelihood terms, aligning decisions with Bayes-optimal scoring. Experiments on standard spurious-correlation benchmarks show gains in worst-group and average accuracy across multiple VLMs in a zero-shot setting.

### Strengths
The proposed method, DAT, enhances robustness without retraining or gradient updates, making it efficient and broadly applicable to pre-trained vision-language models.

The authors provides a principled correction for embedding anisotropy via density-aware scoring derived from the Kent distribution, linking geometric intuition with Bayesian optimality.

### Weaknesses
The proposed method depends on small, balanced reference sets per sensitive group, which undercuts the core appeal of zero-shot classification. Can the proposed method be generalized to the scenarios with no reference set or partially labelled reference sets? 

On several backbones/datasets, the method does not outperform TIE or other competitive baselines, despite requiring more label information and extra computation. Consequently, it is unclear the specific advantages of the proposed method over existing baselines.

The theoretical correction relies on Kent/anisotropy assumptions and local-density estimates that may be brittle or noisy. 

Maintaining balanced references and computing densities could scale poorly as groups or classes grow.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
