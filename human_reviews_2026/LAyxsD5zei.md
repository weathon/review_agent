# Multi-Objective Task-Aware Predictor for Image-Text Alignment

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Evaluating image-text alignment while reflecting human preferences across multiple aspects is a significant issue for the development of reliable vision-language applications. It becomes especially crucial in real-world scenarios where multiple valid descriptions exist depending on contexts or user needs. However, research progress is hindered by the lack of comprehensive benchmarks and existing evaluation predictors lacking at least one of these key properties: (1) $\textit{Alignment with human judgments}$, (2) $\textit{Long-sequence processing}$, (3) $\textit{Inference efficiency}$, and (4) $\textit{Applicability to multi-objective scoring}$. To address these challenges, we propose a plug-and-play architecture to build a robust predictor, $\texttt{MULTI-TAP}$ ($\textbf{Multi}$-Objective $\textbf{T}$ask-$\textbf{A}$ware $\textbf{P}$redictor), capable of both multi and single-objective scoring. $\texttt{MULTI-TAP}$ can produce a single overall score, utilizing a reward head built on top of a large vision-language model (LVLMs). We show that $\texttt{MULTI-TAP}$ is robust in terms of application to different LVLM architectures, achieving significantly higher performance than existing metrics ($\textit{e.g.}$, +42.3 Kendall's $\tau_{c}$ compared to IXCREW-S on FlickrExp) and even on par with the GPT-4o-based predictor, G-VEval, with a smaller size (7$-$8B).
By training a lightweight ridge regression layer on the frozen hidden states of a pre-trained LVLM, $\texttt{MULTI-TAP}$ can produce fine-grained scores for multiple human-interpretable objectives. $\texttt{MULTI-TAP}$ performs better than VisionREWARD, a high-performing multi-objective reward model, in both performance and efficiency on multi-objective benchmarks and our newly released text-image-to-text dataset, $\texttt{EYE4ALL}$. Our new dataset, consisting of chosen/rejected human preferences ($\texttt{EYE4ALLPref}$) and human-annotated fine-grained scores across seven dimensions ($\texttt{EYE4ALLMulti}$), can serve as a foundation for developing more accessible AI systems by capturing the underlying preferences of users, including blind and low-vision (BLV) individuals. Our contributions can guide future research for developing human-aligned predictors.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MULTI-TAP — a scalable, plug-and-play framework for evaluating image-text alignment using large vision-language models (LVLMs). The paper also introduces EYE4ALL, a novel dataset targeting blind and low-vision contexts, containing both pairwise human preferences (EYE4ALLPref) and fine-grained multi-dimensional scores (EYE4ALLMulti).

### Strengths
* The proposed method of adding a single prediction head on the LVLM is model-agonistic.
* Results demonstrate consistent improvements in inference time reductions (14h→1h per 1k samples).
* The proposed EYE4ALL dataset fills a gap by emphasizing accessibility and BLV user preferences, representing an important step toward more inclusive evaluation.

### Weaknesses
* **Unclear motivation and role of the multi-objective setup.**
  The motivation for introducing the multi-objective learning scheme is not clearly articulated. In the current formulation, the multi-objective branch is completely decoupled from the general score prediction task. It is unclear how optimizing multiple objectives contributes to improving the general score prediction, since there is no explicit interaction between them. Moreover, it is not specified which dataset(s) were used to train the multi-objective components — are they based on *Polaris* and *ImageReward* as well? If so, where do the additional annotation dimensions come from? This lack of clarity makes it difficult to assess the necessity and contribution of the multi-objective design.

* **Unfair comparison with IXC-2.5 and unclear interpretation of human preference alignment.**
  The comparison with IXC-2.5 to argue about human preference misalignment is not meaningful. The human preference scores are computed from a **pointwise** dataset, while IXC-2.5 itself is trained on **pairwise** data. Such a comparison is inherently unfair and may not reflect true preference alignment. Moreover, among models that are designed for pointwise prediction, the proposed method’s improvement appears to be quite minor.


* **Lack of justification for the source of performance gains.**
  The paper does not sufficiently justify whether the reported performance improvements mainly stem from the proposed *single reward head training strategy* or from differences in training data. The baseline models used for comparison appear to be trained on different datasets (i.e. IXCREW-S), which undermines the fairness of the comparison. In particular, since the proposed model has been trained on *Polaris*, it naturally achieves higher Kendall’s correlation on the *Polaris* test set — suggesting that the observed gains may largely reflect data overlap rather than model improvements.

* **Misattribution of long-sequence capability as a method advantage.**
  The paper highlights long-sequence processing as one of the key advantages of the proposed approach. However, this capability appears to derive primarily from the underlying LVLM backbone and its context window length, rather than from the proposed method itself. Therefore, it should not be considered an intrinsic advantage of the method. The paper would benefit from a clearer distinction between the contributions of the proposed training strategy and the inherited properties of the base model.

### Questions
* The paper’s organization and figure captions are not sufficiently clear. For instance, in *Figure 1*, the label “Proposed Metric” seems misleading and should likely be “Proposed Method,” as the figure aims to compare proposed method with previous methods. 
* In *Figure 2*, the “Single-objective Reward Head” block is not colored, while “Multi-objective Regression” is highlighted. It is unclear whether these color differences convey any specific meaning or are merely stylistic. The figure would benefit from a clearer and more consistent visual convention to avoid reader confusion.

### Soundness
2

### Presentation
2

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
This paper presents MULTI-TAP, a multi-objective, task-aware predictor for evaluating image–text alignment. It builds a scalar reward head on top of large vision language models (LVLMs) for single-objective alignment and adds a ridge regression layer for multi-objective scoring. The authors also introduce EYE4ALL, a dataset featuring blind and low-vision user preferences. Experiments show MULTI-TAP achieves higher correlation with human judgments and greater efficiency than prior models such as CLIP-S, IXCREW-S, and VisionREW-S.

### Strengths
**Comprehensive formulation**: The paper clearly defines the limitations of existing image–text evaluation models and proposes a unified multi-objective predictor that can handle long-sequence inputs and multi-dimensional alignment.

**Architecture simplicity and scalability**: The two-stage design—frozen LVLM features with ridge regression—balances interpretability, modularity, and low computational cost.

**New dataset (EYE4ALL)**: The dataset broadens evaluation to accessibility-critical scenarios with BLV-oriented annotations, representing a valuable contribution to human-centered multimodal evaluation.

**Strong experimental results**: MULTI-TAP consistently outperforms existing metrics across a variety of datasets and model backbones, showing robustness and cross-architecture generalization.

### Weaknesses
**Limited theoretical novelty.** The proposed two-stage design—adding a scalar reward head in Stage 1 and fitting a linear ridge regression on frozen LVLM embeddings in Stage 2—is conceptually simple and builds upon existing practices in feature reuse and reward modeling. While effective, it lacks theoretical or algorithmic innovation beyond a well-engineered combination of known components.

**Insufficient ablations**: The multi-objective head searches α only by training loss and trains one epoch per α, which risks selection bias and underexploration of generalization.

The interaction between Stage 1 (single-objective) and Stage 2 (multi-objective) components is not explored; the paper does not show how each stage contributes to the final performance.

### Questions
1.Could you elaborate on why a linear ridge regression was chosen for multi-objective prediction? Have you tested other mappings (e.g., MLP or attention-based fusion) to verify whether the simplicity of ridge regression limits performance or interpretability?

2.MULTI-TAP underperforms IXCREW-S in the TI2T setting. Can you explain what causes this degradation.

3.Several datasets were reformulated or subsampled (e.g., OID*, Polaris*). Could you clarify how these modifications influence the comparability of results with prior works, and whether original test splits were also evaluated?

4.Other questions refer to weakness

I will consider increasing the score based on the authors’ response.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MULTI-TAP (Multi-Objective Task-Aware Predictor), a plug-and-play scalar model for evaluating image–text alignment. Built on large vision–language models, MULTI-TAP first learns a single-objective reward head aligned with human judgments using MSE loss, then adds a lightweight ridge regression layer to generate fine-grained multi-objective scores. The authors also present EYE4ALL, a new dataset capturing human and blind/low-vision preferences across seven evaluation dimensions. Experiments show that MULTI-TAP achieves stronger correlation with human ratings and higher efficiency than existing metrics (e.g., CLIP-S, IXCREW-S), performs competitively with GPT-4o-based G-VEval, and surpasses VisionREW-S on multi-objective evaluation tasks.

### Strengths
**Simple, pragmatic design:** The two-stage paradigm  strikes a balance between interpretability, efficiency, and portability across LVLM backbones.

**Multi-objective interpretability:** Dimension-specific continuous scores avoid fragile learned aggregation weights and better support task-aware evaluation and product use cases.

**Efficiency & long-context handling:**  Notable inference-time improvements vs. LLM-as-a-judge and per-dimension reward models; competitive under long-context inputs across I2T/T2I/TI2T.

### Weaknesses
**Limited ablations/attribution**:  The claim that **MSE** is superior to Bradley–Terry/pairwise losses lacks a **controlled ablation** under identical settings (same data/backbone/schedule) against BT/logistic pairwise/Huber/rank-based losses.
 
**Potential data overlap/leakage**: Stage 1 trains on Polaris/ImgREW and evaluates on related/variant benchmarks (e.g., Polaris*). More explicit leakage checks and sensitivity analyses are needed (e.g., overlap statistics).
 
**Aggregation & calibration**: The paper purposely avoids aggregating multi-objective scores, but many applications require **user/task-specific aggregation. Missing post-hoc calibration and explainable aggregators to ensure cross-dataset comparability.

### Questions
1. Please provide controlled comparisons of MSE vs. Bradley–Terry/rank-based  losses (acc, convergence, stability).

2. Why is ridge preferred over small MLPs or low-rank linear heads? Did you experiment with similar architectures, and if so, what differences did you observe in performance or stability?

3. How does MULTI-TAP compare to G-VEval and other generative judges on accuracy–cost trade-offs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MULTI-TAP, a novel plug-and-play architecture designed for efficient and robust evaluation of image-text alignment quality. It adopts a two-stage training strategy built on top of pretrained LVLMs, capable of producing not only an overall scalar score but also multi-dimensional fine-grained scores. MULTI-TAP effectively addresses the limitations of existing metrics in terms of human correlation, long-text handling, reasoning efficiency, and multi-objective scoring. In addition, the authors release a new dataset, EYE4ALL, aimed at evaluating text-image-text alignment for blind and low-vision users. Experimental results demonstrate that MULTI-TAP outperforms existing scalar reward models in both performance and efficiency.

### Strengths
(1) The paper's core claims are well supported by solid experiments. The rationality of the two-stage approach is validated across different LVLM backbones (Qwen2-VL, InternLM, LLaMA-3.2). The authors also conduct extensive comparisons with various state-of-the-art predictors, including the GPT-4o-based G-VEval.

(2) The authors additionally contribute the EYE4ALL dataset, filling the gap in TI2T evaluation benchmarks that reflect the preferences of assistive technology and BLV communities, which holds significant practical value.

### Weaknesses
(1) The LVLM responses in the EYE4ALL dataset are refined using GPT-4o mini, introducing a dependency on a proprietary model and potentially biasing the dataset toward GPT's output style, which may affect model generalization. Moreover, the dataset requires manual verification to ensure quality.

(2) The second stage employs Ridge Regression to learn multi-objective scores. This linear model may limit the ability to capture highly nonlinear interactions among objectives, and as the number of targets increases, the scores may become less accurate.

### Questions
The paper mentions that the single-objective score (Overall Score) in Stage 1 is highly correlated with the fine-grained score (Sufficiency) in Stage 2. Could this correlation be leveraged to guide the training process in Stage 2?

### Soundness
3

### Presentation
3

### Contribution
3
