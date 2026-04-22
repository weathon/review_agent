# S3OD: Towards Generalizable Salient Object Detection with Synthetic Data

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Salient object detection exemplifies data-bounded tasks where expensive pixel-precise annotations force separate model training for related subtasks like DIS and HR-SOD. We present a method that dramatically improves generalization through large-scale synthetic data generation and ambiguity-aware architecture. We introduce S3OD, a dataset of over 139,000 high-resolution images created through our multi-modal diffusion pipeline that extracts labels from diffusion and DINO-v3 features. The iterative generation framework prioritizes challenging categories based on model performance. We propose a streamlined multi-mask decoder that handles the inherent ambiguity in salient object detection by predicting multiple valid interpretations. Models trained only on synthetic data achieve 20-50% error reduction in cross-dataset generalization, while fine-tuned versions reach state-of-the-art performance across DIS and HR-SOD benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces S3OD, a large-scale synthetic dataset (139k images) and a corresponding ambiguity-aware salient object detection (SOD) model. The work proposes a multi-modal diffusion pipeline that fuses FLUX DiT features, concept attention maps, and DINO-v3 embeddings to generate high-quality image–mask pairs. An iterative generation framework prioritizes difficult categories based on model feedback, and a multi-mask decoder handles label ambiguity by predicting multiple plausible saliency maps. Experiments on DIS-5K, HRSOD, UHRSD, and DUT-OMRON show strong cross-dataset generalization, with models trained purely on synthetic data approaching or surpassing state-of-the-art methods.

### Strengths
1.Novel synthetic-data pipeline that unifies diffusion- and representation-based cues (DiT + DINO-v3 + concept maps), addressing weaknesses of prior attention-extraction methods.

2.Large-scale dataset contribution with detailed filtering and iterative feedback, potentially valuable to the SOD community.

3.Ambiguity-aware decoding design that explicitly models multiple valid masks, a practically important perspective for SOD.

### Weaknesses
1.While the dataset generation process is well described, the core algorithmic novelty of the ambiguity-aware model is incremental—largely a DPT backbone with multi-branch outputs and IoU-guided selection. This architectural contribution may not reach ICLR’s expected level of theoretical innovation.
2.The evaluation relies mainly on FID and Inception Score; no user study or domain-gap quantification is provided to show the practical relevance of the generated images for real-world deployment.
3.The comparison with other methods may not control for data volume and pretraining effects (e.g., DINO-v3 backbone vs. Swin-B), so performance gains might arise from model capacity rather than the proposed framework.
4.Critical hyperparameters (diffusion steps, fusion architecture, loss weights) are partially specified but not fully justified. The claim of “2 days on 8×H200” suggests heavy computation that may not be accessible to the community.

### Questions
1.The treatment of ambiguity remains heuristic. The paper models uncertainty through multiple mask branches and a decaying regularizer but lacks a probabilistic or information-theoretic formulation.
2.Loss functions (Focal + IoU) and weighting constants (e.g., λ_mask = 10, γ = 0.2) are empirically set without analytical justification or stability analysis.
3.The relaxed branch assignment mitigates degeneration empirically but offers no convergence proof or theoretical characterization of branch usage.
4.Domain-gap evaluation between synthetic and real data relies solely on FID and Inception Scores—useful but insufficient for capturing distributional or representational discrepancies.
5.The experimental section is extensive and well organized: the authors evaluate cross-dataset generalization, multi-modal ablations, multi-round data synthesis, and backbone comparisons. This breadth supports the main empirical claim that synthetic data can significantly improve SOD generalization. Nevertheless, several confounding factors remain unresolved. The performance gains might partially stem from larger data volumes and stronger pretrained encoders (DINO-v3) rather than from the proposed mechanisms themselves. Moreover, reproducibility is limited by high computational cost (three data-generation rounds, filtering, and multi-GPU training) and incomplete sensitivity analyses of key hyperparameters.

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
In this paper, the authors proposed S3OD method for salient object detection. In addition, the authors introduced S3OD dataset. In particular, they have created a massive synthetic dataset of over 139000 high resolution images with machine generated labels. The S3OD dataset helps improve the performance of the S3OD method.

### Strengths
+ The authors introduced S3OD dataset with 139,000+ samples.
+ Models trained only on the S3OD data show good performance.

### Weaknesses
- The authors use S3OD as the name for both method and dataset. This causes confusion in the paper reading.
- The novelty of the proposed S3OD method is incremental. All components of S3OD exist in literature. 
- The new S3OD dataset is AI-generated. However, all existing SOD datasets used ground truth from humans. The ground truth of S3OD dataset should come from humans. 
- The experiments are unfair. All baselines were not trained on the S3OD dataset. More than 139,000+ samples of S3OD should benefit baselines. 
- The authors should carefully proofread the paper. They have not described Figure 1, Figure 4 and Table 1 in the text.

### Questions
Please see the above comments.

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
S3OD targets the long-standing generalization limits in salient object detection by unifying DIS and HR-SOD through synthetic data and an ambiguity-aware model. The authors build a 139k+ high-resolution dataset via a multi-modal diffusion pipeline that extracts masks from diffusion and DINO-v3 features, then iteratively steers generation toward hard cases based on model feedback.  They propose a streamlined multi-mask decoder that explicitly models multiple valid saliency interpretations, addressing label ambiguity while keeping the architecture simple.  Trained only on synthetic data, their method improves cross-dataset generalization with 20–50% error reductions; with brief fine-tuning on real data, it reaches or surpasses state of the art across DIS and HR-SOD benchmarks.

### Strengths
- Multi-Modal Dataset Diffusion Pipeline that fuses diffusion feature maps, concept attention maps, and DINO-v3 representations to jointly generate images and masks, ensuring strong image–label alignment and enabling a 139k+ high-resolution synthetic set that boosts generalization.

- Ambiguity-aware architecture with a streamlined multi-mask decoder that explicitly models multiple valid interpretations.

- Iterative generation framework that is feedback-driven to prioritize challenging categories, delivering consistent gains and cross-dataset generalization, culminating in SOTA across DIS and HR-SOD and large error-rate reductions versus prior methods.

### Weaknesses
- The author should clarify whether mask extraction in the Multi-Modal Dataset Diffusion pipeline requires any training or calibration. If not, provide rigorous evidence of mask fidelity.

- The proposed data-generation paradigm appears tailored to binary/saliency segmentation; please evaluate transfer to camouflaged object detection (COD) or non-salient classes and report the zero-shot and fine-tuned results.

- The author should strengthen the annotation rationale with interpretable visualizations. Provide side-by-side maps of DINO-v3 features, DiT feature activations, and the resulting masks across easy and challenging scenes, plus quantitative correlations, to demonstrate complementary cues.


- On page 4, line 197, the last two terms in the loss function formula should be grouped together.

- It is recommended not to place Tables 7 and 8 at the very bottom of the page.

- Citations should be enclosed in parentheses.

- On page 2, line 059, “remains” → “remain”.

- On page 2, line 067, “features” → “feature”.

- On page.6, line.270: “order of magnitude” → “an order of magnitude”/“orders of magnitude” contain” → “contains”.

- On page.8, line.401: “siginificantly” → “significantly”.

- On page.8, line.407: “S3OD achieve” → “S3OD achieves”.

### Questions
- Could a teacher model be used to annotate the FLUX-generated images, then train S3OD on these teacher-labeled data, and compare its performance with the S3OD proposed in this paper?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces S3OD, a new synthetic data generation pipeline and ambiguity-aware model for salient object detection (SOD). The approach leverages multi-modal diffusion models to generate over 139,000 high-quality, high-resolution images and corresponding masks that capture diverse, ambiguous segmentation scenarios. S3OD combines iterative, feedback-driven data sampling and a streamlined multi-mask architecture built on a DINO-v3 backbone, aiming to improve generalization across both dichotomous segmentation (DIS) and high-resolution SOD (HR-SOD) tasks. Empirical evaluation demonstrates strong cross-dataset transfer and new state-of-the-art results on multiple SOD/DIS benchmarks. The dataset and code will be released.

### Strengths
**1) Scale and Diversity of Synthetic Data (Figure 1, Table 1, Figure 4, Figure 6):** S3OD delivers an order-of-magnitude increase in dataset scale for SOD, with 139k+ images spanning 1676 unique objects and a wide spectrum of scene types, lighting, and occlusions, as seen qualitatively in Figures 1, 4, and 6 and quantitatively in Table 1. Manually verified mask quality and data curation strategies, including filtering with VLMs, result in a synthetic dataset that rivals or exceeds real sets in annotation fidelity.

**2) Innovative Data Generation Pipeline (Figure 3):** The multi-modal pipeline fuses diffusion latent features, concept attention maps, and DINO-v3 features to create dense, pixel-precise masks without teacher bottlenecks or reliance on pre-existing mask libraries. The iterative feedback loop dynamically adjusts category sampling based on model performance (see Figure 3 and Section 4.2), which is a notable step forward in dataset synthesis methodology.

**3) Ambiguity-Aware Model Architecture (Figure 2):** The proposed multi-mask head on top of a DINO-v3/DPT backbone enables the network to represent inherent annotation ambiguity—a factor often overlooked in SOD. Figure 2 provides a clear and helpful visualization of the model’s architecture, showing multiple candidate mask outputs and predicted IoU for selection at inference.

### Weaknesses
**Potential for Domain Overfitting or Synthetic-“Leakage” Not Fully Addressed:**

1. Although the cross-dataset generalization is well documented, concerns about overfitting to synthetic artifacts (such as those possibly present in highly artificial or LLM-generated prompts) are only partly mitigated by filtering and photo-realism tuning (see Figure 7 and Section B). There is no explicit domain gap or bias quantification (such as t-SNE/UMAP distributions, or model calibration metrics) to back up claims about synthetic-to-real transfer.

2. Table 9 claims strong Inception/FID scores, but these are image-level metrics—less relevant than mask consistency or class-level error for segmentation evaluation.

**Limited Theoretical Justification for Multi-Modal Fusion and Mask Decoding (Section 3.1, Equations):**

1. While practical motivation is given, the paper glosses over the theoretical rationale for fusing FLUX DiT, concept attention, and DINO-v3 features in the supervision pipeline. The process is described at a systems level (“projected to a common 256-dimensional space via separate convolutional branches...”) but lacks rigorous analysis of why this fusion is optimal, how information is disentangled, or how fusion improves over single-modality supervision. Further, the formulation of loss weighting and decay (e.g., $\lambda_{\text{mask}}=10$, $\gamma=0.2$) is heuristically chosen; no ablations or theoretical discussion support these choices.

2. The mask selection (best out of N) and loss propagation scheme is borrowed from multiple-choice learning but could benefit from a more detailed analysis or ablation to determine how many branches (N) are optimal and how performance scales with it.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
