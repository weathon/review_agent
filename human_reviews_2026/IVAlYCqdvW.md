# Sapiens2

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
We present Sapiens2, a model family of high-resolution transformers for human-centric vision focused on generalization, versatility, and high-fidelity outputs. Our model sizes range from 0.4 to 5 billion parameters, with native 1K resolution and hierarchical variants that support 4K. Sapiens2 substantially improves over its predecessor in both pretraining and post-training. First, to learn features that capture low-level details (for dense prediction) and high-level semantics (for zero-shot or few-label settings), we combine masked image reconstruction with self-distilled contrastive objectives. Our evaluations show that this unified pretraining objective is better suited for a wider range of downstream tasks. Second, along the data axis, we pretrain on a curated dataset of 1 billion high-quality human images and improve the quality and quantity of task annotations. Third, architecturally, we incorporate advances from frontier models that enable longer training schedules with improved stability. Our 4K models adopt windowed attention to reason over longer spatial context and are pretrained with 2K output resolution. Sapiens2 sets a new state-of-the-art and improves over the first generation on pose (+4 mAP), body-part segmentation (+24.3 mIoU), normal estimation (45.6% lower angular error) and extends to new tasks such as pointmap and albedo estimation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents SAPIENS2, a significant advancement over the original Sapiens foundation model for human-centric computer vision. The work focuses on increasing generalization, versatility, and output fidelity across a broad spectrum of human-related tasks.The core contributions lie in three areas:

1. Introduction of models ranging from 0.4 billion to a 5 billion parameter variant, trained at native 1K resolution, and new hierarchical 4K models capable of 2K output resolution for dense prediction. The Sapiens2-5B model represents one of the highest-FLOPs vision transformers reported.

2. A novel pretraining objective combining Masked Image Reconstruction

3. Scaling the curated pretraining data to 750 million high-quality human images (HUMANS-750M) and increasing post-training task annotations by 10x for tasks including pose, body-part segmentation, depth, normals, and the newly added pointmap and albedo estimation.

### Strengths
1. The paper demonstrates compelling gains. The increase in performance over the first generation (e.g., +22.3 mIoU on body-part segmentation and +4 mAP on pose estimation) is substantial and justifies the scale and methodological changes. Crucially, the dense probing results (Table 2) confirm that the pretrained features themselves are vastly superior for human tasks, showcasing strong zero-shot generalization. The Sapiens2-5B model setting a new SOTA with 82.3 mAP on dense 308-keypoint in-the-wild predictions is a major result.

2. The core technical argument—that pure MIM lacks semantic understanding, while CL erodes fine-grained photometric detail—is sound, particularly for high-fidelity human-centric dense tasks (like avatar creation or albedo estimation).

3. The decision to avoid injecting handcrafted human-specific priors (beyond data selection) during pretraining is admirable. This "truly inductive prior-free approach" demonstrates that raw scale and a general-purpose objective can, in this domain, outperform many hand-engineered human-centric models

### Weaknesses
No significant weaknesses.

### Questions
How does the performance (especially dense vs. semantic tasks) change as $\lambda$ is varied?

Given the "truly inductive prior-free approach" and the massive scale of the Sapiens2-5B model, what is its performance on general computer vision benchmarks that do not involve humans (e.g., COCO object detection, ImageNet linear probing, or ADE20K segmentation)? This would help contextualize Sapiens2's feature learning against general foundation models like DINOv2 and M-ViT at comparable scales.


The paper mentions several architectural stability features (GQA, QK-Norm, RMSNorm, SwiGLU). Which of these, or combination thereof, were most critical for the stable training of the largest 5B parameter, 4K input hierarchical models? Was the training of the largest variant notably more sensitive to hyperparameter choices compared to the smaller models?

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
3

### Summary
The paper presents SAPIENS2, a family of high-resolution ViT backbones (0.4B–5B params) for human-centric vision. 
The key technical move is a unified pretraining objective that adds a student–teacher contrastive loss on the [CLS] token to MAE reconstruction, aiming to keep low-level fidelity for dense prediction while learning semantic invariances (see Fig. 4 and Sec. 3; the 4K windowed→global hierarchy is sketched in Fig. 5). 

The models are trained on a curated ~750M human-image corpus; post-training heads cover pose (308 kp), 29-class part segmentation, pointmap (XYZ), normals, and albedo. Results on the authors’ curated test sets show improvements over the previous SAPIENS-v1 and strong baselines under a frozen-backbone dense-probe protocol (e.g., Tables 2–6; visuals on pages 8–9).

The approach integrates ideas known in the literature—MAE for reconstruction and self-distilled/contrastive training à la DINOv2/iBOT—and is consistent with hierarchical ViTs for high-res inputs (Hiera) and stability tweaks like RMSNorm, QK-Norm, GQA. The paper’s novelty is primarily engineering/integration at scale for human-centric dense tasks rather than introducing a new learning paradigm.

Summary of Review:
The paper introduces a strong unified pretraining framework combining low-level fidelity (MAE) with semantic alignment ([CLS] distillation) in a 4K-ready hierarchical transformer, showing solid technical design and clear scaling trends. However, the lack of public benchmark validation, detailed ablations, and data transparency limits external interpretability. Overall, it’s a well-executed and promising approach but still short of full maturity, justifying a rating of 6 for solid technical depth with room for broader validation.

### Strengths
1) Unified pretraining that explicitly marries low-level fidelity (MAE) with semantic alignment ([CLS] distillation); design aligns with robust SSL practice (DINOv2/iBOT/JEPA). 
2) 4K-ready hierarchical transformer with practical stability/throughput tweaks (RMSNorm, QK-Norm, GQA) consistent with the literature.
3) Fair frozen-probe protocol across backbones; strong internal scaling trends (0.4B→5B) across pose/seg/geometry (Tables 2–6; pages 8–9).

### Weaknesses
1) Public benchmarks missing. Add results on community standards (e.g., COCO-WholeBody 133-kp, plus any public human-part segmentation/normal datasets) to anchor gains. 

2) Component ablations. Quantify deltas from RMSNorm vs LayerNorm, QK-Norm, GQA, SwiGLU, and the masking-after-local choice in the 4K pipeline

3) FLOPs claim. Standardize FLOPs reporting (input size, attention type, precision) and compare to published large ViTs (e.g., DINOv2-G, etc) under a common convention.

4) Data governance transparency. Given human images at scale, could be nice to add a data/model card (sources, licenses, demographic coverage, minors handling, opt-out) such that to show overall a general idea regarding the data used.

### Questions
1) External benchmarks: Can you report COCO-WholeBody results (with 308→133 mapping) and at least one public normals/segmentation benchmark? 
2) More ablations: What are the per-component contributions of RMSNorm / QK-Norm / GQA / SwiGLU and the masking point in 4K?
3) Data disclosures: Confirm the 750M pretraining size, list high-level source categories, and describe license vetting, deduping, and opt-out mechanisms.
4) FLOPs accounting: Please give per-image FLOPs at 1K and 4K, the attention configuration, and compare to DINOv2-G and ViT-22B.
5) Augmentations: How sensitive are albedo/normal metrics to color jitter/solarize on global vs local crops?

Minor:
In Figure 4, I think it should be “L_mae helps the model to learn…” and not “L_mae helps the model learns …”.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces Sapiens2, a family of high-resolution vision Transformer models ranging from 0.4B to 5B parameters designed for human-centric vision tasks. It aims to achieve stronger generalization, task versatility, and high-fidelity output. Building upon its predecessor Sapiens, Sapiens2 undergoes comprehensive upgrades across three dimensions: pre-training, data, and architecture.

Unified Pre-training Objective: It combines Masked Image Modeling (MAE) with Self-Distilled Contrastive Learning (CL). This approach preserves pixel-level details beneficial for dense prediction tasks while learning high-level semantics beneficial for zero-shot/few-shot generalization. The joint objective avoids the limitations of pure MAE which lacks semantics and pure CL which neglects details.

Large-Scale, High-Quality Human-Centric Data: The Humans-750M dataset is constructed containing 750 million high-quality human images filtered through multiple stages. It covers diverse populations, poses, scenes, and lighting conditions. The only requirement is that each image contains at least one prominent person, with no other manual priors.

High-Resolution Scalable Architecture: The model supports native 1K resolution and pioneers a 4K resolution hierarchical model achieved through window attention plus global attention. It integrates cutting-edge techniques including RMSNorm, Grouped-Query Attention (GQA), QK-Norm, SwiGLU FFN, and a PixelShuffle decoder to enhance training stability and efficiency. The pre-training output resolution reaches 2K, significantly improving detail representation for dense predictions.

Extensive Downstream Task Coverage and SOTA Performance: After pre-training, the model achieves state-of-the-art results across five dense prediction tasks including pose estimation, body part segmentation, depth estimation, surface normal estimation, and albedo estimation. It substantially outperforms its predecessor and existing methods, demonstrating significant improvements such as pose estimation mAP improved by +4, body segmentation mIoU improved by +22.3, and normal estimation angular error reduced by 29.2%.

### Strengths
1. This paper adopts a simple joint pre-training paradigm of "masked reconstruction + contrastive learning." Although this technique has been widely used in various fields, its application in the human-centric vision domain effectively balances low-level detail and high-level semantic learning, addressing the trade-off between dense prediction and semantic understanding in existing self-supervised methods. Additionally, it scales the Vision Transformer to 4K resolution while supporting 2K output.

2. The experiments are exceptionally solid. Not only does it establish new SOTA results across multiple tasks, but it also demonstrates the model's strong generalization capabilities in zero-shot, few-shot, and fully supervised scenarios through dense probing, ablation studies, and extensive visualizations (e.g., Figures 1, 7–10). The descriptions of data, architecture, and training strategies are detailed and reproducible.

3. A large-scale human-centric dataset was collected and used for training, validating the effectiveness of large-scale data and large models in the human vision domain.

4. The provision of a series of models of different sizes can significantly advance industrial applications.

### Weaknesses
1. Limited generalization in crowded scenes: As mentioned on page 9 of the paper, the model performs excellently on images containing 1-4 prominent people but shows performance degradation in crowded scenes or complex multi-agent interactions. This limits its application in real-world scenarios such as surveillance and sports events. Future work needs to explore more robust multi-person modeling mechanisms.

2. Synthetic data discussion: For geometric and material tasks like point maps, normals, and albedo, training relies entirely on synthetic data. Although results show good generalization (Figure 10), the synthetic-real domain gap remains a potential risk.

3. Data processing pipeline details: The data processing pipeline is important in this paper. While the paper mentions the data processing pipeline, details of each step - such as specific annotation verification and sampling strategies - could be provided. Alternatively, forming a methodology for the community would promote development.

### Questions
1. After collecting such a large volume of human-centric data, what instructive conclusions can be drawn? Building upon a decent pre-trained model, can we continue to scale up the data volume, and what type of additional data would be more effective?

2. When real data is exhausted, considering the use of synthetic data—such as leveraging AIGC models to generate human-centric data—would such generated data be beneficial for this type of discriminative model?

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
3

### Summary
The paper introduces Sapiens2, a new family of Vision Transformer models ranging in size from 0.4 billion to 5 billion parameters. This model family constitutes the second generation of the Sapiens line (ECCV 2024), maintaining the original goal of creating versatile foundation models centered on inputs/outputs depicting humans. The core focus is on achieving strong generalization, versatility, and high-fidelity outputs.

The primary differences from the original Sapiens models lie in the larger scale of the training data, the pretraining paradigm (hybrid instead of MAE), and the supported resolution (4K). Specifically, Sapiens2 models are pretrained on a dataset of 750 million human-centric images, isolated and curated from an initial web-scale pool of approximately 4 billion images. The pretraining employs a hybrid self-supervised paradigm that combines Masked Auto-Encoding (MAE) with semantic contrastive learning, a technique previously explored by methods like CMAE (Huang et al., TPAMI 2022) for image classification. 

The models are evaluated across five human-centric tasks: pose estimation, body-part segmentation, depth estimation, surface normals estimation, and albedo estimation. Fine-tuning for each task involves adding and training a lightweight task-specific head while keeping the large backbone frozen.

### Strengths
1. State-of-the-Art Performance: The models demonstrate state-of-the-art results across all five evaluated tasks, showcasing the efficacy of the scaling and pretraining approach.
2.  The evaluation is comprehensive, covering five complex, pixel-level tasks with strong quantitative results and compelling qualitative examples, effectively demonstrating the models' high-fidelity capabilities.
3. The paper is well written and well structured 
4. Contribution of high resolution (up to 4K) specialized foundation models focused on complex human analysis (conditional on release).

### Weaknesses
1. Limited Fundamental Novelty: the primary advances appear to be an increase in scale (data and backbone size) and the application of an existing hybrid self-supervised paradigm (MAE + contrastive learning, previously seen in classification) to a new domain (human-centric generation). The paper does not introduce novel fundamental insights or methodological breakthroughs in transformer architecture or pretraining theory.
2. It is currently unclear whether the dataset, code, or the trained model checkpoints will be publicly released. Given that the main contribution of the paper resides in the achieved performance and the massive scale of the trained model family, the lack of public availability would severely limit the reproducibility, impact, and overall contribution to the research community.

### Questions
1. Are the authors planning to publicly release the curated 750M image dataset, the source code for the pretraining and fine-tuning pipelines, and/or the trained Sapiens2 model weights?

2. Ablation on Pretraining: Could the authors provide an ablation comparing the hybrid self-supervised objective (MAE + Contrastive) against the individual components (MAE-only and Contrastive-only) to justify the added complexity and quantify the specific performance gain attributed to the combined loss?

### Soundness
3

### Presentation
3

### Contribution
3
