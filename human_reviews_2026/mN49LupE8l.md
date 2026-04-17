# GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 2

## Abstract
Recent attempts to transfer features from 2D Vision–Language Models (VLMs) to 3D semantic segmentation expose a persistent trade-off. Directly projecting 2D features into 3D yields noisy and fragmented predictions, whereas enforcing geometric coherence necessitates costly training pipelines and large-scale, annotated 3D data. We argue that this limitation stems from the dominant \textit{segmentation-and-matching} paradigm, which fails to reconcile 2D semantics with 3D geometric structure. The geometric cues are not eliminated during the 2D-to-3D transfer but remain latent within the noisy and view-aggregated features. To exploit this property, we propose \textbf{GeoPurify} that applies a small Student Affinity Network to purify 2D VLM-generated 3D point features using geometric priors distilled from a 3D self-supervised teacher model. During inference, we devise a Geometry-Guided Pooling module to further denoise the point cloud and ensure the semantic and structural consistency. Benefiting from latent geometric information and the learned affinity network, GeoPurify effectively mitigates the trade-off and achieves superior data efficiency. Extensive experiments on major 3D benchmarks demonstrate that GeoPurify achieves or surpasses state-of-the-art performance while utilizing only \textbf{$\sim$1.5\%} of the training data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an open-vocabulary 3D segmentation framework, GeoPurify, which purifies noisy semantic features using 3D geometric guidance. The approach introduces a lightweight Student Affinity Network that learns pure geometric relationships via a Geometric Contrastive Distillation scheme, without requiring any 3D semantic labels. The method demonstrates notable data efficiency, as using only ~1.5% of the training scenes achieves comparable performance to state-of-the-art methods trained on the full dataset.

### Strengths
1. The method achieves SOTA performance using ~1.5% of unlabeled 3D scan data, significantly reducing reliance on costly 3D annotations, offering a practical and promising solution for 3D understanding in data-scarce scenarios.
2. This paper learns a pure geometric prior to regularize and enhance semantic features, which addresses the key issue of geometric fragmentation resulting from the 2D-to-3D projection.

### Weaknesses
1. The choice of 20 scenes lacks empirical justification. An ablation study examining performance trends across different subset sizes (e.g., 10, 30, or 50 scenes) is missing, making it difficult to assess the optimality of the selected size or the method's sensitivity to training data quantity.  
2. Would there be more analysis on the effectiveness of data efficiency? Table 1 presents the severe degradation of CUA-O3D under a ~1.5% subset. Why does the proposed method not show such a degradation? 
3. Lack of discussions with related methods. Recent state-of-the-art methods, such as Open3DIS [1] and UniSeg3D [2], have reported results on ScanNet200, which are not discussed in this paper. Are the experimental settings different?

[1] Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance. CVPR 24.

[2] A Unified Framework for 3D Scene Understanding. NeurIPS 24.

### Questions
Please refer to details in the above section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents GeoPurify, a data-efficient geometric distillation framework for open-vocabulary 3D segmentation. The work identifies a fundamental disconnect between semantic richness and geometric coherence in existing methods and proposes a shift from the "segmentation and matching" paradigm to "segmentation as understanding." The core idea is to leverage a student affinity network to distill latent geometric structure from noisy 2D VLM-generated 3D features under the guidance of a self-supervised 3D teacher model. A geometry-guided pooling mechanism is further introduced to refine features during inference. The method is evaluated on multiple benchmarks and shows competitive or state-of-the-art performance using only ~1.5% of the training data, demonstrating strong data efficiency and generalization ability.

### Strengths
1. The paper clearly articulates the limitations of existing approaches in reconciling 2D semantics with 3D geometry and proposes a principled shift in perspective toward "segmentation as understanding." The idea of recovering latent geometric information from noisy 2D features, rather than learning 3D geometry from scratch, is both novel and well-motivated.

2. The paper provides extensive evaluations across multiple datasets. The comparison with a strong baseline under the same low-data regime is particularly compelling.

### Weaknesses
1. Some figures contain low-resolution text and should be improved. The authors should consider replacing these with higher-resolution images or vector graphics to improve readability.

2. Heavy reliance on pre-trained models; the contribution breakdown is unclear. The performance gains of GeoPurify are built upon powerful pre-trained models. While the framework itself is novel, it remains unclear how much of the improvement comes from the proposed distillation and pooling mechanisms versus the strength of the pre-trained backbones.

### Questions
This paper presents a well-designed framework that effectively integrates contrastive distillation and geometric pooling, but it fails to clearly demonstrate how its core idea of learning geometric affinity structures provides advantages beyond existing methods.

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
4

### Summary
The paper introduces GeoPurify, a method for open-vocabulary 3D semantic segmentation that enhances the 3D consistency of features extracted from 2D vision-language foundation models. The approach first extracts 2D features using X-Decoder and merges them in 3D via a student affinity network trained to learn geometric relationships by distilling knowledge from the 3D foundation model Sonata (Wu et al.). Training is performed on only a small subset of ScanNetV2 (1.6%) and Matterport3D (1.3%), demonstrating data efficiency. Evaluation on ScanNetV2, Matterport3D, and ScanNet200 shows that GeoPurify outperforms training-based baselines trained on similar subsets, and performs competitively-though slightly worse-than full-data and zero-shot baselines.

### Strengths
1. **Effective use of Sonata for geometric distillation.** The paper presents an interesting idea of enhancing the 3D consistency of 2D foundation model features by distilling geometric relationships from a strong 3D foundation model (Sonata). This cross-modal distillation strategy is conceptually sound and well motivated.
2. **Clear and well-structured presentation.** The paper is generally clearly written and easy to follow.

### Weaknesses
1. **Limited performance gains.** Although the method is conceptually interesting, it does not clearly outperform strong zero-shot baselines. This raises uncertainty about the practical benefit of the proposed geometric distillation mechanism.
2. **Restricted evaluation scope.** The method is only evaluated in a low-data regime. Assessing performance when trained on larger portions of the dataset would provide a clearer picture of its scalability and potential advantages over existing approaches

### Questions
1. Would it be possible to elaborate on why in Table 1 GeoPurify does not improve much over the zero shot baselines?
2. I believe the method can be interesting even if trained on more data. Therefore, it would be interesting to evaluate its performances when trained on the whole datasets and see whether in that way it can improve over the other baselines.
3. Is the point cloud required as input at inference time? From the text it seems to be the case (line 155), but from Figure 2 it seems to be needed only at training time.

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
This paper presents GeoPurify, a geometric distillation framework for open-vocabulary 3D semantic segmentation. By introducing geometry-guided contrastive distillation and affinity-based semantic purification, the method aims to reduce projection noise from 2D VLM features and improve 3D semantic consistency. The approach demonstrates strong results on datasets and requires only 1.5 % of the trainning data, showing satisfactory annotation efficiency.

### Strengths
1. The idea of leveraging geometric priors to purify VLM features is meaningful and addresses a real weakness of projection-based pipelines.

2. The method achieves competitive performance with significantly reduced 3D training data, which is practically valuable for indoor scene understanding.

### Weaknesses
1. The claimed paradigm shift from *“Segmentation-and-Matching”* to *“Segmentation-as-Understanding”* is not entirely novel, as recent works such as OV3D[1] and PGOV3D[2] already adopt VLM-based understanding pipelines. The conceptual contribution could be more precisely positioned, and comparisons with these recent methods are somewhat limited.

[1] Jiang, Li, Shaoshuai Shi, and Bernt Schiele. "Open-vocabulary 3d semantic segmentation with foundation models" in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[2] Zhang, Shiqi, et al. "PGOV3D: Open-Vocabulary 3D Semantic Segmentation with Partial-to-Global Curriculum" in Proceedings of ACM Multimedia, 2025.

2. The current subset selection depends on global dataset statistics, which contradicts the “fully annotation-free” motivation. It remains unclear whether the reported gains are due to the method itself or a bias toward selecting higher-quality samples.

3. The method is evaluated only on indoor datasets (e.g., ScanNet). Since the approach relies heavily on geometric consistency, its generalization to outdoor datasets such as nuScenes remains unclear.

### Questions
1. Could the authors provide a quantitative comparison with recent VLM-based understanding pipelines?

2. How robust is GeoPurify if the subset is selected randomly instead of using global statistics?

3. How does the method perform on outdoor datasets such as nuScenes or SemanticKITTI?

### Soundness
3

### Presentation
3

### Contribution
1
