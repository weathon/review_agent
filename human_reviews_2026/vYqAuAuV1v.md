# Temporal Saliency-Guided Distillation: A Scalable Framework for Distilling Video Datasets

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Dataset distillation (DD) has emerged as a powerful paradigm for dataset compression, enabling the synthesis of compact surrogate datasets that approximate the training utility of large-scale ones. While significant progress has been achieved in distilling image datasets, extending DD to the video domain remains challenging due to the high dimensionality and temporal complexity inherent in video data. Existing video distillation (VD) methods often suffer from excessive computational costs and struggle to preserve temporal dynamics, as naïve extensions of image-based approaches typically lead to degraded performance. In this paper, we propose a novel uni-level video dataset distillation framework that directly optimizes synthetic videos with respect to a pre-trained model. To address temporal redundancy and enhance motion preservation, we introduce a temporal saliency-guided filtering mechanism that leverages inter-frame differences to guide the distillation process, encouraging the retention of informative temporal cues while suppressing frame-level redundancy. Extensive experiments on standard video benchmarks demonstrate that our method achieves state-of-the-art performance, bridging the gap between real and distilled video data and offering a scalable solution for video dataset compression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Video-As-Prompt (VAP), a unified framework for semantic-controlled video generation. Instead of relying on pixel-aligned structures or task-specific fine-tuning, VAP treats a reference video as a semantic prompt, guiding generation through in-context control. The method employs a Mixture-of-Transformers (MoT) architecture—combining a frozen Video Diffusion Transformer with a trainable expert network connected via full attention—and a temporally biased position embedding to avoid spurious spatial mappings. The authors also build VAP-Data, a large dataset with 100K paired samples across 100 semantic conditions. Experiments demonstrate that VAP achieves performance comparable to commercial systems like Kling and Vidu, while maintaining strong zero-shot generalization and scalability.

### Strengths
1.	The paper introduces a conceptually novel Video-As-Prompt (VAP) paradigm that reformulates semantic-controlled video generation as an in-context generation problem. This reframing is elegant and unifies previously fragmented approaches (e.g., per-condition fine-tuning, task-specific modules) under a single framework.
2.	The paper provides a clear motivation for addressing the limitations of structure-controlled and condition-specific models. The theoretical formulation and empirical ablations (e.g., on RoPE bias, expert architecture, scalability) are thorough and convincing.
3.	A notable highlight is the demonstrated zero-shot generalization to unseen semantic conditions, suggesting that the model captures abstract semantic correspondences beyond specific training distributions.
4.	The authors construct VAP-Data, the largest dataset for semantic-controlled video generation, covering 100K paired videos across 100 semantic conditions. This dataset provides a strong benchmark and a solid foundation for future research in this area.

### Weaknesses
1.	The Mixture-of-Transformers (MoT) architecture, while powerful, nearly doubles the total parameter count (adding approximately 5 billion parameters) and substantially increases computational cost.
2.	While comparisons with VACE, LoRA finetuning, and commercial models are included, the paper lacks detailed benchmarking against concurrent unified frameworks (e.g., Omni-Effects (Mao et al., 2025)) that also explore multi-condition control. Such comparison would strengthen claims of superiority in unification and generalization.
3.	The zero-shot results (e.g., Fig. 7) are visually compelling but lack quantitative validation. Incorporating objective metrics or user preference scores for unseen conditions would make the zero-shot generalization claim more convincing.

### Questions
See Weakness.

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
4

### Summary
This paper presents a novel and efficient video dataset distillation method that leverages temporal saliency to preserve motion dynamics while compressing video data. The proposed framework achieves superior performance across multiple benchmarks with lower computational cost compared to existing methods.

### Strengths
1. The TSGF mechanism is an innovative and lightweight way to capture temporal importance without relying on heavy optical flow or 3D convolutions.
2. The model design is computationally efficient, adopting uni-level optimization design instead of complex bi-level optimization, which substantially reduces memory footprint and training cost, making it suitable for large-scale video data distillation.
3. The method achieves state-of-the-art results on multiple standard benchmarks (UCF101, HMDB51, Kinetics-400, SSv2), maintaining strong performance even under extreme compression ratios (e.g., IPC = 1).
4. The experimental evaluation is extensive, covering diverse datasets, compression ratios, architectures, and ablation studies, providing convincing empirical support for the proposed framework.

### Weaknesses
1. The paper lacks a formal theoretical definition and justification of “temporal saliency,” and the TSGF design appears heuristic, without mathematical modeling or convergence analysis.
2. Using raw inter-frame differences may not capture complex motion semantics and may fail in scenes with camera motion or background clutter.
3. Although the method claims potential applicability to other video tasks, only preliminary results are presented for temporal action segmentation, without validation on detection, tracking, or video generation tasks.
4. The presentation quality could be improved; for instance, the font sizes in Figure 2 appear inconsistent, and Tables 3 and 4 use mismatched formatting, which slightly affects readability.

### Questions
1. Further clarification on the rationale for using frame differencing as the saliency measure would be valuable, possibly supported by additional theoretical analysis or empirical ablation to justify its necessity and effectiveness.
2. It remains unclear whether TSGF can robustly handle diverse motion types—including rigid, non-rigid, and camera-induced motion—and whether there exist failure cases for specific video categories.
3. The single-stage optimization strategy might be prone to local minima; theoretical or experimental analysis of its convergence and stability, especially on complex video datasets, would strengthen the paper.
4. A deeper discussion comparing the proposed temporal modeling with existing approaches (e.g., optical flow, 3D convolutional features, or recurrent architectures) would help highlight the unique advantages of the proposed method.

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
The paper proposes a uni-level video dataset distillation framework 
that directly optimizes synthetic videos against a fixed, pre-trained classifier while enhancing temporal fidelity 
through a Temporal Saliency Guided Filter (TSGF). 
The framework involves three key stages: 1. training a teacher model on real videos; 
2. optimizing synthetic videos by aligning cross-entropy loss and batch normalization statistics with the teacher model; 
and 3. applying a saliency-guided video augmentation that emphasizes motion-consistent regions. 
Temporal saliency is computed from inter-frame differences with smoothing and used both to modulate gradient magnitude during optimization 
and to adaptively select augmentation regions. Experiments on diverse video datasets demonstrate consistent improvements 
over prior dataset distillation methods, with ablations showing the contribution of each module.

### Strengths
1. Conceptually clean and efficient framework: 
The uni-level training scheme is straightforward yet effective, 
avoiding iterative teacher-student feedback while maintaining strong performance.

2. Temporal Saliency Guided Filter (TSGF): The proposed TSGF provides a principled way to incorporate motion cues 
without requiring optical flow or explicit temporal modeling, which enhances both interpretability and efficiency.


3. Comprehensive empirical validation: The framework achieves consistent gains across diverse datasets 
and varying compression ratios. 
Reported results show competitive scalability even under efficient distillation budgets.

4. Potential for broader application: The idea of saliency-weighted optimization could generalize 
to other video understanding or dynamic scene compression tasks.

### Weaknesses
1. Insufficient positioning relative to decoupled dataset distillation methods.
The pipeline (Sec. 3.2, Eq. (4)–(5)) resembles decoupled optimization schemes such as SRe2L, 
where a frozen teacher guides the synthetic data optimization. 
The distinction between the proposed uni-level framework and existing decoupled methods is not clearly explained, 
leaving the degree of conceptual novelty somewhat ambiguous.



2. Incomplete hyperparameter specification for the Temporal Saliency Guided Filter.
Critical parameters in Eq. (7) and Eq. (8), such as the smoothing window size $k$, weighting coefficients $\alpha_k$, 
and the $\epsilon$ constant for stability, are not reported. 
No sensitivity or ablation analysis is presented for these factors, which limits reproducibility and understanding of robustness.


3. Limited robustness and failure mode analysis.
While the paper claims the TSGF improves robustness under motion variation, 
there is no quantitative test for scenarios with strong camera motion or jitter. 
Failure analysis is limited to coarse class splits in Tab. 7 without qualitative inspection.

4. Restricted evaluation scope across model architectures.
The cross-model evaluation (Tab. 4) considers only small CNN-based backbones. 
There is no validation on transformer-based or modern video architectures, which limits the generalizability claim. 
Additionally, quantitative comparison with a modern decoupled baseline (e.g., SRe2L) is missing, with only a qualitative reference in Fig. 3.

### Questions
1. It would improve clarity to relocate Fig. 2 (framework overview) to the beginning of the Method section 
and Alg. 1 (optimization steps) to its end for smoother narrative flow.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This method proposes a uni-distillation framework for video datasets, introducing a Temporal-Spatial Gating Function (TSGF) for automatic frame attention and data augmentation during post-training. The approach achieves advanced results on various benchmarks, demonstrating robustness to dataset scale and motion strength.

### Strengths
1. Proposed method is effective at pruning redundant frames, making the distillation process significantly faster and more tractable for massive datasets and achieves new state-of-the-art accuracy on multiple benchmarks.
2. The TSGF is a well-motivated addition that explicitly addresses the challenge of preserving temporal dynamics, a critical weakness in previous video distillation methods.

### Weaknesses
1. Dataset distillation maintains performance with significantly smaller synthetic datasets; however, the baseline and its variants appear to underperform (e.g., 22.4% on K400). Since smaller models tend to overfit small-scale datasets, they may not adequately demonstrate the generalization capability of distilled datasets. While Table 4 shows that this method benefits larger models, could the authors further validate this finding using video models of standard scale rather than those with only a few layers?
2. The performance improvements differ between background-based datasets (K400) and motion-based datasets (SSv2), yet this paper lacks analysis of these differences. Additionally, marking the average number of samples per class would facilitate better understanding of the performance variations.
3. What are the sizes of the distilled dataset and the full dataset for MiniUCF, and what is the accuracy achieved on the full dataset?

### Questions
Please refer to Weakness

### Soundness
4

### Presentation
4

### Contribution
3
