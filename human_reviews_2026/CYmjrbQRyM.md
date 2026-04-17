# ASMIL: Attention-Stabilized Multiple Instance Learning for Whole-Slide Imaging

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Attention-based multiple instance learning (MIL) has emerged as a powerful framework for whole slide image (WSI) diagnosis,  leveraging attention to aggregate instance-level features into bag-level predictions. Despite this success, we find that such methods exhibit a new failure mode: unstable attention dynamics.  Across four representative attention-based MIL methods and two public WSI datasets, we observe that attention distributions oscillate across epochs rather than converging to a consistent pattern, degrading performance. This instability adds to two previously reported challenges: overfitting and over-concentrated attention distribution. To simultaneously overcome these three limitations, we introduce attention-stabilized multiple instance learning (ASMIL), a novel unified framework. ASMIL uses an anchor model to stabilize attention, replaces softmax with a normalized sigmoid function in the anchor to prevent over-concentration, and applies token random dropping to mitigate overfitting. Extensive experiments demonstrate that ASMIL achieves up to a 6.49% F1 score improvement over state-of-the-art methods. Moreover, integrating the anchor model and normalized sigmoid into existing attention-based MIL methods consistently boosts their performance, with F1 score gains up to 10.73%. All code and data are publicly available at https://anonymous.4open.science/r/ASMIL-5018/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ASMIL, a modular framework to stabilize attention in attention-based MIL for weakly supervised WSI diagnosis. The method combines: (1) an EMA-updated anchor branch that provides a stable target attention distribution; (2) a normalized sigmoid attention in the anchor to reduce softmax over-concentration; and (3) token dropping for regularization. The anchor is discarded at inference. Experiments on BRACS and CAMELYON16/17 show consistent improvements over strong baselines and gains when plugging the module into existing MIL models. The paper highlights attention instability across training epochs and uses divergence-based measures and visualizations to motivate the approach. Anonymous code is released to support reproducibility.

### Strengths
- The paper focused on attention instability in attention-based MIL for WSI and provides quantitative and qualitative evidence (epoch-wise measures, attention maps) to motivate stabilization.
- The proposed module is simple and practical (EMA anchor + normalized sigmoid + token dropping), integrates with diverse MIL models, and adds no inference-time cost.
- Empirical results show consistent improvements across datasets and baselines, and the paper includes useful training/resource details and hyperparameter guidance (e.g., $m$ and $\beta$ ranges).

### Weaknesses
- Related work could more clearly situate ASMIL within teacher/EMA anchoring in self-supervised ViT literature (e.g., DINOv3 [1], BYOL [2]). Since DINOv3 discusses anchoring trade-offs (stability vs. adaptation), collapse risks, and EMA lag, a brief positioning of ASMIL relative to these mechanisms would help readers contextualize the anchor’s role and expected behavior.
- The stability–performance link is motivated but not statistically established: Including correlation analyses across seeds and hyperparameter sweeps and reporting significance would make the claim that reducing attention instability improves downstream performance more convincing.

[1] Siméoni, Oriane, et al. "Dinov3." arXiv preprint arXiv:2508.10104 (2025).            
[2] Grill, Jean-Bastien, et al. "Bootstrap your own latent-a new approach to self-supervised learning." Advances in neural information processing systems 33 (2020): 21271-21284.

### Questions
- How does ASMIL’s anchor compare to DINO/DINOv3 EMA teacher and anchoring strategies (objective/targets, momentum/lag tuning, collapse avoidance)? Do similar trade-offs arise?
- Can the authors provide statistical correlation/significance analyses linking stability metrics to performance across seeds and hyperparameter sweeps? Are there identifiable over-stabilization regimes?

### Soundness
3

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
4

### Summary
This paper investigates the instability of the attention scores in different training epochs during MIL training. In addition, the authors try to solve two more problems: over-concentrated attention distribution and overfitting. They propose several corresponding solutions: 1) an anchor model updated by EMA, 2) a normalized sigmoid function that replaces softmax, and 3) an effective token drop strategy. The comprehensive experiments have validated the effectiveness of this proposed method.

### Strengths
1.	The authors investigate a commonly neglected problem: the unstable attention distribution throughout the training that might be detrimental to MIL model training.

2.	The paper is well-written and easy to understand. The presentation of this method is quite clear.

3.	The solutions are intuitive and simple yet effective. The proposed solutions are not only a model but several plug-in modules that could be reused in other MIL models, which might broader impact in the community.

4.	The experiments are quite comprehensive and provide most of the ablation studies required to demonstrate the effectiveness of this method.

### Weaknesses
1.	In the model design, the online model is updated through backpropagation, and the anchor model is updated through EMA. They are both randomly initialized, and why the anchor model would help? The anchor model will start with random guess (which at least will not be near the correct attention distribution), and the EMA is somewhat restraining the anchor model within the starting point. Would it be better to start EMA after several epochs (so that the model has learnt something meaningful) to remove the noises of the first few epochs? 

2.	The authors are suggested to give a more comprehensive evaluation of the effectiveness of random drop strategy (e.g., whether this strategy is capable with other models, not just the proposed ASMIL). Also, it would be even better if the authors could compare this strategy with this paper published in ICML2025 “How Effective Can Dropout Be in Multiple Instance Learning?” This paper proposed to drop top-k most important instances. Overall, this would increase the solidity of the proposed strategy.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
4

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
This paper addresses key limitations of attention-based multiple instance learning (MIL) in whole slide image (WSI) analysis, a critical task in computational pathology. The authors first identify three major challenges of existing attention-based MIL methods: (PI) unstable attention dynamics (attention distributions oscillate across epochs instead of converging, quantified via Jensen-Shannon divergence), (PII) over-concentrated attention distribution (excessive focus on a few tiles), and (PIII) overfitting (due to limited WSI training samples). To tackle these issues simultaneously, the authors propose ASMIL (Attention-Stabilized Multiple Instance Learning), a unified framework with three core components: (1) an anchor model updated via exponential moving average (EMA) to stabilize attention dynamics by aligning the online model’s attention with the anchor’s via Kullback-Leibler (KL) divergence; (2) a normalized sigmoid function (NSF) in the anchor (replacing softmax) to prevent attention over-concentration, supported by theoretical analysis (Theorem 1); and (3) token random dropping to mitigate overfitting. Extensive experiments on three public WSI datasets (CAMELYON-16, CAMELYON-17, BRACS) and five non-WSI MIL benchmarks demonstrate that ASMIL achieves state-of-the-art (SOTA) performance (up to 6.49% F1 score improvement over baselines) and consistently boosts the performance of existing attention-based MIL methods (up to 10.73% F1 gain) when integrating its anchor and NSF modules.

### Strengths
1. **Novel Identification of a Critical Underexplored Problem**: The paper is the first to systematically identify and quantify "unstable attention dynamics" (PI) in attention-based MIL for WSI analysis. By measuring JSD between consecutive attention distributions and visualizing convergence failures (e.g., Figure 1), the authors highlight a previously overlooked limitation that harms both performance and interpretability—an important contribution to understanding MIL’s failure modes in WSI tasks.

2. **Unified and Theoretically Grounded Framework**: ASMIL effectively addresses three distinct challenges (PI–PIII) in a single framework, avoiding fragmented solutions. The NSF’s superiority over softmax (Theorem 1) provides rigorous theoretical justification for preventing attention over-concentration, while the EMA-based anchor model’s design (as a stable reference) is well-motivated and distinct from existing EMA teacher models (e.g., MHIM-MIL) that focus on hard-instance mining rather than attention stabilization.

### Weaknesses
1. **Incomplete Analysis of Edge-Case Performance**: While ASMIL outperforms baselines overall, it lags slightly behind SOTA on a specific setting: using an ImageNet-pretrained ResNet-18 backbone on CAMELYON-17, ASMIL’s AUC is 0.002 lower than the best baseline. The authors do not investigate this edge case—e.g., whether it arises from dataset-specific characteristics (e.g., CAMELYON-17’s multi-center staining variability) or interactions between the backbone and ASMIL’s components—weakening the claim of "SOTA across all datasets."

2. **Limited Discussion of NSF’s Application Scope**: The authors explain that NSF is applied only to the anchor (not the online model) to avoid vanishing gradients, but they do not explore potential workarounds (e.g., gradient clipping, modified sigmoid variants) to enable NSF in the online model. This missed opportunity raises questions about whether NSF could further improve performance if integrated more broadly, rather than being restricted to the anchor.

### Questions
1. **Anchor Model Hyperparameter Tuning**: The ablation study shows that an EMA factor \( m = 0.999 \) yields the best performance (Table 10). However, the authors do not explain why this specific value is optimal across datasets (e.g., CAMELYON-16 vs. BRACS) or whether \( m \) should be adaptively adjusted for WSIs with varying tile counts or sparsity (e.g., slides with <5% tumor regions). Could you provide insights into the sensitivity of \( m \) to dataset characteristics?

2. **Token Random Dropping Rationale**: The paper finds that a drop rate \( B \approx 0.5 \) is optimal (Figure 11). However, this value is derived from experiments on CAMELYON-16 and BRACS—does this optimal rate hold for other WSI datasets with different tissue types (e.g., prostate or lung WSIs) or non-WSI MIL tasks (e.g., MUSK)? Additionally, how does token dropping interact with the anchor model (e.g., does dropping affect the anchor’s ability to learn stable attention)?

3. **NSF vs. Alternative Normalization Functions**: The paper compares NSF to softmax (with temperature scaling) and entmax (Table 4, Table 5) but does not explore how NSF performs against other attention normalization methods (e.g., sparsemax, Tsallis entropy-based functions) in WSI scenarios with extreme class imbalance or high tile redundancy. Could you provide additional experiments or analysis to confirm NSF’s robustness across diverse WSI data distributions?

4. **Addressing Tiny Tumor Localization**: The authors note ASMIL’s limitation in focusing on tiny tumor foci (Figure 26). Beyond synthetic data, are there immediate modifications to ASMIL—e.g., adjusting the KL divergence weight \( \beta \) for small regions, or adding a small-region attention regularization term—that could mitigate this issue? Have you quantified how often this limitation occurs (e.g., percentage of WSIs with missed tiny foci) across datasets?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The dynamic instability of attention mechanisms leads to overfitting and excessive concentration in attention distribution. This paper proposes a framework called Attention-Stabilized Multiple Instance Learning, which employs an anchor model to stabilize attention. It replaces the softmax function in the anchor with a normalized sigmoid function to prevent over-concentration and mitigates overfitting through labeled random dropout. The approach demonstrates innovation, and final experiments show promising results.

### Strengths
1、By retaining the same attention module architecture and identical inputs, updates are performed using exponential moving averages instead of back propagation, thereby resolving the dynamic instability issues inherent in previous attention mechanisms.

2、Replacing softmax with a non-symmetric normal distribution in the anchoring model effectively mitigates attention over-concentration, ensuring stable and uniform attention distribution.

3、The paper introduces a feature token dropout mechanism, which randomly drops a portion of feature tokens during training while retaining all tokens during inference to mitigate overfitting.

4、The article provides a comprehensive overview, and the experimental content is rich.

### Weaknesses
1、The motivational section in Subsection 3.3 bears similarities to discussions on issues with attention mechanisms in related work. It requires further clarification of differences or highlighting of innovative aspects, and may also be subject to corresponding revisions or consolidation.

2、The paper compares its results with multiple baseline models, but some of these comparison models (such as CLAM-SB and DSMIL) lack in-depth discussion regarding the rationale for their selection and their applicability. Key differences in design between these models and ASMIL, their respective strengths and weaknesses, and how these factors influence performance across different types of tasks remain unexplored.

3、The font in the model diagram (Figure 2) is partially obscured, and some words appear in a smaller font size that is not sufficiently clear.

4、The paper proposes updating the anchor model via EMA to stabilize the attention distribution of the online model, yet it does not delve deeply into the relationship between these two components. Specifically, the interaction mechanism between EMA updates and backpropagation, as well as the performance of the online model during inference, are not sufficiently elaborated in the method description.

### Questions
1、Although the relationship between the anchoring model and the online model has been theoretically elucidated, their interactive mechanisms in practical applications may influence the final outcomes. Are there further details regarding how the anchoring model guides the attention mechanism of the online model? Does the anchoring model consistently demonstrate robust stability across diverse training scenarios?

2、This paper primarily validates the effectiveness of the ASMIL model in subtype classification and tumor localization tasks, but the multimodal learning framework can also demonstrate advantages in other tasks. Can this method be extended to other tasks?

3、The author proposes reducing overfitting risk by randomly discarding some feature labels, but this mechanism may impact model performance differently under various training settings. Could you provide more experimental data on how the label discard rate (e.g., the B value) affects model performance? Is there a recommended optimal discard rate, or should the discard proportion be adjusted based on different datasets?

### Soundness
3

### Presentation
3

### Contribution
3
