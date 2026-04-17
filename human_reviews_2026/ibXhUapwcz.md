# Improving Black-Box Generative Attacks via Generator Semantic Consistency

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6, 2

## Abstract
Transfer attacks optimize on a surrogate and deploy to a black-box target. While iterative optimization attacks in this paradigm are limited by their per-input cost limits efficiency and scalability due to multistep gradient updates for each input, generative attacks alleviate these by producing adversarial examples in a single forward pass at test time. However, current generative attacks still adhere to optimizing surrogate losses (e.g., feature divergence) and overlook the generator’s internal dynamics, underexploring how the generator’s internal representations shape transferable perturbations. To address this, we enforce semantic consistency by aligning the early generator’s intermediate features to an exponential moving average (EMA) teacher, stabilizing object-aligned representations and improving black-box transfer without inference-time overhead. To ground the mechanism, we quantify semantic stability as the standard deviation of foreground IoU between cluster-derived activation masks and foreground masks across generator blocks, and observe reduced semantic drift under our method. For more reliable evaluation, we also introduce Accidental Correction Rate (ACR) to separate inadvertent corrections from intended misclassifications, complementing the inherent blind spots in traditional Attack Success Rate (ASR), Fooling Rate (FR), and Accuracy metrics. Across architectures, domains, and tasks, our approach can be seamlessly integrated into existing generative attacks with consistent improvements in black-box transfer, while maintaining test-time efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Semantically Consistent Generative Attack (SCGA), to enhance black-box transferability of generative adversarial attacks. While existing generative attacks optimize surrogate losses, the authors focus on the generator’s internal dynamics. They empirically observe that semantic recognizability degrades from early to late blocks and hypothesize that enforcing semantic consistency in early blocks improves transferability.
To achieve this, the authors apply the EMA Teacher to stabilize internal generator features that encode object semantics and a self-feature consistency loss that encourages student early-layer features to be similar to teacher’s features. Additionally, they propose a new evaluation metric, Accidental Correction Rate (ACR), to capture cases where adversarial perturbations inadvertently correct misclassified inputs, improving the interpretability of attack reliability. Extensive experiments across multiple architectures, domains, and tasks demonstrate consistent performance improvements over strong baselines.

### Strengths
1. The paper presents a clear empirical motivation by analyzing semantic drift across generator blocks for the proposed semantic consistency at the generator’s early intermediates.
2. The proposed method integrates the Mean Teacher mechanism with a self-feature consistency loss during training, which increases semantic consistency while not introducing no additional inference cost.
3. The introduction of the Accidental Correction Rate (ACR) metric provides an insightful perspective on attack reliability.

The paper conducts comprehensive experiments across multiple model families, datasets, and tasks, establishing strong empirical credibility.

### Weaknesses
Main concerns: 
1. The paper lacks theoretical analysis or formal justification for why enforcing early-layer feature consistency enhances cross-model transferability, relying mainly on empirical evidence.
2. In Table 2, the integration of the proposed method with the baseline leads to a slight drop in attack performance for some Transformer models (such as CDA and FACL with the proposed method on model p). Could the authors provide a brief analysis on the potential reasons for this performance degradation in these cases?
3. The paper does not evaluate against diffusion-based purification defenses, which are currently regarded as strong pre-processing defenses in adversarial robustness research. Including such a comparison would strengthen the work’s comprehensiveness.
Minor weakness:
1. In Figure 2, the snowflake symbol may unintentionally suggest that the teacher generator is entirely frozen; however, as it is updated iteratively via the EMA rule rather than by backpropagation, clarifying this distinction in the caption or text would avoid confusion.
2. In the ablation of the similarity threshold (shown in the supplementary material), the evaluation metric is not explicitly specified.

### Questions
1. Could the authors provide deeper theoretical intuition or analysis on why early-layer semantic anchoring leads to improved transferability?
2. Table 2 shows that the proposed method generally enhances transferability. However, for some cases such as the integration of the proposed method with the baseline CDA and FACL on Transformers (p), there is a decrease in attack performance. Could the authors briefly discuss the potential cause? 
3. Have the authors considered evaluating the proposed method against diffusion-based purification defenses (e.g., DiffPure[1]) to test its robustness against strong modern preprocessing defenses?

[1] Nie W, Guo B, Huang Y, et al. Diffusion models for adversarial purification[J]. arXiv preprint arXiv:2205.07460, 2022.

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
This paper proposes SCGA, a method for improving black-box adversarial transferability in generative attack frameworks. The approach employs a Mean Teacher architecture with EMA-updated weights to align intermediate features from the generator's early blocks with a teacher network, while maintaining the standard adversarial loss on surrogate model features. The authors partition the generator into early, mid, and late blocks and quantify semantic stability using foreground IoU variability across these blocks. Experimental evaluation is conducted across classification, semantic segmentation, and object detection tasks, with the introduction of an Accidental Correction Rate (ACR) metric alongside traditional ASR, FR, and accuracy measurements.

### Strengths
The paper is well-written and carefully structured. I commend the authors for the high-quality figures and tables, which significantly contribute to the clear presentation of the research.

### Weaknesses
Regretfully, after careful reading and consideration, I found the contributions of this paper to be predominantly incremental. The work applies established techniques without providing sufficient theoretical insights or novel algorithmic advancements to meet the high standard of ICLR.

=====Lack of technical novelty=====
1. The proposed attack techniques are quite similar to previous ones, making this paper seems like a patchwork. 
To name a few:
- Adversarial Loss Design: The adversarial loss formulation in SCGA directly corresponds to in Zhang et al. "Beyond ImageNet Attack" (ICLR 2022). Both employ identical surrogate feature-based similarity metrics in their loss functions.
- Generator Architecture and Training Pipeline: The generator block division (early/mid/late) and associated weighting strategy in SCGA replicate in Zhang et al. "Beyond ImageNet Attack" (ICLR 2022), including hyperparameter configurations and EMA implementation details.
- Domain-Invariant Generation Strategy: The domain-invariant generator objective in SCGA mirrors in Naseer et al. "Cross-Domain Transferability of Adversarial Perturbations" (NeurIPS 2019), sharing fundamental algorithmic approaches and frequency-domain processing methods.
- Intermediate Feature Alignment: The self-feature consistency mechanism (Equation 3) in SCGA substantially overlaps with in Krishna Nakka & Salzmann "Learning Transferable Adversarial Perturbations" (NeurIPS 2021). Both utilize intermediate layer feature similarity metrics with comparable layer selection strategies, differing only in the shift from direct alignment to EMA teacher alignment.
- Generator Training Framework: The generator training flow, architecture diagram, and projection operator P(·) in SCGA correspond to **Technical Point C** in Poursaeed et al. "Generative Adversarial Perturbations" (CVPR 2018), sharing norm constraint implementations and projection procedures.
- Semantic Clustering Methodology: The semantic region clustering and attention-based masking approach in SCGA duplicates **Technical Point C** in Aich et al. "GAMA: Generative Adversarial Multi-object Scene Attacks" (NeurIPS 2022), particularly in feature-clustering and Grad-CAM attention operations.
- Contrastive Learning Integration: The contrastive learning objective and frequency-domain randomization in SCGA overlap with **Technical Point C** in Yang et al. "FACL/PDCL" (2024), employing nearly identical CLIP-driven prompt mechanisms and domain robustness strategies.
To address this weakness, I strongly urge the authors set the core contribution apart from baseline techniques of others and switch the focus of this paper on the core contribution instead of these standardly adopted baseline techniques.

=====Lack of theoretical insights=====
2. This paper brings almost no theoretical insights. As a reader, I get no takeaways after reading this paper. All the mechanistic analysis methods in the paper are just standard techniques used in previous works and appeared to be included merely to pad out the content. 
Specifically:
- Figure 1 (Semantic Variability): The visualization of intermediate feature maps and foreground IoU variability is purely descriptive. It shows that early blocks retain more structure, but offers no theoretical explanation for why this occurs or how it fundamentally relates to transferability. The connection between "lower variability" and "higher transferability" is asserted, not derived from any theoretical principle.
- Figure 3 (Qualitative Results): The Grad-CAM comparisons and perturbation visualizations merely demonstrate that the method works—not why it works from a theoretical standpoint. Showing that perturbations align more with object regions is an empirical outcome, not an insight into the underlying mechanisms of adversarial generalization.
- Figure 4 (Feature Differences): The thresholded difference maps are visually intuitive but theoretically shallow. Highlighting where perturbations are added does not explain the fundamental reasons for improved transferability, such as the relationship between feature semantics and model decision boundaries.
- Spectral Energy Analysis (Table 6): The analysis of low-frequency and high-frequency energy across blocks is a measurement, not an insight. The authors note that their method alters spectral distributions but fail to explain how this connects to theoretical properties of adversarial examples (e.g., spectral bias, frequency-based generalization). The observations are correlational, not causal.
- Generator Intermediate Analysis: The partitioning of the generator into early/mid/late blocks is an architectural choice, not a theoretical contribution. The analysis does not provide a theoretical model for how or why semantic consistency in early blocks should propagate to enhance transferability—it only shows that it does so in practice.
- Loss Formulations: The consistency loss (Eq. 3) is a technical implementation detail. The paper does not justify it from a theoretical perspective (e.g., information theory, optimization theory, or generalization bounds). It is presented as a heuristic.


=====Incremental experimental improvements=====
3. Compared to the latest baseline PDCL and FACL, the cross-modal & cross-domain attack performance gain (Tab. 2 & Tab. 3) in average is only ~1%, a very incremental improvement. Comparatively, PDCL improves ~3%.
4. The baseline setup when evaluating the attack against defense models is very doubtful (Tab. 4). Only Zhang et al, 2022 is adopted as baseline. Why not also compare with PDCA, FACL, and GAMA? 
5. The fooling rate (FR) and accidental correction rate (ACR) is redundant and is not enough to be counted as a contribution.
Indeed, previous works usually adopt only one metric (e.g. PDCL adopts only accuracy) in evaluation. However, it is reasonable and fair as long as all the attacks are evaluated using the same metric. There is unnecessary to evaluate attacks using too many metrics, as the final conclusion does not change at all no matter which metric you use.
6. (Suggestion) Craft targeted adversarial examples to see whether the proposed SCGA can improve targeted transfer-based attack.
7. (Suggestion) The following two references are suggested to be cited since they are also generative-model-based cross-domain transfer attacks.
Li M, Deng C, Li T, et al. Towards transferable targeted attack. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 641-649.
Wang Z, Yang H, Feng Y, et al. Towards transferable targeted adversarial examples. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 20534-20543.

=====Awful writing=====
8. This paper did not cite the reference paper of each compared baseline to their acronyms. For the convenience of review discussion, I list them here:
[CDA] Naseer M M, Khan S H, Khan M H, et al. Cross-domain transferability of adversarial perturbations. Advances in Neural Information Processing Systems, 2019, 32.
[LTP] Salzmann M. Learning transferable adversarial perturbations. Advances in Neural Information Processing Systems, 2021, 34: 13950-13962.
[BIA] Zhang Q, Li X, Chen Y, et al. Beyond imagenet attack: Towards crafting adversarial examples for black-box domains. ICLR, 2022
[GAMA] Aich A, Ta C K, Gupta A, et al. Gama: Generative adversarial multi-object scene attacks. Advances in Neural Information Processing Systems, 2022, 35: 36914-36930.
[FACL] Yang H, Jeong J, Yoon K J. Facl-attack: Frequency-aware contrastive learning for transferable adversarial attacks. AAAI, 2024
[PDCL] Yang H, Jeong J, Yoon K J. Prompt-driven contrastive learning for transferable adversarial attacks. ECCV, 2024
9. In the abstract, what is “EMA”?
10. The font size of Fig.2 is too small, especially for the characters in the “residual learning” and “Upsampling” part. Please improve it.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Semantic-Consistency guided Generative Attack (SCGA ) for transfer-based black-box image attacks. Instead of only optimizing a surrogate loss (e.g., cross-entropy / feature divergence) on a perturbation generator, the authors align early generator features to an EMA “mean-teacher” copy to preserve object-aligned structure while crafting perturbations. They also introduce Accidental Correction Rate (ACR) to quantify cases where an attack fixes a model’s original mistake—arguably a blind spot of common metrics like FR/ASR/Accuracy. Empirically, the method plugs into several generative baselines and improves transfer across architectures (CNN/ViT/Mixer/Mamba), domains, and tasks (classification, segmentation and detection) without test-time overhead.

### Strengths
1.	Simple, orthogonal mechanism: The EMA teacher + early-block consistency integrates into several strong generative baselines without test-time overhead; the approach is easy to adopt.
2.	Clear, generator-internal motivation: The diagnostic showing that early generator features retain object contours while later ones blur them is compelling and grounded in measurable variability (foreground-IoU std across blocks).
3.	Broad evaluation: Cross-model results span CNN, ViT, Mixer, and SSM/Mamba families, with consistent average gains; the method also improves cross-domain (CUB, Cars, FGVC Aircraft) and cross-task transfer to SS/OD models.
4.	Robust-model/defense stress tests. The method improves over baseline against adversarially trained models and input preprocessing defenses (JPEG, bit-depth reduction, randomization & padding).

### Weaknesses
1.	Scope of gains & negative deltas. While averages improve, several cells in Table 2 are near-zero or negative (notably for certain transformer targets and PDCL).
2.	Frequency analysis lacks operational detail. The spectral-energy analysis is interesting but currently underspecified. Precisely define the transform (e.g., 2-D FFT with magnitude spectrum), the radial banding scheme (cutoffs in normalized frequency), and whether energies are computed on perturbations or activations. Provide explicit formulas (e.g., radial masks in the Fourier plane) and thresholds so others can reproduce the plots.
3.	Compute disclosure is incomplete. Training doubles forward passes (student+teacher), but only forward overhead is reported. Please report end-to-end training wall-clock, backward cost, peak memory, GPU/TPU model & count, and batch/step counts.
4.	Interplay with CLIP-driven baselines: The authors themselves note only marginal improvements when stacking on PDCL-style CLIP objectives, and briefly speculate that optimizing in CLIP's high-dimensional space may "override or dilute" the structural consistency enforced by SCGA. This explanation is underdeveloped and warrants a deeper investigation.
5.	Insufficient Ablation: While the ablation study in Table 5 is useful, it is incomplete. It demonstrates that applying the consistency loss to early blocks is optimal and that all proposed components contribute positively. However, it fails to fully disentangle the benefits of the EMA-updated teacher from the consistency loss itself. The reported gain from "MT" could stem from the general smoothing effect of weight averaging, or it could be that the consistency loss is only effective when provided with a stable teacher target. A crucial missing experiment would be to apply $\mathcal{L}_{cons.}$ without the EMA teacher (e.g., by using a frozen copy of the student from a previous iteration as the target). This would isolate the unique contribution of the temporal ensembling.
6.	No hyperparameter sensitivity. The paper introduces at least two critical hyperparameters, the EMA smoothing coefficient $\eta$ in Eq. 2 and the consistency loss weight $\lambda_{cons.}$ in Eq. 5. There are no analysis of the method's sensitivity to their values.

### Questions
1.	How sensitive are results to η (EMA smoothing), τ (similarity threshold), and the choice/number of early blocks? Any principled way to select them across generator backbones?
2.	Since training doubles generator passes (student+teacher), could you provide full wall-clock with backward, peak memory, accelerator type/count, batch size, and total steps/epochs—along with a forward vs. backward breakdown?
3.	Could you add side-by-side feature/attribution visualizations against the baseline (the Figure-3.1 setting) to substantiate “object-aligned” perturbations ?
4.	When SCGA is combined with PDCL/FACL, what specifically conflicts—frequency bands, spatial regions, or representation space mismatch (CLIP vs. surrogate)? Any mitigation (e.g., decoupled schedules)?
5.	What exact transform and banding do you use (e.g., 2-D FFT magnitude, radial frequency masks, normalized cutoffs), and are energies computed on perturbations or activations (per-channel or aggregated)? Please include explicit formulas and thresholds to ensure reproducibility.
6.	To disentangle the effect of the EMA teacher from $\mathcal{L}_{cons.}$, could you add a variant that applies the consistency loss without EMA (e.g., a frozen or lagged snapshot of the student as the target)? This would clarify the unique contribution of temporal ensembling.

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
This paper presents a new augmentation for generative adversarial attacks aimed at improving their transferability in black-box settings. The core idea is to enforce semantic consistency in the generator's early blocks using a EMA framework with a self-feature consistency loss, thereby stabilizing object-aligned intermediate representations. Then the authors  introduce a new evaluation metric called ACR to detect unintended benign corrections, aiming to provide a more comprehensive view of attack reliability beyond conventional metrics. Through extensive quantitative and qualitative evaluations, including ablation studies and spectral analysis, they demonstrate systematic improvements across diverse architectures, domains, and tasks.

### Strengths
1. The self-feature consistency loss is well-motivated and mathematically specified.

2.  The experiments are comprehensive, including multiple model architectures,  cross-domain and cross-task scenarios and fine-grained ablation.

3. The finding of  the semantic drift across generator layers that degrades black-box transferability is novel.

### Weaknesses
1. The baseline methods should be introduced before presenting the experimental results, as omitting this order significantly reduces readability.

2. Although the method claims to be architecture-agnostic, there is room for a stronger demonstration across more varied generator or victim types (e.g., diffusion)

### Questions
1. Why does the work only consider untargeted attacks as baselines? It seems that targeted attack methods (e.g., [1]) could also be included for a more comprehensive comparison.

2. How Spectral energy by band is defined and calculated in  Table 6?

3. How would the method compare against the strongest black-box transfer attacks that do not rely on generator-based pipelines (e.g., [2]) ? 



[1] Fang, Hao, et al. "Clip-guided generative networks for transferable targeted adversarial attacks." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[2] Wang, Xiaosen, et al. "Admix: Enhancing the transferability of adversarial attacks." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the intermediate features within the perturbation generator that are often overlooked in previous generative adversarial attacks.  The authors first note that a stronger attack better preserves the coarse shape from the early layers in the generator. Based on the observation, the authors introduce a lightweight EMA teacher to the early blocks during generator training, which regulates the features to maintain object contours and shapes. Extensive experiments show that the proposed strategy can serve as a plug-and-play technique to improve existing attack methods.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed method is simple and intuitive.
3. The authors provide sufficient experiments across different models and data domains to prove the effectiveness of their method.
4. Experiments about the intermediate block-level analysis are interesting and provide evidence of preserving image contours and shapes.

### Weaknesses
1. The designed method is simple and intuitive, which somewhat lacks novelty as feature-level guidance has been widely investigated in various existing studies.
2. As shown in Table 2, the proposed method only brings marginal improvements for powerful attacks such as PDCL, raising concerns about its necessity and effectiveness.
3. Lack experiments against input-processing-based defense methods.
4. In line 69, a missing full stop after "generator intermediate blocks".

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
