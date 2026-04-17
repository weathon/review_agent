# Boosting Domain Generalization in Object Detection through the Lens of Phase Invariance

- Decision: Reject
- Scores: 4, 2, 6, 4, 6

## Abstract
Temporal and seasonal variations in dynamic real-world environments result in diverse visual appearances, posing significant challenges for object detection models to maintain consistently high performance. Although existing Domain Generalization (DG) methods have shown promise in enhancing model robustness, they often neglect the spatial structural relationships of objects during the learning of domain-invariant features, thereby limiting their effectiveness in object detection tasks compared to classification tasks. From the perspective of Preserving Phase Invariance (PPI), we propose a novel methodology that aims to enhance model generalization while preserving accurate object localization. This methodology comprises three complementary modules: Mix Normalization Perturbation (MNP), which synthesizes diverse styles to improve robustness; Sensitive Channel Perturbation (SCP), which suppresses domain-specific features at the channel level; and Attention on Amplitude (AOA), which applies spectral attention to the amplitude component. Together, these modules promote phase-invariant representations and contribute to improved cross-domain detection performance. Our approach fundamentally reduces the domain generalization gap in classification and detection by maintaining the integrity of key structural information. Our proposed methods achieve state-of-the-art performance on Unsupervised Domain Adaptation and Single Domain Generalization Object Detection benchmarks, even outperforming most recent state-of-the-art Domain Adaptation techniques. The code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose a novel methodology from the perspective of Preserving Phase
Invariance (PPI). By incorporating three complementary modules, such as Mix Normalization Perturbation, Sensitive Channel Perturbation, and Amplitude-aware Attention, the propsoed method enhances the generalization ability of object detection models in cross-domain tasks. The overall idea is clear, the experimental results are comprehensive, and the proposed method achieves significant improvements, on the Single-Domain Generalized Object Detection task.

### Strengths
－ This paper proposes a new method for domain generalization object detection from the perspective of Preserving Phase Invariance.

－ The motivation and main idea of the paper are clearly presented. 

－ Extensive comparative experiments demonstrate the effectiveness of the proposed method, and ablation studies provide insights into the underlying mechanisms.

### Weaknesses
－ The paper’s formatting appears disorganized. For instance, in line 426 ("noise sampling") and line 377 ("Observation"), it is unclear whether these terms should be bolded or placed on separate lines. Consistency in formatting should be maintained throughout the manuscript. In addition, several figures and tables, such as Figure 4 and Tables 5–8, are presented in a cluttered and inconsistent layout, which affects overall readability and visual coherence.

－ There are numerous spelling, grammatical, and formatting errors throughout the manuscript. For example, "Table .2.2" in line 110 and "Table. 2.3" in line 146 contain formatting issues, while line 282 includes grammatical mistakes. Inconsistencies are also observed around Figure 7 and line 198. Similar problems appear in other parts of the paper and should be carefully checked and revised to ensure overall accuracy and consistency.

－ The analysis of frequency-domain information using the attention mechanism and the demonstration of phase invariance in Table 7 are interesting. However, as shown in Table 7, MNP appears to play the dominant role, while AOA contributes relatively less. I suggest the authors further investigate the AOA mechanism and strengthen the ablation study, for example, by designing a more innovative AOA structure, analyzing feature distribution changes before and after AOA, and evaluating detector performance under different domain conditions.

－Some methodological descriptions are vague and symbol definitions are inconsistent. For example, in the MNP module, the noise sampling process and the values of the mixing parameters ($w_1$, $w_2$, $\alpha$, $\beta$) are not clearly specified. Additionally, in line 274, the spatial feature map is defined as f, while in Figure 4 the output is denoted as x, which is inconsistent. Clarification and consistency are recommended.

－As noted in Section 4.2, FFDI, UAV-OD, and HybridAugment++ also aim to preserve phase invariance. It is recommended that the authors include experimental comparisons with these methods.

－In Tables 1 and 2, the authors claim that existing domain-invariant or consistency-based methods typically increase per-object classification accuracy while degrading localization quality. However, the limited number of compared methods reduces the persuasiveness of this claim.

－UDA methods are now well established and achieve strong performance in C2F and Real-to-Artistic scenarios. In Tables 3 and 5, the number of compared methods is limited, and the performance gains on C2F are modest. It is therefore recommended that the authors focus primarily on the single-domain generalization task.

－ There are inconsistencies in the reported results, for example, the mAP values in Table 7 of the main text differ from those in Tables 12–15 in the appendix. The authors should provide further clarification and explanation for these discrepancies.

### Questions
－ The paper’s formatting appears disorganized.

－ There are numerous spelling, grammatical, and formatting errors throughout the manuscript.

－ In-depth Exploration and Analysis of the AOA Mechanism.

For details, please refer to the Weaknesses section.

### Soundness
3

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
This paper studies domain generalization for object detection, aiming to improve model robustness across unseen domains. From the perspective of Preserving Phase Invariance (PPI), the authors propose a method consisting of three components: (1) Mix Normalization Perturbation (MNP), which synthesizes diverse styles to improve robustness; (2) Sensitive Channel Perturbation (SCP), which suppresses domain-specific features at the channel level; and (3) Amplitude-aware Attention (AOA), which applies spectral attention to the amplitude component. Experiments on Unsupervised Domain Adaptation and Single Domain Generalization Object Detection benchmarks demonstrate improved performance compared with several baselines.

### Strengths
1. The ablation study is thorough and analyzes the contribution of each module.

2. Experiments are conducted using different backbones and detectors, which supports the robustness and generality of the proposed approach.

### Weaknesses
1. The proposed method seems to combine existing techniques. The authors themselves cite related works for each component in the introduction, making it unclear what the actual innovation is.

2. Many works have already explored using frequency-domain information for domain adaptation or generalization [a,b,c]. The paper should explicitly discuss and compare differences with these methods.

[a] FDA: Fourier Domain Adaptation for Semantic Segmentation

[b] Spectral Unsupervised Domain Adaptation for Visual Recognition

[c] SA-GDA: Spectral Augmentation for Graph Domain Adaptation

3. The literature review and problem analysis are somewhat outdated. The limitation of enforcing only image-level semantic consistency for object detection has been widely studied. Many recent works address detection-specific adaptation or domain generalization, but the comparisons here are limited to older methods.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

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
This paper addresses the challenge of domain generalization (DG) in object detection, where existing classification-oriented DG methods fail to transfer effectively due to their neglect of spatial structural consistency. The key insight is that while classification requires semantic consistency, detection additionally demands geometric stability (localization accuracy). The authors propose ‌Phase Invariance (PPI)‌ as a core principle, leveraging the Fourier transform's property that phase preserves spatial structure while amplitude encodes domain-specific style. By enforcing phase stability during feature extraction, the method ensures robust localization under domain shift. 
The contributions include: (1) Diagnosing the limitation of classification-focused DG in detection, highlighting the necessity of structural consistency; (2) Introducing PPI as a frequency-domain principle to decouple style (amplitude) from geometry (phase); and (3) Proposing a ‌divergence-to-convergence framework‌ with three modules. This approach significantly improves cross-domain detection performance by aligning both semantic and spatial representations.

### Strengths
1. The introduction of phase invariance is quite interesting.

2. The proposed method in the paper demonstrates promising performance.

### Weaknesses
1. The paper employs a significant number of abbreviations (e.g., PPI, MNP, LNP, AOA, SCP), some of which are not defined or cited, making the paper difficult to follow.

2. In the MNP module, the variable $y_{style}$ is never mentioned again in subsequent equations. What is the purpose of this variable? Additionally, why do the two types of perturbations defined for $y_{style}$ effectively mimic domain shifts? More analysis is needed—what are the specific use cases for $y_{style}$?

3. It appears that only the AOA module involves an FFT transformation to decompose the signal into frequency and phase domains. Since the other two modules lack this constraint, could they potentially disrupt phase consistency?

### Questions
1. What are INP and LNP? The paper does not provide clear definitions.

2. Are "Temporal and seasonal" the core and only factors in domain generalization, and are they the primary focus of this paper? The title emphasizes improving domain generalization, but the paper seems to highlight the advantages of the proposed method specifically in "Temporal and seasonal" aspects. Does this imply that the method is only applicable to certain domain generalization scenarios?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper attacks single-source domain generalization for object detection by arguing that prior DG work optimizes image-level semantic invariance and therefore sacrifices spatial accuracy.  It introduces “Preserving Phase Invariance” (PPI): amplitude can vary across domains but phase is kept fixed, so object geometry is explicitly conserved.  Three modules are designed to implement PPI in a CNN detector: Mix Normalization Perturbation (MNP) diversifies shallow-layer styles without touching phase; Sensitive Channel Perturbation (SCP) suppresses amplitude-dominant channels that react to domain shift; and Amplitude-aware Attention (AOA) re-weights low-frequency amplitude bands that carry stable cues.  Extensive experiments on C2F, Diversity-Weather and Real-to-Artistic show +5–14 mAP over strong DG/DA baselines, and ablations on Faster-R-CNN, RetinaNet and DETR confirm that each module is necessary and architecture-agnostic.  The work is the first to bring explicit phase conservation into DG detection and achieves SOTA on SDGOD benchmarks while adding <1 % parameters and 5 % inference time.

### Strengths
1. The writing of paper is good.
2. The motivation is interesting.

### Weaknesses
While the authors prove that normalization-style perturbations leave phase unchanged, they do not analyze under which conditions the entire CNN stack preserves phase or how non-linear activations, stride or padding affect the constraint; the claim that “geometry is exactly conserved” is therefore asserted rather than rigorously guaranteed.

All experiments are vision-only and revolve around weather, lighting and artistic style; there is no evaluation on more realistic geospatial or sensor-shift scenarios (different camera intrinsics, LiDAR-to-RGB, cross-country datasets) where phase might be less reliable.

HybridAugment++, DFF and UAV-OD already manipulate amplitude; the paper positions them as “amplitude-only” baselines, but does not compare directly under the same single-source protocol or adopt their stronger augmentation recipes, so the incremental benefit of explicit phase locking is not fully isolated.

The method loses 0.4–0.8 mAP on some clear-weather categories (Bike in Table 12); the authors do not explain when PPI hurts and whether it amplifies low-frequency artefacts such as shadows or lens flare.

### Questions
How does phase invariance interact with modern augmentation heavy pipelines (large-scale jitter, MixUp, Copy-Paste) that explicitly warp or paste objects and thereby modify phase?

The AOA module focuses on low-frequency amplitude; could high-frequency phase edges be equally important for small-instance localization, and would a dual attention mechanism help?

What happens when the source domain itself contains large motion blur or defocus—does the memory bank in SCP erroneously flag structurally important but “sensitive” channels?

The paper claims “architecture-agnostic” improvements, yet DETR gains are largest; is the benefit simply due to the fact that global attention layers already mix phase information, and would the same hold for conv-next or Swin backbones?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a well motivated approach to improving domain generalization in object detection. The authors first identify a key limitation of classification oriented DG methods, showing that they fail to transfer to detection because they ignore the need for spatial structural consistency, leading to localization degradation. To address this, they introduce the principle of Preserving Phase Invariance (PPI) which maintains phase information for object geometry allowing amplitude variations associated with style changes, Building on this, the authors propose a divergence-to-convergence framework with three modules: Mix Normalization Perturbation (MNP) to generate diverse styles and enhance robustness, Sensitive Channel Perturbation (SCP) to suppress domain-specific feature channels and Amplitude-aware Attention (AOA) to apply spectral attention on the amplitude spectrum and emphasize domain-invariant cues. Together, these modules aim to produce structure preserving representations reducing the domain generalization gap in classification and detection tasks.

### Strengths
The authors provide a compelling analysis to show that the classification focused DG approaches fail in detection as they ignore spatial structural consistency.

Emphasizing phase preservation as a way to maintain object geometry is conceptually appealing.

The proposed pipeline MNP, SCP and AOA is conceptually coherent, each component is simple and complements the others, combining feature diversification and convergence toward domain-invariant representations.

### Weaknesses
Despite its promising direction, the paper has several limitations that needs further clarification:

While normalization-based perturbations are shown to preserve phase, there is no analysis of whether non-linear activations, strided convolutions or padding in the network also preserve phase. Therefore the claim that “geometry is exactly conserved” is asserted without any proof.

Some variables in the MNP equations are introduced only once and never used again. Noise parameters, mixing weights and feature map symbols are inconsistently used throughout the paper.

The paper does not fully analyse stability or robustness across parameter variations. As an example SCP depends on threshold percentiles and EMA parameters, while MNP involves noise sampling choices.

The theoretical grounding for channel perturbation and amplitude attention is not  thoroughly explored.

Experiments are mainly focused on weather shifts and artistic style shifts and diverse domain gaps remain untested limiting claims of generality. Comparisons to DG-transformer hybrid is missing, weakening the claim as architecture-agnostic and universally beneficial.

### Questions
I have few questions that if answered would help clarify the methodology and improve the paper’s clarity:

Some variables in the MNP equations (e.g., mixing parameters, noise terms) appear once and are never used again. Can you please explain why? How do the two types of perturbations correspond to real-world domain shifts?

Some results in the main text (e.g., Table 7) do not match their counterparts in appendix Tables 12–15. Can you clarify which numbers are correct and explain the discrepancies?

How sensitive is the method to the choice of perturbation strength, sampling distributions and SCP threshold percentiles? Could you provide guidelines or sensitivity analysis?

### Soundness
3

### Presentation
3

### Contribution
3
