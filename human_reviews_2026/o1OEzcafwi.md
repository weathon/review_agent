# Adaptive Training of INRs via Pruning and Densification

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Encoding input coordinates with sinusoidal functions into multilayer perceptrons
(MLPs) has proven effective for implicit neural representations (INRs) of low-
dimensional signals, enabling the modeling of high-frequency details. However,
selecting appropriate input frequencies and architectures while managing parameter
redundancy remains an open challenge, often addressed through heuristics and
heavy hyperparameter optimization schemes. In this paper, we introduce AIRe
(**A**daptive **I**mplicit neural **Re**presentation), an adaptive training scheme that refines
the INR architecture over the course of optimization. Our method uses a neuron
pruning mechanism to avoid redundancy and input frequency densification to
improve representation capacity, leading to an improved trade-off between network
size and reconstruction quality. For pruning, we first identify less-contributory
neurons and apply a targeted weight decay to transfer their information to the
remaining neurons, followed by structured pruning. Next, the densification stage
adds input frequencies to spectrum regions where the signal underfits, expanding
the representational basis. Through experiments on images and SDFs, we show
that AIRe reduces model size while preserving, or even improving, reconstruction
quality. Code and pretrained models will be released for public use.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AIRe (Adaptive Implicit neural Representation), a novel adaptive training framework designed to refine the architecture of Implicit Neural Representations (INRs), specifically those based on sinusoidal functions, during the optimization process. The primary goal is to address the challenge of parameter redundancy.

### Strengths
1. The paper introduces AIRe, a novel adaptive training framework that effectively combines neuron densification and pruning
2. The numerical results consistently confirm the theoretical assumptions laid out for the AIRe framework

### Weaknesses
1. Lack of comparison with:
- SPDER: Semiperiodic Damping-Enabled Object Representation
- FreSh: Frequency Shifting for Accelerated Neural Representation Learning

2. Authors do not commenton the  relation between pruning and densification in INR and Gaussian Splatting
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering

which can be interesting for readers.

3. In related works, authors should mention that Gaussian Components can also reconstruct 2D images

- GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting
- MiraGe: Editable 2D Images using Gaussian Splatting

4. Since the method introduces multiple additional training phases (densification, TWD, fine-tuning), the real efficiency gain remains unclear.

5. The experimental evaluation primarily contrasts AIRe with generic pruning algorithms such as RigL and DepGraph, which are not tailored to INRs.

6. The proposed densification and pruning schedules rely on manually chosen hyperparameters (e.g., number of added neurons, pruning thresholds, or training epochs).

7. The ablation studies show that the densification component offers little benefit for the FINER architecture.

### Questions
1. How does AIRe compare quantitatively with recent adaptive INR methods such as SPDER or FreSh? 

2. Could the authors clarify the conceptual relation between pruning/densification in INRs and Gaussian Splatting techniques? 

3. Why are recent 2D Gaussian-based representations (e.g., GaussianImage, MiraGe) not discussed in Related Work? 

4. What are the actual training and inference time gains achieved by AIRe after including the extra adaptation phases? 

5. Why are only generic pruning baselines (RigL, DepGraph) considered instead of INR-specific adaptive models? 

6. Are the pruning and densification hyperparameters selected manually or adaptively during training? 

7. Why does the densification component provide little or no improvement for the FINER architecture?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AIRe (Adaptive Implicit Neural Representation), an adaptive training framework for INR, which primarily employs neuron pruning and densification strategies to adjust the network. The introduction of the TWD mechanism allows information from low-contributing neurons to be transferred to important neurons, theoretically ensuring the stability of pruning. Input frequency densification enhances the network’s representational capacity. Experiments across multiple tasks demonstrate that AIRe can reduce model size while maintaining or even improving reconstruction quality.

### Strengths
1. The paper proposes pruning and densification strategies for INR, including the TWD mechanism to transfer information from low-contributing neurons. 

2. Experiments across multiple tasks comprehensively validate the approach, showing that it reduces model size while maintaining or sometimes improving reconstruction quality. 

3. The work has potential significance for pruning and densification in the INR domain.

### Weaknesses
1.Some theoretical explanations are unclear. It is not specified how the 2ωj frequency is determined during densification and why this particular frequency is chosen.

2.Discussion of pruning effects on input and hidden layers is limited.For the SDF task (Lines 324–334), hidden layers are pruned, while for the image fitting task (Lines 378–385), input neurons are pruned. The paper only mentions that pruning the input layer may harm reconstruction.

3.The pruning threshold ϵ is not clearly defined; it is unclear whether it is fixed, layer-wise adaptive, or percentile-based. Clarification on how ϵ is chosen or tuned would be helpful.

4.Experiments mainly report final PSNR or CD. Including spectral visualizations and convergence curves would better illustrate how pruning and densification affect optimization and frequency coverage.

5.Experimental setup is somewhat unclear. In Table 1, it is not specified whether the reported “large” baseline is based on SIREN, FINER, or something else.

6.Some figures (3, 4, 6) could be improved for clarity. Figures 3 and 4 could be redesigned to provide richer visual comparisons, and the right half of Figure 6 is somewhat confusing in the information it intends to convey.

### Questions
1.Why is densification performed using 2ωj for new neurons rather than other frequency choices? Is there a theoretical justification or is it empirically determined?

2.In the densify-before-prune schedule, are newly added neurons immediately considered low-contributing and pruned by TWD? 

3.What does the “large” baseline in Table 1 correspond to (SIREN, FINER, or their average)?

4.What criteria guide the choice of pruning input versus hidden layers, and can more theoretical or experimental analysis be provided on the pruning effects on input and hidden layers?

### Soundness
3

### Presentation
2

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
This paper introduces AIRe (Adaptive Implicit Neural Representation), a training framework that progressively adapts a potentially overparameterized INR to the target data through two complementary operations: pruning and densification of neurons. It provides a general 
framework for the adaptive training of INRs, driven by pruning and densification. In the theoretical side, it leverage a harmonic expansion of sinusoidal neural networks to derive principled densification schemes, and prove stability of our neural networks under magnitudebased pruning. The method was mainly applied to SIREN and FINER for the experiments. Experiments were conducted on images, SDFs, and NeRFs,

### Strengths
The integration of pruning and frequency densification within INR training is innovative and addresses a key limitation—manual architecture tuning. In addition, the paper provides mathematical proofs (Theorem 1 and 2) explaining spectral densification and pruning stability, enhancing methodological rigor.

### Weaknesses
[1] The method only tested on low-dimensional signals (2D images, SDFs, small NeRF scenes). Therefore, it should be tested on different kinds of datasets. For example,  PDEs.

[2] One of the major drawbacks of INR is the long training time. By adding pruning and densification, will it increase the training time? An analysis of training time should be provided.

[3[How about the GPU comsumption? Like the Gfloop

[4]I understand that it was applied to SIREN and FINER and reports some results. However, some other baseline should also be added to the experiments.  For example, LosslessINR [r1] 
Han, Woo Kyoung, et al. "Towards Lossless Implicit Neural Representation via Bit Plane Decomposition." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
See the weakness

### Soundness
2

### Presentation
3

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
The paper addresses the challenges of parameter redundancy and limited representational capacity in implicit neural representations (INRs). It proposes AIRe (Adaptive Implicit neural Representation), an adaptive training framework that alternates between neuron pruning to remove redundant units and frequency densification to introduce additional input frequencies where the signal underfits. This process yields compact networks that better balance efficiency and reconstruction quality.

The method is validated on several signal representation tasks, including image reconstruction, 3D surface fitting, and neural rendering (NeRF). AIRe consistently matches or surpasses larger overparameterised baselines in reconstruction accuracy while using substantially fewer parameters, and outperforms general-purpose adaptive and pruning methods  (RigL, DepGraph). However, the gains are less pronounced on complex, practically relevant tasks such as NeRF-based view synthesis, where improvements over smaller fixed models remain marginal and size reduction benefits limited. Future revisions could strengthen the work by discussing the intended applicability of the approach beyond signal fitting, analysing inference-time efficiency, and broadening or clarifying comparisons with existing INR-specific pruning and sparsity methods.

### Strengths
**S1. Exploration of INR-specific network pruning.** The paper tackles a timely and important problem—adapting pruning strategies to the unique characteristics of implicit neural representations (INRs). This direction is both compelling and relevant, as multilayer perceptrons (MLPs) remain a major computational bottleneck in tasks such as neural rendering. The proposed approach demonstrates clear benefits over general-purpose pruning methods (Table 2), showing consistent and INR-aware improvements in efficiency–accuracy trade-offs.

### Weaknesses
**W1. Limited performance gains on relevant or real-world tasks.** The most extensive experiments (Table 8, supplementary) show only marginal improvements on the NeRF reconstruction task. The proposed method achieves roughly 20 % model-size reduction but delivers only minor accuracy gains over the same-size model trained from scratch. This raises questions about its effectiveness for complex, practically relevant scenarios such as neural rendering.

**W2. Missing analysis of inference efficiency.** While the paper discusses reductions in model size and training time (Table 7, supplementary), it does not analyse the effect of pruning on inference time, a major bottleneck for INRs in applications like real-time rendering and novel-view synthesis. Understanding how architectural adaptation impacts forward-pass latency is essential for assessing practical utility.

**W3. Incomplete comparison with prior INR pruning work.** The omission of Zell et al. (2022) from quantitative comparison is insufficiently justified. Although that method reduces model size only through input-layer pruning, it remains—by the authors’ own admission—“the only prior work exploring the pruning (or adaptation) of INRs” (l. 118–120) and should be included for completeness. Moreover, other relevant studies addressing sparsity or compression in INRs [1–2] are not discussed; establishing their relation to the proposed approach would clarify the work’s novelty and scope.


References

[1] Lee, J., Tack, J., Lee, N., & Shin, J. Meta-learning sparse implicit neural representations. NeurIPS 2021.

[2] Jayasundara, D., Rajagopalan, S., Ranasinghe, Y., Tran, T. D., & Patel, V. M. (2025). SINR: Sparsity-Driven Compressed Implicit Neural Representations. CVPR 2025.

### Questions
**Q1. Applicability to real-world INR tasks.** If the improvements on neural rendering (e.g., NeRF) remain minor, where do the authors envision this approach having the greatest practical impact? In which INR domains or signal types does adaptive pruning most clearly translate into meaningful efficiency or accuracy gains?

**Q2. Inference efficiency.** What is the impact of pruning and architectural adaptation on inference time—particularly for forward-pass latency in real-time or high-resolution INR settings—beyond the reported reductions in model size and training time?

**Q3. Comparison with prior INR pruning methods.** The paper would benefit from a deeper discussion of related INR pruning and sparsity works (e.g., Zell et al., 2022; Lee et al., 2021; Jayasundara et al., 2025). If feasible, these could be included as additional baselines; otherwise, it would be helpful to clarify why such comparisons were not performed and how the proposed approach conceptually differs from them.

### Soundness
3

### Presentation
3

### Contribution
2
