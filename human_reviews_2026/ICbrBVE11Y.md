# Progressive Residual Tensor Networks for Adversarial Purification

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Adversarial perturbations remain a critical threat to modern vision systems, and tensor network–based purification is a promising direction thanks to its low-rank priors. Yet these methods face a fundamental reconstruction–denoising conflict: preserving pixel fidelity risks recovering adversarial residues, whereas aggressive suppression sacrifices semantic detail. We address this trade-off with a Laplacian pyramid–inspired framework that performs progressive residual reconstruction across scales. At coarse levels, the rank is kept relatively unconstrained to capture global semantic structure, while a monotonically decreasing rank schedule constrains capacity at finer levels, preventing the accumulation of fine-scale perturbations. In addition, we introduce a wavelet-based residual regularization that penalizes the energy of reconstructed high-frequency subbands, discouraging residuals from absorbing adversarial noise without undermining semantic recovery. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet show that our method preserves strong reconstruction quality on clean images while substantially improving robustness under diverse attacks, demonstrating the effectiveness of a frequency-aware, progressive tensor-network purifier.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a tensor-network-based Adversarial Purification (AP) method, termed Progressive Residual Tensor Networks. It follows a recent line of research exploring the reconstruction of adversarially perturbed images through tensor-network representations. The proposed framework aims to alleviate the inherent trade-off between reconstruction fidelity and adversarial robustness.

The method comprises three key components:

1) Laplacian Pyramid-based Reconstruction: The input image is decomposed into layer-wise frequency bands, and each band is reconstructed by a dedicated model specialized for that frequency range.

2) Progressive Residual Refinement: The reconstruction proceeds progressively, ensuring that each subsequent level focuses only on the residual frequency components not captured by previous stages.

3) Wavelet-based Residual Regularization: An additional wavelet-domain regularization term is introduced into the reconstruction loss to enforce smoothness and prevent overfitting to adversarial noise.

Overall, the idea of leveraging a quantized tensor-network representation for reconstructing clean examples is novel, and the authors make commendable efforts to prevent the recovery of adversarial perturbations from high-frequency bands.

### Strengths
1) The paper presents an efficient and effective adversarial purification (AP) approach by quantizing layer-wise representations before feeding them into a Tensor-Train (TT) structure. It further leverages a progressive residual refinement principle to preserve semantic content while avoiding the reconstruction of adversarial perturbations.

2) Figure 4 provides a clear and expressive visualization of the reconstructed clean examples. The work offers valuable insights for designing purification methods that jointly consider reconstruction fidelity and residual-based refinement, highlighting the potential of tensor-network-driven approaches for systematic adversarial robustness.

### Weaknesses
1) The adversarial robustness evaluation focuses mainly on classification tasks. Hence, I suggest improving the presentation by clarifying ambiguous notations (e.g., avoid using $y$ to represent data samples).

2) The performance improvement appears limited. The proposed method’s robustness gain seems to depend heavily on using a robust classifier; when evaluated with a standard classifier, the results significantly fall below baseline performance. Hence, the experimental evidence weakens the claimed effectiveness of the approach.

3) The pipeline Figure 1 could be better organized and the Algorithm 1 could be expanded or separated for atomic presentaitons. Adding clearer subheadings and workflow guidance would improve readability and help convey the method’s structure more effectively.

### Questions
1) Since the method is built upon a Tensor-Train (TT) framework with efficient quantization, I suggest that the authors compare the inference cost with existing diffusion-based adversarial purification (DBP) methods, since DBP methods have been well explored as a strong purification paradigm. Such a comparison could better highlight the efficiency advantages of the proposed AP method.

2) For each input resolution, the method requires training $k$ separate tensor networks. While this design accounts for sensitivity to resolution, it substantially increases the training cost. Moreover, datasets such as CIFAR-10 and ImageNet differ greatly in resolution, leading to varied pooling depths and numbers of tensor networks, which may limit generalization to downstream tasks. Could the authors discuss whether potentially unifying the tensor-network architecture could significantly improve the overall soundness and scalability of the method?

3) The evaluation is limited to AutoAttack. To more rigorously validate the method, it should be tested against stronger adaptive attacks, such as PGD+EOT and BPDA+EOT, as well as purification-specific adaptive attacks like Diff-PGD [1] and DiffAttack [2]. A broader evaluation under various adaptive settings would strengthen the empirical credibility of the proposed defense.

4) More implementation details are needed, including the tensor-network configurations, AutoAttack parameters, and the design of the adaptive attack setup used to evaluate the defense.

5) It would be helpful to include a visual comparison or supporting evidence demonstrating that coarse-to-fine tensor networks [3] tend to recover adversarial perturbations more frequently than the proposed method. Since the work is inspired from the previous findings[3].

[1] Xue, et al. "Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability", NeurIPS 2023.

[2] Kang, et al. "DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification", NeurIPS 2023.

[3] Loeschcke, et al. "Coarse-tofine tensor trains for compact visual representations", ICML 2024.

### Soundness
3

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
To address the inherent conflict between reconstruction fidelity and denoising robustness in tensor-network-based defenses, the paper proposes a Progressive Residual Tensor Network for adversarial purification. The core idea is to introduce a Laplacian pyramid–inspired progressive residual reconstruction that decomposes the image into frequency bands and reconstructs residuals across scales. Specifically, at coarse levels, the model captures global semantics with higher rank capacity, while finer levels use decreasing ranks to suppress high-frequency adversarial residues. In addition, wavelet-based residual regularization (WRR) penalizes high-frequency energy, discouraging adversarial noise amplification. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet demonstrate superior robustness and competitive clean accuracy compared to state-of-the-art adversarial training and purification baselines.

### Strengths
1.	The design of progressive residual reconstruction and wavelet-based regularization is valuable, offering a frequency-aware solution to the reconstruction–denoising conflict in tensor-based defenses.
2.	The paper provides a clear motivation grounded in frequency-domain analysis and prior Laplacian pyramid theory, which justifies the proposed multi-scale residual modeling.
3.	The proposed method shows superior robustness and competitive clean accuracy across various datasets, exhibiting strong practicality.

### Weaknesses
1.	More design details about the rank schedule are needed. Although the experiments in Table 2 preliminarily compared the performance of different reconstruction settings, a more detailed design of the decreasing rank schedule (e.g., how to select and tune the value, interval, and corresponding resolution of ranks) remains heuristic. Providing analytical or empirical guidance for rank selection will enhance reproducibility.
2.	A broader range of attack strategies should be considered. Specifically, this study evaluates the proposed method only under AutoAttack. It would be beneficial to include analyses or experiments involving more recent and sophisticated attacks that demonstrate stronger capabilities in bypassing purification-based defenses, such as BPDA [A] and DiffHammer [B].
3.	A more detailed description of the white-box setting used in the proposed method is necessary. Specifically, AutoAttack comprises multiple modes, including both black-box and white-box attacks. It remains unclear how the proposed purification module is exposed to white-box adversaries. Since this aspect critically influences the validity of the robustness evaluation, a clearer and more thorough explanation is required.
4.	Lack of analysis on computational overhead. Specifically, the paper does not sufficiently discuss the training and inference cost of multi-stage tensor reconstruction, which may be substantial given the multiple QTT decompositions per resolution level. A runtime or memory comparison against prior works would be valuable.
5.	In addition to adversarial purification and adversarial training, adversarial detection methods [C] [D] [E] also achieve defense against attack samples. It would be valuable for the authors to further discuss and compare these approaches, highlighting their similarities and distinctions relative to the defense strategy proposed in this work.

**Minor issues**

1.	The pseudo-code can be reformatted into a consistent two-column layout. The current presentation is hard to read.

2.	There is a figure index error in section 4.4.

### Questions
None

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
Authors propose a new adversarial purification method based on Feature Pyramid Networks (FPN) that progressively recovers the clean component. Each level filters a distinct frequency band, so layers do not interfere with each other. A Haar wavelet–based regularizer is applied at every layer to trim the high‑frequency domain at each resolution level.

### Strengths
1. The architecture of the progressive residual reconstruction is clearly explained and overall reasonably motivated.
2. The usage of Wavelet-Based Residual Regularization is supported by Figure 2a.

### Weaknesses
1. The absence of reported time measurements is a significant limitation. Quantifying runtime efficiency is crucial for assessing the method's suitability for real-time applications and for clearly defining its practical use cases. I would also like to see a discussion about the method complexity.
2. The manuscript lacks a defined strategy for selecting the decomposition rank, as well as any method for rank adaptation. This is a noticeable oversight, as the optimal rank is likely dependent on image-specific factors such as resolution and detail complexity.
3. The evaluation is limited to a 2-level pyramid, and the rationale for this constraint is not provided. Clarification is needed on whether this choice was due to empirical performance plateaus, computational trade-offs, or degradation in output quality with additional levels.

### Questions
1. Could you clarify what is the reason behind using "sub-bands" over which you compute $\ell_1$ norm (P.6, eq.9)? 
2. It would be beneficial to report compression ratios to better understand the method. 
3. What is implied by "Strategy A" and "Strategy B" in Section 5.3?
4. Did you try wavelet transform separately from Tensor network purification?

Minor presentation issues:
- At the top of Figure 1 rank should be $r_{d-l}$ instead of $d-l$.
- $A^{1}$ is repeated in Figure 1.
- In Figure 1, the label near the right arrow should be $y_d$, not $x_d$.
- Placing algorithm in a two-column mode makes it more difficult to analyse.
- In the second column of the Algrotihm 1, $\mathcal{L}$ should have a subscript index.
- Table 2 does not have any dataset reference.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Proposes a Laplacian pyramid–inspired progressive residual reconstruction for tensor-network adversarial purification, with resolution-dependent rank scheduling and a Haar wavelet–based residual regularization (WRR) on high-frequency subbands. Evaluated on CIFAR-10/100 and ImageNet under AutoAttack, with comparisons to AT/AP baselines.

### Strengths
- Clear articulation of the reconstruction-denoising conflict and a structured attempt to address it via scale separation and capacity control.
- Progressive residual formulation is simple, potentially broadly applicable, and compatible with existing classifiers.
- Includes comparisons to several AT/AP methods and a basic reconstruction-strategy ablation.

### Weaknesses
- Justification of Laplacian choice is insufficient: the paper motivates LP historically, but does not compare to alternative pyramids such as Gaussian pyramids or wavelet pyramids, nor to multi-level wavelet reconstructions that more directly align with the WRR penalty. The only reconstruction-strategy ablation is Table 2 (direct vs progressive vs LP) with limited scope.

- Ablations are thin: Table 2 is the sole substantive ablation; there is no study varying the decomposition family, number of scales, or alternative rank policies beyond a few QTT rank pairs.

- Subband handling is oversimplified: WRR penalizes the concatenated high-frequency bands, but no per-band analysis (LH/HL/HH individually or pairwise) is provided, despite claims about “high-frequency” components; Fig. 2a labels “high vs low”, while Haar yields LL, LH, HL, HH.

- Trade-off instability: In Table 2, settings that improve robust accuracy tend to hurt clean (IID) accuracy and vice-versa, and the method without a robust backbone trails strong AT in some settings; the paper does not resolve when LP+WRR is preferable.

- Protocol clarity: Table 1 aggregates results across datasets with different classifier backbones (ResNet-50 vs WideResNet-28-10), complicating method-to-method comparison. QTT rank settings used for headline Table-1 numbers are not explicit.

- Baseline coverage and consistency: Although several AT/AP baselines are listed in Table 1, alignment with common RobustBench protocols is unclear, and baseline choices/backbones differ across datasets, reducing comparability.

- Missing related work: No mention of FLC/anti-aliasing pooling literature (e.g., [1]) that is relevant to frequency-aware robustness and could inform design choices.

References:
[1] Grabinski, Julia, et al. "Frequencylowcut pooling-plug and play against catastrophic overfitting." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

### Questions
- Decomposition choice: Why Laplacian pyramid over Gaussian or wavelet pyramids? Could multi-level wavelet reconstruction subsume LP+WRR?

- Subband granularity: WRR aggregates LH/HL/HH. What happens if only HH is penalized, or LH/HL individually, or LH+HL without HH? Any per-band sensitivity that supports the “high-frequency energy” claim beyond the aggregate.

- Headline settings: Which QTT rank/scale settings and WRR λ were used for the headline Table-1 results on each dataset?

- Backbone consistency: Why switch to WideResNet-28-10 on CIFAR-100 while using ResNet-50 elsewhere. Can the main table be normalized to a single backbone per dataset for fair comparison?

- Benchmark alignment: Were the AT baselines reproduced under RobustBench-style evaluation, or are they literature numbers with heterogeneous training data/regularization. If the latter, can a standardized subset be added?

### Soundness
2

### Presentation
2

### Contribution
2
