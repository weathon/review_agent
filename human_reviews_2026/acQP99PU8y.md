# TriQDef: Disrupting Semantic and Gradient Alignment to Prevent Adversarial Patch Transferability in Quantized Neural Networks

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Quantized Neural Networks (QNNs) are widely deployed in edge and resource-constrained environments for their efficiency in computation and memory. While quantization distorts gradient landscapes and weakens pixel-level attacks, it offers limited robustness against patch-based adversarial attacks—localized, high-saliency perturbations that remain highly transferable across bit-widths. Existing defenses either overfit to specific quantization settings or fail to address this cross-bit vulnerability.  
We propose \textbf{TriQDef}, a tri-level quantization-aware defense framework that disrupts the transferability of patch-based attacks across QNNs. TriQDef integrates: (1) a \emph{Feature Disalignment Penalty (FDP)} that enforces semantic inconsistency by penalizing perceptual similarity in intermediate features; (2) a \emph{Gradient Perceptual Dissonance Penalty (GPDP)} that misaligns input gradients across quantization levels using structural metrics such as Edge IoU and HOG Cosine; and (3) a \emph{Joint Quantization-Aware Training Protocol} that applies these penalties within a \emph{shared backbone} jointly optimized across multiple quantizers.  
Extensive experiments on CIFAR-10 and ImageNet show that TriQDef lowers Attack Success Rates (ASR) by over 40\% on unseen patch and quantization combinations while preserving high clean accuracy. These results highlight the importance of disrupting both semantic and perceptual gradient alignment to mitigate patch transferability in QNNs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces TriQDef, a tri-level quantization-aware defense framework designed to improve the robustness of Quantized Neural Networks (QNNs) against patch-based adversarial attacks. While quantization naturally distorts gradients and reduces vulnerability to pixel-level attacks, it remains weak against localized, transferable patch attacks across different bit-widths.

TriQDef mitigates this by integrating three components:
1. Feature Disalignment Penalty (FDP): penalizes perceptual similarity in intermediate features to enforce semantic inconsistency;
2. Gradient Perceptual Dissonance Penalty (GPDP): misaligns input gradients across quantization levels using structural metrics like Edge IoU and HOG cosine similarity;
3. Joint Quantization-Aware Training: optimizes these penalties jointly across multiple quantizers in a shared backbone.

Experiments on CIFAR-10 and ImageNet show that TriQDef reduces attack success rates by over 40% on unseen patch–quantization combinations while maintaining strong clean accuracy, demonstrating that disrupting both semantic and perceptual gradient alignment is key to improving QNN robustness.

### Strengths
1. The paper motivates the proposed components through quantitative analysis across different patch-based adversarial attack methods, QNN architectures, and numbers of quantized bits. With the quantitative results, the proposed components are easily convincing.

2. The proposed FDP and GPDP components are effectively breaking semantic alignment among bit-widths and perceptual similarities in the gradient maps through penalizing feature disalignment and gradient perceptual dissonance. Particularly, the proposed components utilize softDice and smooth HOG features to measure structural and textural alignments in feature and gradient space. Due to this, the proposed components are effective to be used for defending against patch-based adversarial attack methods.

3. The paper demonstrates the strength of the proposed components by comprehensive experiments. In the experiments, the prposed components reduce ASR numbers significantly.

### Weaknesses
1. In experiments, the paper only shows quantitative results on ResNet-56 and ResNet-34. It would be great if the paper included more results for different network architectures.

### Questions
1. Do the proposed components also reduce ASR numbers significantly across distinct bit-widths?

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
4

### Summary
This paper investigates the transferability of adversarial patches across QNNs with different bitwidths and concludes that QNNs are vulnerable to transfer attacks. To address this, the paper proposes TriQDef, an adversarial training (AT)-based defense by combining different bitwidth backbones and mitigating attack transferability in intermediate features during AT. Experimental results show that TriQDef improves robustness compared to three baselines: full-precision patch-based adversarial training (PBAT), quantized pixel-based adversarial training (DWQ), and full-precision pixel-based adversarial purification (DiffPure).

### Strengths
- Adversarial transferability across QNNs is an important topic. 

- This paper thoroughly investigates the current research frontier and proposes two novel techniques (FDP and GPDP) to address the problem.

- The technical novelty of the proposed defense is sufficient.

### Weaknesses
The text font size seems to noticeably decrease after Equation 2. The reviewer is not sure if this violates the ICLR policy.
There are major problems in the motivation experiments and baseline comparisons. 

1. Table 1 is somewhat confusing:
(1) Which model architecture is used for the surrogate model? 
(2) Which quantization method is used?
(3) The results seem not to fully support the conclusion that "Adversarial Patches Transfer Effectively Across Bit-Widths." The ASR gradually decreases as the quantization bitwidth of the target model decreases, indicating that larger divergence between the surrogate (32-bit) and target model bitwidths reduces attack transferability. Although the ASR remains high for 2-bit QNNs (70%+), this only suggests that adversarial patches are generally difficult to defend against across all CNN models. It is better to reframe the paper to focus on "leveraging the low attack transferability of QNNs for improved defense" rather than "addressing QNN vulnerability to transfer attacks."
(4) If the authors still want to emphasize "Adversarial Patches Transfer Effectively Across Bit-Widths," it is necessary to evaluate attack transferability using surrogate models with different bitwidths rather than only the 32-bit model, and the highest attack success rate across different surrogate bitwidths should be reported as the "aggregated ASR." For example, if attacking a 2-bit target model with 32-bit, 8-bit, 4-bit, and 2-bit surrogate models yields ASRs of 70%, 75%, 80%, and 90%, respectively, the aggregated ASR for the 2-bit target model would be 90%. This setting is much more realistic because defenders cannot anticipate the adversary's quantization bitwidth. In this setting, if the 2-bit model exhibits the highest aggregated ASR, it would support the conclusion that lower-bitwidth QNNs are more vulnerable to transfer attacks, justifying the need for the proposed defense. The authors should also control the surrogate (or target) model when adjusting the other. 

2. Table 2 has similar issues:
(1) Specify the model architecture as well as the quantization method and bitwidth for QAT and PTQ. In particular, was PyTorch Quantization used? Why is the quantization method inconsistent between Table 1 and Table 2? The authors should report results for (32, 8, 4, 2-bit) target models using the same quantization method combined with adversarial training.
(3) What do "8×8 (Unseen, 4-bit)" and "10×10 (Unseen, 2-bit)" mean? Are these adversarial patches crafted using 4-bit and 2-bit surrogate models, respectively?
(4) More comprehensive experiments involving adversarial training, as described above.

3. The design of Bit-Width-Aware Curriculum Training (BACT) resembles Double-Win-Quant (DWQ). If BACT does not offer significant novel contributions compared to DWQ, the authors should reduce the content about BACT.

4. The idea of training a robust model from the perspective of structural and textural features is novel. It would be better to emphasize FDP and GPDP as the core contributions and reduce descriptions of previous techniques, particularly in the Methodology section.

5. For patch-based adversarial training, this paper only compares with PBAT from 2020, which is outdated. More recent patch-based adversarial training papers should be included as baselines, for example:

Li X, Zhu Y, Huang Y, et al. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection. arXiv preprint arXiv:2506.23581, 2025.

Metzen J H, Finnie N, Hutmacher R. Meta adversarial training against universal patches. arXiv preprint arXiv:2101.11453, 2021.

6. The paper should compare with patch-based preprocessing defenses rather than the pixel-based defense DiffPure. DiffPure is designed for pixel-based Lp adversarial attacks and does not address adversarial patches. Comparing with DiffPure is unfair. Instead, please include the following patch-based preprocessing defenses. In general, it makes more sense to see if the proposed (training-based) defense can be integrated with patch-based adversarial purification for better performance.

Kang C, Dong Y, Wang Z, et al. Diffender: Diffusion-based adversarial defense against patch attacks. European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 130-147.

Jing L, Wang R, Ren W, et al. PAD: Patch-agnostic defense against adversarial patch attacks. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 24472-24481.

Liu L, Guo Y, Zhang Y, et al. Understanding and defending patched-based adversarial attacks for vision transformer. 2023.

### Questions
Was the text font size intentionally adjusted to be smaller after Equation 2?

### Soundness
3

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
2

### Summary
The paper looks at a specific gap: even when you quantize a model to very low bit-widths, patch attacks crafted on the full-precision model can still transfer and succeed. To fix this, the paper proposes TriQDef, a training-time defense with three parts: (i) make features from different bit-widths less aligned, (ii) make input gradients from different bit-widths less aligned, and (iii) use a bit-aware curriculum so multi-bit training doesn’t collapse. Experiments on CIFAR-10 and ImageNet with several patch attacks show lower attack success while keeping accuracy close to normal QAT.

### Strengths
a) The problem is well-motivated: “quantization ≠ free robustness” is a message that’s worth saying explicitly for patch attacks. 

b) The defense is mechanism-driven, not just “train on more patched images”: it tries to break cross-bit semantic/gradient alignment, which is a reasonable explanation for transfer. 

c) The idea of using perceptual cues (edge IoU, HOG-like descriptors) on features and gradients is a nice twist on older adversarial detection/defense lines that only used feature distances.

d) Everything happens at training time and keeps inference clean, which is nice for edge/QNN scenarios.

### Weaknesses
a) Parts of the idea are close in spirit to earlier “detect / separate / de-align” or “make internal representations less exploitable” work, but the paper doesn’t cite that line clearly. For example: MagNet (Meng & Chen, CCS’17), feature squeezing (Xu et al., NDSS’18), and transferability analyses (Tramèr et al., 2017/2018) all talk about representation/gradient similarity as a transfer channel. 
﻿
b) The method is engineered around patch attacks. It’s not metioned how much of TriQDef would still help for non-local, non-patch adversaries on quantized models (e.g. query-efficient, non-sparse perturbations). 
﻿
c) Enforcing pairwise disalignment across many bit-widths on a shared backbone can get expensive. It’d be good to spell out the cost vs. vanilla QAT.

### Questions
a) You show feature/gradient perceptual similarity is the real issue, not cosine. Do you know which layer range (early vs. mid vs. late) contributes the most, and could TriQDef be applied only there to save cost?
﻿
b) The bit-aware curriculum turns on lower-bit models gradually. How stable is this if we add even more bit-widths (e.g., 3-bit, 6-bit)?
﻿
c) if a vendor tool later re-quantizes/fuses ops, do you expect the disalignment effect to survive, or is it tied to the exact QAT pipeline you trained with?

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
This paper proposes a defense framework called TriQDef, aiming to address the vulnerability of Quantized Neural Networks (QNNs) to transferable patch attacks. TriQDef consists of three main mechanisms: feature displacement penalty (FDP), gradient-aware dissonance penalty (GPDP), and bit-width-aware curriculum training (BACT). Experimental results show that TriQDef can reduce the attack success rates of unseen patch and quantization combinations without sacrificing clean accuracy.

### Strengths
1. This paper addresses the critical and understudied problem of adversarial patch transferability in quantized neural networks.
2. The proposed tri-level defense is comprehensive.
3. The experiments covering various quantization bit widths, attack methods, and model architectures reflect the generalization ability of TriQDef in certain settings.

### Weaknesses
1. Lack of rigorous mathematical modeling or theoretical boundary analysis on the generalizability of the proposed defense (Q1).
2. Incomplete analysis of FDP's side-effects on semantic integrity. It reports a minor accuracy drop but fails to investigate how this disalignment affects fundamental recognition capabilities (Q2-Q3).
3. Evaluation and scalability concerns (Q4-Q7).

### Questions
1. The TriQDef framework relies on numerous heuristic hyperparameters ($\alpha$, $\beta$, $\lambda_{FDP}$, $\lambda_{GPDP}$, $k$, $q$, etc.). How to prove the tuning of these hyperparameters can ensure the generalizability of TriQDef across different model architectures and datasets.
2. In the face of real-time generated adversarial patches or video stream attacks, how stable is the defense mechanism of TriQDef? Does it require the design of an incremental training module?
3. The FDP utilizes hand-crafted descriptors (Edge IoU and HOG) that primarily capture low-level features. Will this reliance limit its robustness against novel attacks targeting mid-level semantic representations?
4. How is the scalability of TriQDef on larger models? Will it experience a decline in defense effectiveness under extreme quantization (such as 2-bit)? Is there a specific size threshold that leads to the failure of defense?
5. How is the spatial robustness of TriQDef (e.g., center vs. corner patches) under larger, unseen patch sizes?
6. The training procedure of TriQDef introduces additional complexity due to the simultaneous maintenance of multiple quantizers. Will this design impact training stability and convergence?
7. The experimental setup described on line 360 includes AlexNet and DenseNet-121. Why are the corresponding results of these models not reported in the main text or appendices?

### Soundness
3

### Presentation
2

### Contribution
2
