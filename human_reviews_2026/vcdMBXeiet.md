# Training-Free Defense Against Adversarial Attacks In Deep Learning MRI Reconstruction

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Deep learning (DL) methods have become the state-of-the-art for reconstructing sub-sampled magnetic resonance imaging (MRI) data. However, studies have shown that these methods are susceptible to small adversarial input perturbations, or attacks, resulting in major distortions in the output images. Various strategies have been proposed to reduce the effects of these attacks, but they require retraining and may lower reconstruction quality for non-perturbed/clean inputs. In this work, we propose a novel approach for mitigating adversarial attacks on MRI reconstruction models without any retraining. Based on the idea of cyclic measurement consistency, we devise a novel mitigation objective that is minimized in a small ball around the attack input. Results show that our method substantially reduces the impact of adversarial perturbations across different datasets, attack types/strengths and PD-DL networks, and qualitatively and quantitatively outperforms conventional mitigation methods that involve retraining. We also introduce a practically relevant scenario for small adversarial perturbations that models impulse noise in raw data, which relates to herringbone artifacts, and show the applicability of our approach in this setting. Finally, we show our mitigation approach remains effective in two realistic extension scenarios: a blind setup, where the attack strength or algorithm is not known to the user; and an adaptive attack setup, where the attacker has full knowledge of the defense strategy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a training-free defense framework to mitigate adversarial perturbations in deep learning based MRI reconstruction, particularly for physics-driven deep learning (PD-DL) networks such as MoDL.

The proposed defense exploits cyclic measurement consistency (CMC): by resimulating undersampled k-space measurements from model reconstructions and enforcing self-consistency, the method detects and corrects adversarial perturbations without retraining or parameter modification.

The method formulates a reverse projected gradient descent (PGD) optimization in the input space to find a “corrective perturbation” that restores CMC. It is evaluated on fastMRI knee (Cor-PD) and brain (Ax-FLAIR) datasets, showing improvements over adversarial training and SMUG (smoothed unrolling). The method also generalizes to various PD-DL architectures and remains effective under blind and adaptive attacks.

### Strengths
The paper introduces a novel use of cyclic measurement consistency as a defensive objective, not just a training or calibration tool. This idea is well grounded in MRI physics and elegantly bridges signal reconstruction principles with adversarial robustness.

The approach works with any trained PD-DL model. This is a major advantage over retraining-based defenses like AT or SMUG. It can be combined with existing robust training for further gains.

Evaluated across multiple datasets, perturbation levels, and attack types. Also assessed on several architectures (MoDL, XPDNet, RIM, E2E-VarNet, Recurrent-VarNet), confirming generality.

Provides a theoretical analysis (Theorem 1) relating k-space perturbations and error propagation, lending physics-based interpretability.

### Weaknesses
The defense requires multiple forward passes through the reconstruction model (often dozens per iteration of reverse PGD). The runtime cost, though reported, may limit practical use in real-time MRI settings.

While strong within PD-DL MRI, the paper could discuss whether the approach generalizes to other imaging modalities or to non-physics-driven DL pipelines.

### Questions
na

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
5

### Summary
The paper proposes a training-free adversarial defense for physics-driven deep-learning (PD-DL) MRI reconstruction. Instead of retraining the model (as in Adversarial Training or SMUG), the authors introduce a cyclic measurement-consistency objective that perturbs the input slightly to restore reconstruction fidelity. A reverse-PGD procedure (Algorithm 1) iteratively corrects the attacked input by minimizing discrepancies between k-space projections of reconstructions from synthesized undersampling masks. The method is evaluated on fastMRI brain and knee datasets using MoDL and several other PD-DL architectures, and is shown to improve PSNR/SSIM under both standard and adaptive attacks.

### Strengths
- Thorough Evaluation: Multiple datasets (Cor-PD knee, Ax-FLAIR brain). Multiple architectures (MoDL, XPDNet, RIM, E2E-VarNet, Recurrent-VarNet). Multiple attack types: ℓ∞ image-domain, ℓ₂ k-space, sparse ℓ₀ (herringbone artifacts), and adaptive attacks.

- Physics-Driven Explanation: Theorem 1 offers an interpretable bound linking perturbations on acquired lines Ω to residuals on unacquired Ωᴄ, clarifying why cyclic inconsistency signals attacks.

- Strong Empirical Results: +1–3 dB PSNR gains over AT and SMUG; visual sharpness maintained.

### Weaknesses
1. While cyclic consistency is repurposed cleverly, much of the mathematical machinery (MoDL unrolling, PGD, consistency loss) builds directly on existing ideas; clarifying the conceptual leap beyond earlier “cycle-consistency” works (e.g., Zhang & Akçakaya 2024) would strengthen novelty.

2. Reverse-PGD plus multiple reconstructions is expensive (each iteration requires forward + inverse passes). Reported runtimes and wall-clock comparisons to AT/SMUG would help.

3. Paper should compare to concurrent defenses such as diffusion-based purification (Alkhouri et al., ICASSP 2024), as both are training-free physics-based methods.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- The paper considers the task of mitigating adversarial attacks in PD-DL MRI reconstruction. The attack methods are assumed to be gradient-based approaches that find small perturbations within an $l_p$​ ball, applied to k-space or image observations, which lead to large deviations in the reconstruction output according to some metric (e.g., MSE).
- The paper proposes a method based on the cyclic consistency property of well-trained PD-DL networks to mitigate adversarial perturbations applied to the input image. The key idea is that adversarial perturbations break the cyclic consistency assumption of the trained PD-DL network. Therefore, minimizing the inconsistency of simulated reconstructed images can help recover the unperturbed image.

### Strengths
- The paper is well written, with a clear motivation for the proposed framework.
- The method is training-free, though it requires per-sample optimization.
- Experiments demonstrate the effectiveness of the proposed method against several standard adversarial attacks in MRI reconstruction.

### Weaknesses
- The need to mitigate adversarial attack in the domain of MRI reconstruction does not convince me. 
- The proposed method relies on the assumption that the perturbation causes large changes in $\Omega^C$ and small changes in $\Omega$. However, quantitative justification for the bounds (lines 254–255) is missing. It is difficult to measure the effectiveness or generality of the proposed method across different scenarios.

### Questions
See weakness

### Soundness
3

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
This paper proposes a novel training-free defense method against adversarial attacks in physics-driven deep learning (PD-DL) MRI reconstruction. The approach leverages cyclic measurement consistency with synthesized under-sampling patterns to mitigate adversarial perturbations without retraining the model. The key idea is that adversarial attacks disrupt the consistency between reconstructions from actual and synthesized measurements, and this discrepancy can be minimized by optimizing a corrective perturbation within a small ball around the adversarial input. The authors demonstrate the effectiveness of their method across multiple datasets, attack types, PD-DL architectures, and realistic scenarios such as herringbone artifacts and adaptive attacks. The method also works in a blind setting without prior knowledge of attack strength or type.

### Strengths
(1) The method is well-motivated by the physics of MRI acquisition, particularly the role of data fidelity in unrolled networks and the propagation of perturbations to unacquired k-space locations.

(2) While cyclic consistency has been used in training and self-supervised learning, its adaptation for adversarial defense, especially in a training-free setting—is innovative.

### Weaknesses
(1) The adaptive attack results (Table 2) show that the iterative version of the defense remains effective, but the computational cost is glossed over. For T=100 unrolls, the iterative defense requires around 100 iterations—this is computationally prohibitive for real-time MRI reconstruction. The authors fail to analyze the trade-off between defense strength and computational efficiency. 

(2) The paper mainly focuses on worst-case adversarial perturbations but does not test the method against realistic, non-malicious perturbations such as motion artifacts. A robust defense should also improve resilience to these common disturbances.

(3) The comparison is limited to defense methods that require retraining (adversarial training, SMUG). This fails to situate the method within the broader literature on training-free adversarial defenses, such as: (a) Input purification via denoising (e.g., diffusion models, Autoencoders); (b) Randomized smoothing or input transformations.

### Questions
(1) Theorem 1 does not clearly explain why the proposed mitigation works. It bounds the output error but does not directly justify the cyclic consistency objective. A more insightful analysis should connect the optimization landscape of cyclic loss to the removal of adversarial perturbations.

(2) The method heavily depends on generating realistic k-space data from previous reconstructions. If the PD-DL network has inherent biases or artifacts in its reconstructions, won't these be amplified through the cyclic process?  How does the performance degrade when the base reconstruction network itself is suboptimal or when there's a domain shift between training and test data? The defense seems to assume a nearly perfect base model.

(3) The current comparative analysis overlooks several training-free defense methods, such as diffusion-based purification, JPEG compression, and randomized smoothing. To provide a more comprehensive and fair evaluation, it is necessary to include at least two such baselines  with a clear discussion of the computational and performance trade-offs.

(4) The paper claims that this approach is physically driven, but the entire defense pipeline is conducted through the forward propagation of the deep learning network. Is this still essentially a "black box against black box" approach? If an attacker uses a completely different physical model (such as a different coil configuration) to generate an attack, is your defense mechanism still effective?

(5) All experiments appear to be on 2D MRI slices. How would the method be scaled to 3D volumes or dynamic MRI acquisitions where computational costs are substantially higher?  Does the cyclic consistency concept extend naturally to higher dimensions?

### Soundness
2

### Presentation
2

### Contribution
2
