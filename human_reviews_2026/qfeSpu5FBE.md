# Treating Neural Image Compression via Modular Adversarial Optimization: From Global Distortion to Local Artifacts

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
The rapid progress in neural image compression (NIC) led to the deployment of advanced codecs, such as JPEG AI, which significantly outperform conventional approaches. However, despite extensive research on the adversarial robustness of neural networks in various computer vision tasks, the vulnerability of NIC models to adversarial attacks remains underexplored. Moreover, the existing adversarial attacks on NIC are ineffective against modern codecs. In this paper, we introduce a novel adversarial attack targeting NIC models. Our approach is built upon two core stages: (1) optimization of global-local distortions, and (2) a selective masking strategy that enhances attack stealthiness. Experimental evaluations demonstrate that the proposed method outperforms prior attacks on both JPEG AI and other NIC models, achieving greater distortion on decoded images and lower perceptibility of adversarial images. We also provide a theoretical analysis and discuss the underlying reasons for the effectiveness of our attack, offering new insights into the security and robustness of learned image compression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel adversarial attack against neural image compression (NIC) models. Compared to existing attacks, the proposed attack optimizes both global and local distortions and leverages a selective frequency-domain mask to enhance attack stealthiness.

### Strengths
- The authors provided a thorough review of the literature on NIC and its attack.
- The proposed attack is effective against modern codecs such as JPEG AI.

### Weaknesses
- The paper's introduction fails to clearly motivate the significance of its local optimization module, which is a key claimed contribution. The authors are suggested to move the justification from Section 4.3 to the introduction to immediately clarify why this local attack is necessary and effective.

- The proof of the proposition implicitly assumes independence across all pixels in the image, which is likely to be violated for natural images where pixels are spatially correlated. Additionally, the implication of this result on practical attacks is unclear, as efficiency is not typically a primary bottleneck for adversarial attacks.

- The general idea of using the frequency domain to improve an attack's stealthiness is not novel. It's a well-established concept in adversarial attack research (e.g., [1,2]).

- The l_\infty distortion bound used in experiments is not specified. Furthermore, it lacks an ablation study on this budget, making it hard to assess how the attack's performance scales at different perturbation levels.

[1] Luo, Cheng, et al. "Frequency-driven imperceptible adversarial attack on semantic similarity." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
[2] Qin, Yao, et al. "Imperceptible, robust, and targeted adversarial examples for automatic speech recognition." International conference on machine learning. PMLR, 2019.

### Questions
- The performance gap between optimizing with and without normalized direction seems to be small. Would the signed gradient method achieve similar performance if optimized for more iterations?

- Does the generated adversarial perturbation transfer across different codecs?

- What is the perurabtion budget used in the experiments?

### Soundness
2

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
This paper proposes a modular adversarial attack framework tailored for neural image compression (NIC) models. The framework combines global distortion maximization, local artifact amplification, and frequency-domain masking techniques. Experimental results on widely used codecs, such as JPEG AI, demonstrate that the proposed method outperforms existing attacks across multiple compression models (including JPEG AI), producing higher distortion while maintaining imperceptibility.

### Strengths
1. The experiments are relatively systematic, involving a comparison of the representative learnable compression methods HiFiC and ELIC with the traditional JPEG method. These experiments confirm that the former provides better defense efficacy, offering insights for researchers aiming to improve adversarial robustness through data preprocessing techniques.
2. The discovery of the "multi-round compression" approach, though simple, provides valuable ideas for lightweight defense strategies.

### Weaknesses
1. The motivation behind the use of block-based SSIM difference and pooling to calculate local artifacts in the local optimization module is unclear. Additionally, the parameter PPP has not been subjected to hyperparameter analysis.
2. The design of the evaluation metrics in Section 5.3 is somewhat arbitrary. It lacks integration with subjective assessments or perceptual experiments to guide metric selection.
3. The definitions of the symbols in the formulas are unclear. For example, in Section 4.3, the symbols P, u, and v are not sufficiently explained, making them difficult to understand.
4. The paper lacks image examples of adversarial samples generated using different input methods. There is also an absence of comparative results at different bit rates and across different datasets.

### Questions
1. Is attack performance sensitive to block size P and pooling kernel size (e.g., 32)? Has hyperparameter analysis been conducted?
2. In the alternating use of symbolic gradients and normalized gradients, is the frequency of changes fixed or dynamically adjusted? Is there a design to adaptively adjust it based on the loss function?
3. Does your attack framework remain effective when applied to NIC models that have been trained with defense techniques?
4. What is the transferability of adversarial samples generated by your attack framework? For example, if adversarial samples are generated for one NIC model, can they successfully attack another NIC model? Have such experiments been conducted?
5. Why did you choose to use SSIM rather than other perceptual metrics such as LPIPS or DISTS?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an adversarial attack on neural image compression models that operates in two stages: (1) global distortion optimization using MSE, and (2) local artifact amplification using SSIM-based block-wise analysis. The method alternates between signed and normalized gradient directions and includes a frequency-domain masking module for improved stealthiness. Experiments show the attack outperforms prior methods (I-FGSM, FTDA, SRDA) on both traditional NIC models and the modern JPEG AI standard.

### Strengths
1. As NIC systems such as JPEG AI are being standardized and deployed, understanding their susceptibility to adversarial perturbations is of significant practical and academic importance. The focus on JPEG AI, a learned codec with real-world relevance—makes this work particularly impactful within the security and image compression communities.

2. The proposed two-stage optimization strategy is conceptually clear and practically effective.

3. The experimental evaluation is thorough and convincing. The authors conduct tests on multiple NIC architectures, including JPEG AI (BOP/HOP variants), and across several datasets such as Kodak, Cityscapes, and NIPS 2017.

### Weaknesses
1. While the proposed method is well engineered and empirically effective, its conceptual novelty is limited. The two-stage optimization (MSE → SSIM) and gradient alternation strategies are primarily adaptations or combinations of existing adversarial attack techniques rather than fundamentally new algorithmic ideas. Consequently, the contribution is more engineering-oriented.

2. The paper should include visualizations of the pre-encoding adversarial images (i.e., the perturbed inputs before compression) and their direct comparisons with the original images, since the most straightforward way to verify whether an attack is “imperceptible to the human eye” is to visually inspect the original versus the perturbed (pre-encode) images.

3. The current attack objective maximizes the distortion between the adversarial and original reconstructions (as described in Section Problem Formulation), rather than the distortion with respect to the original image. This seems somewhat strange to me, as it does not directly measure the degradation of visual quality relative to the ground truth.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
