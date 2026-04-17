# Mutual Information Guided Diffusion Model for Partial Label Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
This paper proposes a novel paradigm for partial label learning (PLL) that integrates diffusion mechanisms with mutual information (MI) estimation to address the challenge of label disambiguation. In PLL, each training instance is typically associated with multiple candidate labels, among which only one is the ground truth. To simulate the process of label degradation, we introduce noise into the labels through forward diffusion to simulate candidate labels, and then perform reverse denoising to recover the true labels in a probabilistic sense. Unlike in conventional image generation tasks, we present the Mutual Information Guided Diffusion Model for Partial Label Learning (MDMPLL), which adapts diffusion models to weakly supervised learning and incorporates MI estimation to strengthen the consistency between denoised labels and data features, thereby improving label disambiguation. Furthermore, Dual-Path Attention Feature Fusion (DAFF) strategy is employed to enhance data representation, enabling more effective label disambiguation. Experimental results demonstrate that the proposed method significantly outperforms existing state-of-the-art PLL approaches on multiple datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the problem of partial-label learning. It leverages the reverse denoising process of diffusion models to disambiguate partial-label data. To encourage the model to retain task-relevant information during training and to enhance robustness under ambiguous labels, the paper introduces a feature–label mutual information maximization mechanism and incorporates it into the training objective as an additional supervisory signal.

### Strengths
1.	The paper has a strong motivation, applying the denoising process of diffusion models to label disambiguation in partial-label learning.
2.	The writing is well organized and adheres to academic conventions.

### Weaknesses
1.	The paper’s review of existing methods is overly abstract and insufficiently specific. For example: “On the one hand, label ambiguity and the complex relationships among candidate labels make it difficult for some methods to achieve stable disambiguation in a single pass, leading to limited robustness.” Nearly all PLL methods aim to address precisely this issue, so this statement does not accurately pinpoint the concrete shortcomings of prior work.
2.	Regarding “On the other hand, label disambiguation and feature learning are often decoupled, and lack mechanisms that effectively associate labels with the semantic content of images.” The intended claim seems to be that disambiguation and representation learning are decoupled and lack a mechanism to link them for stable, mutually reinforcing training. However, the proposed DAFF targets only representation learning and does not provide a joint mechanism. Moreover, many existing PLL approaches already consider disambiguation and representation learning jointly rather than as two separate stages.
3.	The paper does not explain why diffusion models are suitable for label disambiguation in PLL, which is crucial. Since the label information fed into the diffusion model is noisy, if the denoising process outputs a noisy label at each step, can it ultimately recover the true label? This justification is missing.

### Questions
1.	Are there any fundamental issues with applying diffusion models to partial-label learning? Why is $Y_0$ initialized via disambiguation from similar examples (KNN), and how can label information that is still noisy be used as input to a diffusion model? In image denoising, the diffusion model’s initial input is a clean, fully observed image. When applying diffusion to PLL, the input labels are noisy. Why does using KNN-based initialization improve the stability of disambiguation? What problems arise if it is not used?
2.	The proposed DAFF module appears to act only on representation learning and does not resolve the decoupling between disambiguation and representation learning found in prior methods.
3.	From the ablation results, the contributions of components D and F are modest, yet the overall gain over existing methods is large. This suggests the diffusion model itself contributes substantially. It would be better to show ablations across multiple datasets.
4.	After training, can the model use the diffusion denoising process to directly predict labels for unseen examples?
5.	In Eq. (15), what does $𝑉$ denote? What is the relationship between Eq. (9) and Eqs. (10), (15), and (16)?

### Soundness
2

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
3

### Summary
This paper proposes MDMPLL, a diffusion-based framework for partial label learning (PLL).
The authors treat label ambiguity as a forward diffusion process that gradually injects noise into labels and design a reverse denoising process for label disambiguation.
They further incorporate a Dual-Path Attention Feature Fusion (DAFF) module to combine shallow and deep features, and a Feature–Label Mutual Information (FLMI) objective to encourage alignment between features and labels.
Experimental results on several benchmark PLL datasets demonstrate that the proposed method performs competitively compared to prior approaches.

### Strengths
1.	The attempt to interpret label ambiguity via diffusion dynamics is novel and interesting.
	2.	The idea of integrating mutual information maximization into label disambiguation is conceptually meaningful.
	3.	The paper is generally well written and easy to follow.

### Weaknesses
1. Although Section 3 provides mathematical formulations, the approach largely mirrors standard DDPM procedures with only superficial adaptation to labels. The claimed connection between diffusion and mutual information is not derived or justified; no analysis is provided on how FLMI quantitatively improves disambiguation or generalization.
2. No ablation on the diffusion depth or comparison with simpler PLL baselines using consistency regularization is provided.
3. Visualization results (Fig. 4–5) are qualitative and do not convincingly demonstrate the claimed progressive disambiguation.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
This paper introduces MDMPLL, a diffusion-based framework for partial label learning (PLL) that models label ambiguity via forward noise addition to simulate candidate labels and reverse denoising for disambiguation. It incorporates Dual-Path Attention Feature Fusion (DAFF) to blend shallow and deep features as conditional inputs, and maximizes Feature-Label Mutual Information (FLMI) to align labels with semantics. A KNN-based initial disambiguation and dynamic label updates enhance stability.

### Strengths
1. nnovative adaptation of diffusion models to PLL, reframing disambiguation as probabilistic denoising with theoretical ties to variational bounds.

2. DAFF effectively fuses multi-level features via attention, improving representation quality; FLMI provides additional supervision for robustness.

3. Comprehensive ablations on components (DAFF, FLMI, diffusion steps) and hyperparameters; strong gains over baselines like PRODEN, CC, RC (e.g., +5-10% on CIFAR-100).

### Weaknesses
1. Role of mutual information estimation in disambiguation using diffusion model is not clearly explained; DAFF module seems disconnected from other parts of the method, and its relationship with other modules is unclear.

2. Mutual information guidance, as one of the core contributions, should not be excluded from Figure 1's overall framework.

3. In Algorithm 1 table, isn't the ultimate goal of the algorithm to get a model with strong generalization ability? Why return a clean label matrix?

### Questions
1. How does FLMI estimation interact with diffusion variance schedules? Could adaptive scheduling based on MI improve convergence?

2. Why not integrate DAFF into the noise prediction network end-to-end instead of as a separate fusion step?.

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
5

### Summary
The work presents a method for partial label learning or label disambiguation by invoking a diffusion process in the label space and progressively noising and denoising the labels to learn a robust denoiser. The method is guided by mutual information between labels and features derived from the shallow and deep sections of encoders, fused using Dual Path Attention Feature Fusion, eventually maximizing Feature Label Mutual Information using the InfoNCE loss.

### Strengths
1. The method shows decent empirical gains against baselines.
2. Incorporation of attention based features and MI guidance may be well informed in this problem and leads to demonstrable gains.

### Weaknesses
1. Some of the proofs are hand-wavy and/or already shown (Appendix D and E). The forward and reverse processes come from diffusion models literature. Appendix F outlines known results from EM literature. These proofs don't incorporate the specifics of the method proposed to synthesize something novel.

2. Comparisons against LRA-Diffusion [1] are missing which also uses pre-trained network features as guidance.

3. The authors could shed more light into the rationale behind using both pre-trained and un-trained encoders for deep and shallow feature sets respectively.

4. The motivation behind label and feature vector similarity may be mis-guided. Labels lie in a categorical simplex, whereas features are continuous embeddings. The authors could justify this choice in principle.

5. Training and inference runtime overheads for the methods aren't reported.


[1] Chen, Jian, et al. "Label-retrieval-augmented diffusion models for learning from noisy labels." Advances in Neural Information Processing Systems 36 (2023): 66499-66517.

### Questions
1. Is there a study that demonstrates the effects of the label filtering stage at the beginning?
2. Could the authors kindly delineate the differences between LRA-Diffusion and this work?

[1] Chen, Jian, et al. "Label-retrieval-augmented diffusion models for learning from noisy labels." Advances in Neural Information Processing Systems 36 (2023): 66499-66517.

### Soundness
2

### Presentation
2

### Contribution
2
