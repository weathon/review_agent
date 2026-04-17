# Multi-Level Stochastic Latent Noise Perturbation for Single Domain Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 2

## Abstract
Single domain generalization (SDG) is challenging because models trained on a single domain often suffer from out-of-distribution (OOD) shifts at inference time. Existing augmentation techniques often sacrifice semantic consistency for diversity or vice versa, and are largely confined to vision tasks. We propose a Stochastic Latent Noise Perturbation Module (SLNP) that automatically computes multiple MMD thresholds based on the source domain’s intra- and inter-class statistics, and then maximizes the sum of noise under these adaptive bounds. This unified objective generates diverse yet semantically faithful samples, applied independently of the downstream training loop—without requiring adversarial training or auxiliary loss terms. In addition, SLNP complements normalization methods, yielding synergistic improvements when the two are combined. Furthermore, our method is modality-agnostic and applicable to any distribution-based data. Experiments on image benchmark demonstrate that our approach integrates easily into existing pipelines and improves state-of-the-art SDG baselines, and additional results on speech data show its applicability beyond the vision domain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a fundamental yet challenging problem, namely single domain generalization (single-DG). Existing approaches based on data augmentation typically face a trade-off between semantic consistency and data diversity. To overcome this limitation, the authors propose a new strategy termed Stochastic Latent Noise Perturbation (SLNP), which enhances generalization by introducing stochastic perturbations in the latent space. SLNP is designed to be a pluggable module applicable to existing single-DG frameworks. Experiments conducted on both image and audio datasets demonstrate the effectiveness of the proposed method.

### Strengths
-	The paper is well organized and clearly written.

-	The proposed strategy is novel and interesting. It introduces distribution-based augmentations in a modality-agnostic manner, which is conceptually elegant. 

-	Experiments on both vision and speech datasets provide evidence for the generality of the proposed approach.

### Weaknesses
-	The experimental scope is relatively limited. The evaluation focuses mainly on two image datasets (PACS and CIFAR-10-C) in the vision domain and one audio dataset (TAU) in the audio domain. Since a key argument of the paper is the modality-agnostic property of SLNP, it would be more convincing to include experiments on additional modalities. For instance, MAD [1] evaluates performance on 1D, 2D, and 3D classification as well as 2D dense segmentation tasks, which provides a stronger demonstration of generality.

-	The analysis section is rather brief, containing only an ablation study. It would strengthen the paper to include more comprehensive analyses, such as hyperparameter sensitivity, visualization, or component-level generalization analysis [2].

-	The result discussion could provide deeper insights, particularly for cases where the vanilla performance is inferior to the baseline (e.g., Table 3). It would be helpful to explain potential reasons and implications for such cases. 


**References:**

[1] Modality-agnostic debiasing for single domain generalization. CVPR, 2023.

[2] High-frequency component helps explain the generalization of convolutional neural networks. CVPR, 2020.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
2

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
This paper introduces a Stochastic Latent Noise Perturbation module for single-domain generalization.

### Strengths
The proposed method perturbs latent representations using a flow-based model under multiple Maximum Mean Discrepancy constraints that are automatically computed from intra-class and inter-class statistics of the source domain. By doing so, it aims to generate diverse yet semantically consistent augmented samples without adversarial training or additional loss terms. The approach is designed to be classifier-independent and applicable across modalities, demonstrated on both vision (PACS, CIFAR-10-C) and speech (TAU Urban Acoustic Scenes) datasets. Experimental results show that SLNP improves robustness and complements normalization-based methods for better generalization to unseen domains.

### Weaknesses
1. **Lack of theoretical justification for stochastic latent perturbation's effectiveness in generalization**

It is well known that introducing perturbations can enhance a model’s generalizability. The paper does not clearly explain (1) the unique advantages of the proposed stochastic latent perturbation and (2) the underlying reasons why it can effectively enhance generalization. The effectiveness of the noise injection remains empirically observed rather than theoretically supported.

2. **The calculation method for $\epsilon$ risks circular logic.**
 
Both $\xi_{inter}$ and $\xi_{intra}$ rely on statistics of class distributions from the training data, which themselves can be affected by data scale and noise. If the source domain's inter-class distance is minimal or the intra-class variance is large, the calculation of $\epsilon_{max}$ may become unstable or fail.

3. **The design of modal adaptation mechanisms is not uniform.**

The paper claims that "the same framework can handle both vision and speech", but the speech part modifies the structure (1D flow, temporal splitting) and adjusts the hyperparameters (K=3 instead of 15), indicating that the method still requires manual parameter adjustment in different modalities and is not a truly unified algorithm.

4. **Limited experimental scale and improvement.**

(1) The improvement on PACS is marginal, only 0.06% (70.22% vs 70.16%), which is almost within the margin of error.

(2) On the speech task, using the method alone actually degrades performance (35.25% → 31.82%).

(3) This suggests the method's intrinsic contribution is limited and lacks statistical significance verification (e.g., t-tests, confidence intervals).

5. **Insufficient ablation studies**

The study only presents a sensitivity analysis for the parameter K, lacking ablations for the following critical factors:

(1) Whether multi-level MMD is truly necessary (single-level vs. multi-level).

(2) The impact of $\lambda_1$, $\lambda_2$, and $\alpha$ on performance.

(3) Whether the method remains effective if the MMD metric is replaced (e.g., with Wasserstein distance or cosine similarity).

6. **Imbalance between vision and speech Tasks.**

(1) While the vision component is relatively well-experimented (two datasets), the speech component relies on only one small dataset (TAU-ASC).

(2) The lack of cross-task validation (e.g., speech recognition, emotion classification) weakens the argument for the method's cross-modal generalization capability.

### Questions
Refer to the Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses Single-Domain Generalization (SDG) problem, where a model trained on a single domain is to generalize to unseen domains. Specifically, it proposes a new augmentation framework that trains a flow-based model under a multi-level Maximum Mean Discrepancy constraint and perturbs latent representations in a semantically consistent way. Experiments on both image and speech benchmarks demonstrate its superior generalization performance.

### Strengths
1. The idea of using a flow-based model and injecting noise in the latent space is interesting, as it expands diversity without risking semantic degradation.

2. The proposed module is independent of the main network and can serve as a plug-and-play data augmentation component, making it easily combinable with other methods.

### Weaknesses
1. The paper is not easy to follow. First, the organization is unclear: there is no summary of contributions, and the first two paragraphs of Section 3.4 read more like related work, yet they are placed under the Proposed Method section. Moreover, the paper lacks a framework figure and a brief introduction to RealNVP[1], which makes it difficult for readers unfamiliar with RealNVP to understand the method. Second, the logical consistency in some sentences is weak. For instance, in lines 265–267, the sentence first mentions “both methods” but then provides a reason that applies only to ADA, creating a logical disconnect between cause and subject. Additionally, in line 268, ASR-Norm is mentioned for the first time without any reference. Third, in line 205, Third, in line 205, the symbols $k$ and $x'$ in $E_x [k(x,x)]-E_(x,x') [k(x,x')]$ are coincided with those in Eq. (3), but the two expressions have different meanings, which may confuse readers.

2. In line 153, the paper states that “a learnable per-channel global scale is applied at the end of the flow.” However, the motivation for this design choice is not clearly explained. Could the authors provide more intuitive reasoning for this?

3. In line 271, the paper claims that the method “perturbs only domain-specific factors.” However, there is no disentanglement constraint or explicit mechanism to separate domain-specific and semantic factors in the latent space. Could the authors clarify how the model ensures that only domain-specific factors are perturbed?

4. For the image datasets, the authors evaluate their method on only two small datasets and do not include comparisons with recent works. Since the paper was submitted in September, it should consider more up-to-date baselines, such as SAC [2]. Could the authors conduct additional experiments on a larger benchmark, such as DomainNet [3], and compare their method with SAC to provide a more comprehensive evaluation?

5. For ablation study, the paper introduces two loss terms in Eq. (3); however, there is no ablation analysis to evaluate their individual contributions. Could the authors include additional experiments to analyze the effect of each loss component on model performance?

[1] Dinh, Laurent, et al. "Density estimation using Real NVP." International Conference on Learning Representations. 2017.

[2] Zhang, Zhen, et al. "Split-and-Combine: Enhancing Style Augmentation for Single Domain Generalization." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

[3] Peng, Xingchao, et al. "Moment matching for multi-source domain adaptation." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

### Questions
Please see the above weaknesses. If you can conduct additional experiments to further evaluate your method, I would be willing to raise my score.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a modality-agnostic augmentation module for single-domain generalization. Inputs are mapped through an invertible RealNVP-style flow, perturbed with stochastic latent noise, and reconstructed; perturbation magnitude is governed by K automatically derived MMD thresholds computed from intra/inter-class statistics, with a curriculum-like schedule. The module is pretrained outside the main training loop and then used as a plug-in for both images and speech; sensitivity to K is analyzed.

### Strengths
1. The method presents a general mechanism suitable for both speech and image data. Implementation details are clearly outlined for 1D and 2D flows.

2. Automatic thresholding is achieved using intra- and inter-class MMD. The analysis includes a sensitivity study on the parameter K.

### Weaknesses
1. The empirical evidence needs strengthening. A direct, quantitative comparison against strong SDG baselines across multiple datasets is missing. This comparison should report mean±std and effect sizes, as several claims currently rely on qualitative statements.

2. The RealNVP-style flows introduce additional training and inference costs. The practical impact on wall-clock time is not discussed, nor is it clear if cheaper perturbations could yield similar performance gains.

### Questions
1. Provide a summary table with mean±std results across multiple runs for PACS, Digits, CIFAR-10-C, and TAU-UAS. The table should compare (a) backbone only, (b) best normalization baseline, (c) SLNP, and (d) SLNP+norm.

2. Quantify the training and inference overhead from the flow module, both during pretraining and in the main loop. Report this in GPU hours and images/second.

3. Show examples of failure cases where a large K causes semantic drift. Also, indicate how frequently this occurs in practice.

### Soundness
2

### Presentation
2

### Contribution
2
