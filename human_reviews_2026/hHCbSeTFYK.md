# GeoDetect: Geometric Adversarial Detection for VLPs

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Vision-language pre-trained models (VLPs) are widely used in real-world applications. However, they remain vulnerable to adversarial attacks. Although adversarial detection methods have demonstrated success in single-modality settings (either vision or language), their effectiveness and reliability in multimodal models such as VLPs remain largely unexplored. In this work, we investigate the embedding spaces of VLPs and find that the image embedding space exhibits anisotropy. Our theoretical analysis shows that this anisotropic structure increases the separation between clean and adversarial examples (AEs) in the embedding space. Specifically, we demonstrate that AEs consistently exhibit greater expected distances to randomly sampled points than their clean counterparts, indicating that adversarial perturbations tend to push inputs out of manifold regions. Building on these insights, we propose GeoDetect, which leverages these off-manifold deviations to identify AEs. Through comprehensive evaluations, we show that our approach reliably detects adversarial attacks across various VLP architectures, including but not limited to CLIP, providing a robust and practical approach to improving the safety and reliability of these models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents GeoDetect, a theoretically grounded and effective framework for detecting adversarial examples in vision-language pre-trained models (VLPs). The authors uncover geometric anisotropy in VLP embeddings and leverage it for robust detection. Experiments across multiple datasets and models show excellent results (AUC≈1.0). The paper is well-motivated and clearly written, though more analysis on computational cost and real-world scalability would strengthen it.

### Strengths
1.	GeoDetect is grounded in a strong theoretical analysis of the geometric structure of vision-language model embeddings, demonstrating that adversarial examples tend to deviate from the data manifold. This provides clear interpretability and a robust theoretical basis for the proposed detection approach.
2.	The method is task-agnostic and does not rely on specific network architectures or attention mechanisms, making it applicable to a wide range of vision-language models and downstream tasks with strong generalization ability.
3.	GeoDetect operates directly on existing model embeddings without additional training or parameter tuning, offering a lightweight and easily deployable solution with low computational overhead.

### Weaknesses
1.	Although presented as efficient, the computation of geometric measures such as k-NN or KDE in high-dimensional embedding spaces can be resource-intensive, limiting scalability for large-scale or real-time applications.
2.	While GeoDetect supports multimodal data in theory, the experiments mainly focus on image perturbations, with limited evaluation on joint image–text adversarial attacks, leaving its robustness against complex cross-modal attacks less explored.
3.	Although the analysis of embedding-space anisotropy is carefully conducted, the findings are somewhat expected — prior work has already hinted that adversarial examples tend to move off the data manifold. Thus, while the theoretical framing is sound, the insight is not particularly surprising.

### Questions
Please see weakness

### Soundness
2

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
The authors propose a theoretical and practical framework for detecting adversarial examples in VLPs. The authors first show that VLP embedding spaces are anisotropic (i.e., that data representations cluster unevenly along certain dimensions). They prove that this anisotropy causes adversarial perturbations to push samples off the data manifold, resulting in increased geometric distances from clean examples. Building on this insight, the paper introduces GeoDetect, a lightweight, model-agnostic detection method that computes geometric scores (Local Intrinsic Dimensionality, k-NN distance, Mahalanobis distance, and Kernel Density Estimation) on image or multimodal embeddings to distinguish clean and adversarial inputs.

### Strengths
- The authors provide a theoretical foundation for adversarial detection in VLPs by linking embedding-space anisotropy to off-manifold deviations under adversarial perturbations. This insight unifies and generalizes previous intuition from unimodal settings.
- Based on this foundation, they introduced GeoDetect, a simple yet powerful and model-agnostic detection framework that uses classical geometric metrics to achieve near-perfect detection performance across diverse architectures and tasks without retraining or fine-tuning.
- They demonstrate strong empirical generalization and robustness, showing consistent detection accuracy across VLP families and downstream tasks. 
- They also demonstrate their method's robustness under adaptive attack scenarios.

### Weaknesses
- Their fundamental claim that adversarial examples lie off the manifold in latent space is actually not new, having been demonstrated in prior unimodal contexts. The authors have actually done well to discuss this, but as this is foundational to their defense design the novelty is limited (originality claim remains strong however).

- The anisotropy analysis, though central to the theoretical argument, remains largely descriptive. While measures such as $(I_1, I_2)$, and effective rank support anisotropy qualitatively, the paper never quantifies how anisotropy correlates with detection performance or adversarial vulnerability across architectures. This weakens the causal link between the theoretical foundation and the empirical effectiveness of GeoDetect.

- The method’s scalability and computational efficiency are underexplored. Geometric metrics such as k-NN, Mahalanobis, and KDE in practice scale poorly with embedding dimensionality and dataset size. The paper claims GeoDetect is “lightweight,” but offers no analysis of runtime complexity, memory footprint, or potential bottlenecks for real-time applications.

- The paper focuses primarily on standard, gradient-based PGD-style attacks. It lacks experiments on modern, semantically aligned or diffusion-based attacks that perturb both modalities coherently. Without this, it’s unclear if the geometric signal exploited by GeoDetect generalizes to more adaptive or distribution-preserving adversaries.

- While the authors have done most things right in the paper, they omitted analysis on failure cases and sensitivity to model variance. While results show near-perfect AUC scores, no discussion is offered on outliers, false positives, or potential degradation in low-sample or highly anisotropic conditions (e.g., small-scale VLPs or different pre-training datasets). This absence makes the method’s reliability across unseen regimes uncertain, especially given that embedding geometry can vary substantially between pre-training objectives.

### Questions
Check weaknesses above

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
GeoDetect is a geometry-based adversarial detection framework for vision–language models (VLPs).  
It leverages the anisotropy of multimodal embeddings and formulates a measurable *expected distance gap* to separate clean and adversarial samples through geometric scores (LID, k-NN, Mahalanobis, KDE).  
The approach is model- and task-agnostic, showing strong results on zero-shot classification and image-text retrieval, though it still relies on key assumptions and limited attack coverage.

### Strengths
* **Clear theoretical motivation.** The paper clearly connects VLP anisotropy and “off-manifold” adversarial behavior, and mathematically supports using geometric scores (LID, k-NN, Mahalanobis, KDE) as detection criteria.  
* **Model- and task-independent.**  The framework works without access to task heads or logits, transferring naturally between zero-shot classification and cross-modal retrieval.  
* **Broad experimental setup.** The authors evaluate single-modal, multimodal, and text-perturbation scenarios, and report robustness to several adaptive attacks and different attack backbones.

### Weaknesses
* **Dependence on anisotropy assumptions.** The method’s validity heavily relies on the anisotropy hypothesis. It remains uncertain whether this still holds under stronger attacks such as **M-Attack** or **SA-AET (Semantic-Aligned Adversarial Evolution Triangle)**
, which might deliberately align adversarial embeddings with the original manifold.  
* **Limited baselines.** Only one prior baseline from 2022 is compared, which is insufficient to demonstrate progress against more recent adversarial detection methods for VLPs.  
* **Reference set dependence.** The method requires a clean sample pool for computing geometric metrics, but the paper does not explain how to construct, maintain, or recalibrate this reference set under distribution shift or contamination.  
* **Layer-specific sensitivity.** Although the paper notes that LID is most effective at multimodal layers, it also shows that in certain ALBEF multimodal attacks, k-NN/KDE degrade substantially. A systematic analysis or automatic layer-selection strategy is missing.

### Questions
The chosen attacks and baselines are overly simplistic; the current experimental setup is insufficient to demonstrate the method's effectiveness under stronger or more realistic threats.

### Soundness
2

### Presentation
3

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
In this paper, the authors introduce GeoDetect, a lightweight and model-agnostic approach for detecting adversarial examples on vision-language pre-trained models (VLPs), such as CLIP. The authors observe that VLP embeddings are anisotropic, meaning that clean data embeddings lie on a data manifold, and as a consequence adversarial examples move off that manifold. GeoDetect uses simple geometric metrics—like k-NN, Mahalanobis distance, LID, and KDE—to quantify these off-manifold shifts and identify adversarially attacked examples. Experiments on several datasets and attacks show that GeoDetect achieves good adversarial detection performance across architectures and tasks, offering an efficient defense for multimodal models.

### Strengths
* The paper addresses a timely and important challenge, detecting adversarial attacks in vision-language pre-trained models, offering a practical contribution to improving the safety of multimodal AI models.

* The approach is simple, efficient, and broadly applicable, requiring no model fine-tuning or task-specific modifications.

* It is clearly written and easy to follow; the main idea of the proposed method is intuitive.

### Weaknesses
* one of the main weaknesses of the paper is the limited novelty of the proposed approach. Using embedding-space distances or density-based measures for adversarial detection has been explored in prior works (e.g., [0, 1, 2]), which makes the contribution primarily an adaptation of existing ideas to multimodal settings.

* the method does not account for adaptive attacks specifically designed to keep adversarial examples on the clean data manifold. Such attacks could significantly degrade detection performance, yet this scenario is not evaluated in the experiments.

* moreover, prior studies have already demonstrated that adversarial attacks incorporating distance or manifold regularization terms can successfully bypass similar geometric defenses (e.g., [4, 5]), highlighting the need for a more thorough robustness evaluation.

[1] Ma, Xingjun, et al. "Characterizing adversarial subspaces using local intrinsic dimensionality." arXiv preprint arXiv:1801.02613 (2018).

[2] Lee, Kimin, et al. "A simple unified framework for detecting out-of-distribution samples and adversarial attacks." Advances in neural information processing systems 31 (2018).

[3] Cohen, Gilad, Guillermo Sapiro, and Raja Giryes. "Detecting adversarial samples using influence functions and nearest neighbors." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[4] Athalye, Anish, Nicholas Carlini, and David Wagner. "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples." International conference on machine learning. PMLR, 2018.

[5] Bryniarski, Oliver, et al. "Evading adversarial example detection defenses with orthogonal projected gradient descent." arXiv preprint arXiv:2106.15023 (2021).

### Questions
* how does GeoDetect differ from prior geometric or distance-based adversarial detection methods ([1–3]) beyond applying them to multimodal settings?

* how would the method perform against adaptive attacks designed to keep adversarial examples on the clean data manifold?

* given that prior works ([4, 5]) show such attacks can bypass geometric defenses, how robust is GeoDetect under similar adaptive scenarios?

### Soundness
2

### Presentation
3

### Contribution
2
