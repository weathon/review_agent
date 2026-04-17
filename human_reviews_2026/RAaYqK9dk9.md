# Diffuse and Disperse: Image Generation with Representation Regularization

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
The development of diffusion-based generative models over the past decade has largely proceeded independently of progress in representation learning. These diffusion models typically rely on regression-based objectives and generally lack explicit regularization. In this work, we propose $\textit{Dispersive Loss}$, a simple plug-and-play regularizer that effectively improves diffusion-based generative models. 
Our loss function encourages internal representations to disperse in the hidden space, analogous to contrastive self-supervised learning, with the key distinction that it requires no positive sample pairs and therefore does not interfere with the sampling process used for regression.
Compared to the recent method of representation alignment (REPA), our approach is self-contained and minimalist, requiring no pre-training, no additional parameters, and no external data. We evaluate Dispersive Loss on the ImageNet dataset across a range of models and report consistent improvements over widely used and strong baselines. We hope our work will help bridge the gap between generative modeling and representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces the Dispersive Loss as a plug-and-play regularization term for diffusion-based generative models. The Dispersive Loss encourages the hidden feature representations to disperse (or spread out) in the latent space, analogous to the non-alignment term of contrastive learning, but uniquely requires no positive sample pairs. The authors claim this enhances the quality of generated images and improves training efficiency. They report consistent improvements over strong baselines across various models on the ImageNet dataset.

### Strengths
1. Simplicity and Plug-and-Play Design: The proposed Dispersive Loss is highly appealing due to its self-contained and minimalist nature. It is a simple regularization term that can be added to the standard diffusion loss without requiring pre-training, external data, or additional visual encoders. This makes it easy to integrate into existing diffusion pipelines.

2. Demonstrated Training Acceleration: The paper shows an intriguing result of training acceleration without the need for external data or modules (unlike methods like REPA). While the gain is marginal, the fact that a simple internal regularization can accelerate convergence is a valuable finding.

### Weaknesses
## 1. Fundamental Gaps in Theoretical Justification and Representation Analysis

The paper's core weakness lies in the disconnect between its SSL-inspired motivation and its analysis.

- Lack of Deep Theoretical Insight: The paper does not provide sufficient theoretical insights into why the Dispersive Loss leads to improvement, nor does it explore what kind of high-level representations generative models fundamentally require. The mechanism of dispersion is asserted to be beneficial without a rigorous explanation of its effect on the learned noise-prediction function or the resulting generative manifold.

- Missing Representation Utility Analysis: Since the method is explicitly motivated by SSL, it should verify the "rich semantic features" from SSL methods. Analysis techniques like linear probing on the average-pooled features (at the intermediate or final layer) for tasks like classification or retrieval are essential to validate the core hypothesis that the regularization improves semantic content.

- Unclear Design Rationale: It is asserted that the alignment (positive pair) term of contrastive loss is detrimental to the generation task, leading to its exclusion, as in Table 2. This is a strong claim that is not clearly supported. Intuitively, the alignment term would, at worst, be redundant; the paper must theoretically or empirically demonstrate why it would be actively harmful.

## 2. Lack of Experimental Rigor and Insufficient Evidence

The experimental section suffers from issues of completeness, methodology, and evidence that undermine the paper's claims.

- Missing Empirical Evidence. The proposed method has not been thoroughly investigated. While the paper evaluates various settings of the regularizers, e.g., different models and depths, it does not sufficiently analyze why this method is effective. 

- Missing Scalability and High-Resolution Tests: The paper lacks experiments in high-resolution settings, such as ImageNet-512x512 or text-to-image generation. This omission makes it impossible to assess the method's scalability and effectiveness on datasets where modern generative models typically face their greatest architectural challenges.

## 3. Marginal Performance and Missing Training Dynamics Analysis

The reported performance gains are modest, and the paper fails to investigate the interaction between its components.

- Marginal Performance Gains: The overall performance gains achieved by Dispersive Loss are marginal. The resulting models neither consistently match SOTA generative performance (e.g., REPA) nor demonstrate effectiveness in any downstream context.

- Missing Integration with Faster Frameworks: While the paper notes its potential for acceleration, it is unclear whether the proposed framework can be combined with existing faster training frameworks like REPA. An experiment or intuition on this synergy is needed.

- Unanalyzed Training Dynamics: The paper completely lacks an in-depth analysis of the training dynamics involving the new loss term. The authors should show the evolution of the Dispersive Loss over the course of training to see if it plateaus or exhibits different behavior from the standard denoising loss. An analysis of the conflict or synergy between the denoising loss and the Dispersive Loss (e.g., by analyzing the direction of their respective gradients) is necessary to clarify the training process. Furthermore, an investigation into how the dispersive loss varies across diffusion timesteps is crucial, as its impact is intuitively expected to be different for high-noise versus low-noise regimes.

- Unfair Comparisons: The paper lacks comparison with fairer versions of SOTA models. For example, when comparing against REPA (800 epochs), the comparison should be normalized for total training time or resources consumed to provide a true measure of efficiency (Disperse loss uses >=1200 epochs in Table 6).

- Incomplete Results. The paper only compares with SiT and SiT-REPA (except for one-step generation) and uses FID as their single evaluation metric. Authors should conduct comprehensive comparisons with current state-of-the-art methods that share the same idea, such as Lighting-DiT [`1], DC-AE1.5 [2], DDT [3], etc. Besides, more evaluation metrics, such as Inception-Score, should be included in the paper.




**I recommend that the authors conduct a deeper investigation into the learned representations and strictly refine their experimental validation and theoretical justification during rebuttal period.**


[1] Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models

[2] DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space

[3] DDT: Decoupled Diffusion Transformer

### Questions
1. Could the authors provide theoretical or empirical evidence to clarify what specific kinds of high-level semantic features are required by image generation models, particularly diffusion models?

2. how does the proposed dispersive loss specifically facilitate the learning of such required semantic features?

3. Given that contrastive learning frameworks typically rely on heavy image augmentations to learn high-level semantics, how is the Dispersive Loss able to learn similar semantic information without relying on these image augmentations?

4. To support the main claim, shouldn't experiments be conducted to explicitly demonstrate the model's ability to learn rich semantic features, perhaps including evaluations on downstream tasks to provide valuable support?

5. Does a high representation norm truly indicate the capture of semantic information, and could the authors validate whether the model has learned semantic features, similar to the REPA method, by using linear probing?

6. Could the authors include feature similarity comparisons, such as Centered Kernel Alignment (CKA), with other feature extractors to better demonstrate the improved semantics?

7. To further validate the effectiveness of dispersive loss, would it be beneficial to test it on more complex generative tasks, such as text-to-image generation or text-to-video generation?

8. Since the method claims removing alignment term in line 231, shouldn't there be some investigation into why the alignment term deteriorates performance?

9. What happens when augmented images are used for positive pairs (i.e., should an ablation study compare the use of the same image versus an augmented image)?

10. How does down-weighting the alignment term in the contrastive loss affect the overall performance (i.e., how does performance change when moving from a standard contrastive loss toward the dispersive loss)?

11. Does this additional regularization lead to features that are usable for other discriminative tasks (even though this is not the main focus, it could demonstrate a strength of the learned features)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This authors try to address a key limitation of generative diffusion models with representation learning: diffusion models trained with regression-based denoising objectives often lack explicit regularization for the feature space, and therefore are limited in performance. To tackle this, the authors introduce a Dispersive Loss, which eliminates the need for explicit positive sample pairs, compared to the contrastive learning counterpart.

### Strengths
1. The authors aim to address a fundamental problem, namely that the generation task should not be left to stand alone with representation learning, and offer insightful perspectives.

2. The paper features a clear structure and coherent logic.

3. The proposed method does not rely on a pretrained encoder.

### Weaknesses
1. The authors used limited evaluation metrics. As they don't rely on a pretrained encoder and claim the importance of representation learning in the generative task, the authors should evaluate how the method performs with metrics like linear probing.

2. It's not clear why the major improvements were made in the case without CFG, while the performance with CFG only achieves very limited improvements. The authors should provide deeper analyses to explain this discrepancy and not rely only on FID (which is not in favor of the proposed method when evaluated with CFG).

3. It's not clear how the proposed method scales with the diversity and the size of datasets. Some empirical verification or theoretical motivation would be useful.

4. There is a slight redundancy in the mathematical derivation process of Section 3.2.

### Questions
Please see the weakness part above.

### Soundness
3

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
2

### Summary
The paper introduces a plug-in regularizer designed to enhance diffusion-based models by promoting the dispersion of internal representations in the hidden space. To demonstrate its effectiveness, the authors apply the proposed regularizer to two representative diffusion-based models, DiT (Peebles & Xie, 2023) and SiT (Ma et al., 2024). Experiments conducted on the ImageNet dataset show that the regularizer effectively improves model performance.

### Strengths
1. The proposed regularizer can be directly integrated into diffusion models with intermediate representations and requires little additional computational effort. 
2. It elegantly incorporates concepts from self-supervised learning into diffusion model training in a straightforward and theoretically sound manner. 
3. Experiments on a real-world image dataset demonstrate that adding the regularizer significantly improves performance, and the experimental results are comprehensive.

### Weaknesses
1. It would be helpful to clarify the scope of applicability. Can the proposed regularizer be applied to all diffusion models with intermediate representations? 
2. Qualitative comparisons between images generated with and without the proposed regularizer would make the improvements more intuitive and visually convincing. 
3. Although experiments explore different blocks, loss weights, and temperatures, it would be beneficial to provide systematic guidance or heuristics for selecting these hyperparameters.

### Questions
See "Weaknesses"

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Dispersive Loss, a simple yet effective plug-and-play regularizer designed to improve diffusion-based generative models by encouraging dispersion of internal representations in hidden space. Unlike prior work such as REPA [1], which requires pre-trained external encoders and additional parameters, the proposed method is self-contained, requiring no pre-training, no extra parameters, and no external data.

Conceptually, Dispersive Loss can be interpreted as a “contrastive loss without positive pairs.” It regularizes the hidden representations by penalizing excessive clustering and promoting diversity, inspired by the repulsive component of contrastive learning. The authors provide theoretical motivation, multiple instantiations (InfoNCE-, Hinge-, and Covariance-based), and efficient implementations requiring only a few lines of code.

The paper provides extensive empirical evaluation across several diffusion backbones, including SiT [2], DiT [3], and MeanFlow [4]. Experiments on ImageNet 256×256 consistently show significant FID improvements (up to ~11–13%) over strong baselines and even outperform contrastive variants that require two-view sampling. The method is also shown to generalize well to one-step generation and smaller datasets like CIFAR-10, demonstrating broad applicability.

[1] Yu, Sihyun, et al. "Representation alignment for generation: Training diffusion transformers is easier than you think." arXiv preprint arXiv:2410.06940 (2024).

[2] Ma, Nanye, et al. "Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[3] Peebles, William, and Saining Xie. "Scalable diffusion models with transformers." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

[4] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).

### Strengths
- The idea of removing positive pairs while retaining the repulsive regularization aspect is conceptually appealing and practically justified by diffusion models’ intrinsic alignment objective.
- Comprehensive experiments across multiple architectures (DiT, SiT, MeanFlow) and scales (S/B/L/XL) show consistent improvements in FID and Inception Scores.
- The improvement trend scales with model size, indicating the loss acts as an effective regularizer for large-capacity models prone to overfitting.
- Dispersive Loss outperforms all contrastive baselines even with careful tuning of noise schedules.
- The plug-and-play simplicity (no multi-view augmentation or external encoders) is convincingly demonstrated.
- The authors provide ablations for hyperparameters ($\lambda$, $\tau$), layer placement, and different loss variants (Table 2–4), showing robustness across configurations.
- Implementation details are transparent (Algorithm 1–2, Table 8).
- The inclusion of MeanFlow and CIFAR-10 experiments supports generality across diffusion and flow-matching paradigms.
- Figures (e.g., Fig. 1–4) clearly illustrate how Dispersive Loss integrates into existing architectures with negligible computational overhead.
- Comparisons with REPA (Table 6) highlight the system-level efficiency advantage (no 1.1B-parameter pre-trained model, no 142M external images).
- The paper adheres to reproducibility standards (code, README included) and presents results with careful quantitative analysis.
- The method provides a bridge between generative and representation learning, a frontier that is conceptually and practically valuable for the field.

### Weaknesses
- The method is motivated intuitively but lacks a formal analysis of why dispersion improves generation quality. A deeper information-theoretic or geometric argument (e.g., on latent coverage or mutual information bounds) would strengthen the theoretical grounding.
- While FID and Inception Scores are strong indicators, evaluation on semantic diversity, perceptual similarity, or representation quality (e.g., CLIP-based metrics) could better reveal what aspects of representation regularization improve.

### Questions
- Although CIFAR-10 is included, other generative domains (text-to-image or high-res synthesis) could further establish generality. It would be interesting to see whether Dispersive Loss also benefits conditional or multimodal diffusion models.
- While Figure 3 shows increased representation norms, further qualitative or visualization-based analyses (e.g., embedding t-SNEs) would make the mechanism more intuitive.
- Some missing works on representation learning using/within diffusion models should be included [5, 6, 7, 8, 9] in related works.

Overall, I think this is a good paper and I would be happy to raise the score if my comments are addressed properly.

[5] Wang, Yingheng, et al. "Infodiffusion: Representation learning using information maximizing diffusion models." International conference on machine learning. PMLR, 2023.

[6] Mittal, Sarthak, et al. "Diffusion based representation learning." International conference on machine learning. PMLR, 2023.

[7] Hudson, Drew A., et al. "Soda: Bottleneck diffusion models for representation learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[8] Zhang, Zijian, Zhou Zhao, and Zhijie Lin. "Unsupervised representation learning from pre-trained diffusion probabilistic models." Advances in neural information processing systems 35 (2022): 22117-22130.

[9] Yang, Xingyi, and Xinchao Wang. "Diffusion model as representation learner." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Soundness
4

### Presentation
4

### Contribution
3
