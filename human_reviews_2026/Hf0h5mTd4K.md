# MLE-UVAD: Minimal Latent Entropy Autoencoder for Fully Unsupervised Video Anomaly Detection

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
In this paper, we address the challenging problem of single-scene, fully unsupervised video anomaly detection (VAD), where raw videos containing both normal and abnormal events are used directly for training and testing without any labels. This differs sharply from prior work that either requires extensive labeling (fully or weakly supervised) or depends on normal-only videos (one-class classification), which are vulnerable to distribution shifts and contamination. We propose an entropy-guided autoencoder that detects anomalies through reconstruction error by reconstructing normal frames well while making anomalies reconstruct poorly. The key idea is to combine the standard reconstruction loss with a novel Minimal Latent Entropy (MLE) loss in the autoencoder. Reconstruction loss alone maps normal and abnormal inputs to distinct latent clusters due to their inherent differences, but also risks reconstructing anomalies too well to detect. Therefore, MLE addresses this by minimizing the entropy of latent embeddings, encouraging them to concentrate around high-density regions. Since normal frames dominate the raw video, sparse anomalous embeddings are pulled into the normal cluster, so the decoder emphasizes normal patterns and produces poor reconstructions for anomalies. This dual-loss design produces a clear reconstruction gap that enables effective anomaly detection. Extensive experiments on two widely used benchmarks and a challenging self-collected driving dataset demonstrate that our method achieves robust and superior performance over baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new approach to fully unsupervised VAD, called MLE-UVAD. The method uses a convolutional autoencoder and introduces a novel MLE loss in combination with reconstruction loss to address the problem of detecting anomalies in a single-scene video setting. The proposed system aims to detect abnormal events in videos without any labeled data, making it suitable for privacy-sensitive applications. Through experiments on several datasets, the authors show that their method outperforms various baseline models, including those with one-class classification and other unsupervised approaches.

### Strengths
1. The MLE loss is a novel approach to guide the autoencoder to collapse anomalous embeddings into the dominant normal cluster. This entropy-guided regularization improves the reconstruction gap, which is crucial for detecting anomalies in the absence of labels.
2. The proposed method operates in a fully unsupervised setting, meaning no labeled data is required during training, which is a significant advantage for real-world scenarios where labeled data is scarce or unavailable.
3. The method is tested on three diverse datasets, including real-world data and synthetic data, showing robust performance across various anomaly types and settings, outperforming both fully supervised and unsupervised baseline methods.

### Weaknesses
1. The paper uses the same unlabelled data for both training and testing, which leads to potential information leakage. It also tracks AUC with true labels, which can bias results.
2. The MLE loss has high computational complexity and relies on sensitive hyperparameters like σ/λ, but there's no efficiency data such as FPS or memory usage. 
3. The experiments use smaller datasets and report good AUC scores, even with a high anomaly ratio. It is recommended to test on larger and more diverse benchmark datasets, including more metrics such as precision, recall, and F1 score.

### Questions
1. How does the method perform when the assumption of dominant normal samples is violated in real-world scenarios? Could you provide more detailed results for cases with a higher proportion of anomalies?
2. Could you clarify how the hyperparameters, particularly σ and λ, are selected in practice? Are there any automated strategies or guidelines for setting these values, especially when no labeled data is available?

### Soundness
2

### Presentation
2

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
This paper proposes a novel conceptual understanding of unsupervised video anomaly detection in single-scene settings via frame reconstruction. In such task, the common practice in the field has been to train autoencoder-style models to reconstruct normal videos and use the difference between the reconstructed and the ground truth frames to identify anomalous frames. The assumption is that the model learns to reconstruct normal frames very well, while struggling to reconstruct anomalous frames due to the data imbalance in the training set (consisting of more normal videos than abnormal, or only normal videos). The paper argues that clustering together the latent representation of normal and abnormal frames alike via the proposed Minimal Latent Entropy (MLE) limits the model's capability to adapt its latent distribution to both the distributions of normal and abnormal frames, leading to much more evident differences in the reconstructed anomalous frames. The experiments presented on three standard benchmark datasets show that such approach is effective.

### Strengths
* The paper is written clearly and concisely. The figures are informative and the ablation studies extensive. 
* The conceptual design of the MLE loss is well thought and the mathematical formulation of MLE appears to be correctly formulated and presented.

### Weaknesses
* The experimental settings for the main table are inconsistent with common practice in literature. In lines 313-314 the authors write that the models are trained and evaluated only on a single anomaly type ("protest") of the Corridor dataset. This is not necessarily incorrect due to the focus on single-scene anomaly detection, the paper's contribution would benefit from similar experiments conducted on the rest of the anomaly types in the same scene (i.e. "chasing", "fighting", "suspicious object", etc...), either jointly or separately. Similar considerations hold for the experiments on the UBnormal dataset, of which the authors only use the "fire alarm" anomaly type (lines 318-319).

### Questions
* What is the combined effect of MLE with other reconstruction loss functions such as SSIM?

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
4

### Summary
The paper tackles single-scene, fully unsupervised video anomaly detection (VAD): training and detection are done directly on raw videos that contain a mix of normal and abnormal frames, with no labels at any stage. A convolutional autoencoder is trained with a dual loss: standard MSE reconstruction plus a Minimal Latent Entropy (MLE) regularizer (Rényi-2 entropy estimated via KDE). MLE encourages latent embeddings to concentrate around highdensity (normal) regions, effectively collapsing anomalous latents toward the normal cluster and widening the reconstruction gap between normal and abnormal inputs. Reconstruction quality is measured with the Pearson Correlation
Coefficient (PCC); anomalies are detected via a simple lower-tail threshold. On Donkeycar, Corridor, and UBnormal, the method reports high AUC and outperforms fully unsupervised baselines.

### Strengths
1. Simplicity with impact. Adding a single entropy regularizer (MLE) to a vanilla AE reliably amplifies the reconstruction gap, making anomalies easier to separate without complicated inference machinery.

2. Mechanistic clarity. The Rényi-2 + KDE formulation (with the pairwise Gaussian kernel view) and the kernel-gradient interpretation provide a convincing mechanism for why latent entropy minimization draws sparse anomaly embeddings toward the dominant normal manifold

### Weaknesses
1. Dependence on the “normal-majority” prior. The approach implicitly assumes normal frames dominate. Robustness under higher anomaly ratios (e.g., 50%) still needs fuller characterization despite some experiments.

2. Limited scope beyond single scene. The study targets single-scene deployments. This remains to be shown how well the trained model generalizes across multiple scenes/cameras without adaptation.

### Questions
1. When the majority prior breaks. If anomalies make up 30–60% of a video, how do the PCC distribution and the variance-aware threshold behave? Can the authors provide AUC curves versus anomaly ratio?

2. Beyond a single camera. What is the roadmap for transferring an AE+MLE trained on one scene to new cameras/scenes. Through camera-aware priors or domain adaptation, while preserving the same simple decision rule?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an entropy-guided autoencoder for unsupervised video anomaly detection. By combining the reconstruction loss with a minimal latent entropy loss, the proposed dual-loss mechanism distinguishes normal and abnormal frames through reconstruction gap in reconstruction errors, thereby enabling effective anomaly detection. While the idea is conceptually reasonable, the overall contribution and novelty appear to be incremental.

### Strengths
1. The proposed minimum latent entropy loss is conceptually grounded in information theory and is derived from Rényi entropy with kernel density estimation, providing a clear and interpretable formulation for latent-space regularization.

2. The method reports high AUC scores on three datasets, showing consistent improvements over several baseline approaches, including semi-supervised and fully unsupervised methods.

### Weaknesses
1. The proposed method offers limited novelty, as its main contribution lies in combining reconstruction loss with a minimum latent entropy loss.

2. The compared methods appear somewhat outdated. In particular, the fully unsupervised approaches listed in Table 1 are relatively old, with the most recent baselines dating back to 2022, and the evaluation metrics used for comparison are also rather limited.

3. The experiments are conducted only on three relatively small datasets. Evaluating on more challenging and diverse datasets could better demonstrate the robustness and generalization capability of the proposed MLE-UVAD.

### Questions
1. How does the MLE loss prevent all embeddings, including those of normal samples, from collapsing into a trivial cluster? Is there any mechanism designed to preserve intra-class variability among normal samples?

2. In Equation (4), the value of α is empirically set to 2. Is there any theoretical rationale supporting this choice? Additionally, have the authors investigated how varying α affects model performance?

3. The authors mention that, as normal frames dominate the raw video, sparse anomalous embeddings tend to be pulled into the normal cluster. How would the proposed approach handle scenarios where the ratio of normal to abnormal frames is balanced, or where anomalous frames occur more frequently?

### Soundness
2

### Presentation
3

### Contribution
2
