# Real-Aware Residual Model Merging for Deepfake Detection

- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Deepfake generators evolve quickly, making exhaustive data collection and repeated retraining impractical. We argue that model merging is a natural fit for deepfake detection: unlike generic multi-task settings with disjoint labels, deepfake specialists share the same binary decision and differ in generator-specific artifacts. Empirically, we show that simple weight averaging preserves Real representations while attenuating Fake-specific cues.
Building upon these findings, we propose Real-aware Residual Model Merging (R$^2$M), a training-free parameter-space merging framework. R$^2$M estimates a shared Real component via a low-rank factorization of task vectors, decomposes each specialist into a Real-aligned part and a Fake residual, denoises residuals with layerwise rank truncation, and aggregates them with per-task norm matching to prevent any single generator from dominating.
A concise rationale explains why a simple head suffices: the Real component induces a common separation direction in feature space, while truncated residuals contribute only minor off-axis variations. Across in-distribution, cross-dataset, and unseen-dataset, R$^2$M outperforms joint training and other merging baselines. Importantly, R$^2$M is also composable: when a new forgery family appears, we fine-tune one specialist and re-merge, eliminating the need for retraining.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the first method that introduces model merging to the deepfake detection task. It analyzed the real similarity and fake distinction among the DF40 dataset, and then proposed R2M with a theoretical explanation. The results show its effectiveness in balancing different types of forgery methods in comparison with the specialist.

### Strengths
1.	The paper introduces model merging into deepfake detection for the first time.
2.	It provides a reasonable analysis of real–fake similarity on the DF40 dataset.
3.	The proposed R2M method shows balanced performance with theoretical support.

### Weaknesses
1.	Since DF40 is used, all real samples come from the same FF++ real, so it is unsurprising that they share the same distribution. However, this also limits the applicability of the conclusion. What if the training data contain reals from different distributions, e.g., FF++ and CDF?
2.	Regarding generalization: while it is understandable that model merging can improve in-domain performance, it remains unclear why merging would enhance cross-domain generalization. This point requires more explanation.
3.	In Table 1: The experimental results appear unusual — the All-in-one model fails completely on EFS detection, and even reverses predictions. Why would changing the real distribution cause previously learned forgery types to invert their detection behavior? This phenomenon warrants deeper analysis, particularly to clarify how model merging could lead to label inversion.
4.	More experiments on generalization are needed, for example, by evaluating on additional datasets (e.g., DFDCP, DFD) and comparing with more generalization-oriented methods (e.g., [1] and [2]).
5.	The writing requires further proofreading. For instance, DF40 is incorrectly cited — it is not (Qian et al., 2024) but rather (Yan et al., 2024).
6.	Within the same forgery category, the fake cases also require further analysis to validate the similarity findings in Fig. 2 — for example, within EFS. It would be helpful to train specialist models separately using DiT, SiT, and StyleGAN to support this analysis.

[1] Effort: Efficient orthogonal modeling for generalizable ai-generated image detection //ICML’25

[2] Can we leave deepfake data behind in training deepfake detector? //NIPS'24

### Questions
Please refer to the weaknesses.

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
4

### Summary
This paper proposes a training-free model merging approach that enhances the generalization ability of deepfake detectors by combining specialist parameters trained on different forgery types. Specifically, the authors argue that the “real” class direction can be well preserved during merging, while class-specific forgery artifacts are suppressed. To this end, they employ SVD decomposition to extract dominant directions and obtain a generalizable merged model.

Extensive experiments on the DF40 dataset demonstrate that the proposed method largely retains each expert’s detection capability. The authors further claim that their method exhibits high scalability.

### Strengths
1. This paper introduces a valuable setting: Merging detectors via a training-free approach to obtain generalized deepfake detectors at low cost.

2.  The authors provide rich theoretical justifications supporting the effectiveness of their proposed method.

3. The proposed approach is extensively evaluated under multiple protocols, showing competitive or superior AUC retention on seen tasks and improved generalization to unseen forgeries.

### Weaknesses
1. Although the authors have discussed using a single averaged linear head after merging is enough to get the results, empirical validation is missing. The authors are encouraged to compare the following heads: i) Averaged linear head; ii) Specialist-specific linear heads; iii) Re-tuned linear head (after merging).

2. Theoretical analysis relies on several assumptions: i) Local linearity; ii) Bounded remainder terms; and iii) Mild spectral gap. These assumptions may fail when specialist models differ substantially or when the networks exhibit high nonlinearity. Consequently, the validity of the core theoretical result (Proposition 1) hinges on whether such local properties still hold in large-scale models.

3. The proposed approach shows scalability with six models. However, as the diversity of deepfake generation methods continues to grow, the current scale remains modest. It is unclear how the method performs when the number of experts increases dramatically (dozens or hundreds), especially when specialists have uneven capabilities or have been trained on partially overlapping domains. Since the proposed method depends on low-rank SVD, the dominance of a shared “real” direction may diminish as task vectors diversify.

4. This paper argues for the asymmetry between shared “Real” and generator-specific “Fake” features. Nevertheless, it does not explore scenarios where the real data distribution changes or diversifies. For instance, when real samples come from new domains (e.g., different camera sensors or capture conditions). In such cases, the assumption of a single, stable “Real” direction may no longer hold.

5. Merging parameters across different specialists could introduce vulnerabilities such as trojan signatures or model poisoning. The authors  are suggested to perform a security analysis to R2M robustness in this regard.

### Questions
Please see Weaknesses

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
5

### Summary
This paper introduces **R²M**, a novel method for merging expert models in the domain of Deepfake detection. The core idea is to leverage SVD to decompose task vectors, treating shared "Real" features as a principal component and generator-specific "Fake" artifacts as residuals. This allows for a training-free and efficient merging process without a drop in performance. However, I have some concerns regarding the method's practical utility in real-world scenarios and its overall impact on advancing the field.

### Strengths
1. **Well Motivation:** The paper's motivation is clear and reasonable. Figure 2 provides a compelling and intuitive illustration of the core hypothesis: different specialist models share a common understanding of "Real" data while diverging in their representation of "Fake" data.
    
2. **Efficient and Effective Method:** The proposed R²M method is elegant in its simplicity. Being training-free, it offers a highly efficient way to combine specialists, and the experiments show that it does so with negligible performance degradation on seen tasks.

### Weaknesses
1. **Comparison of Similar SVD Technique Used for Detection is Missing:** Previous work like Effort (ICML'25) also proposed the similar  SVD-based approach for improved detection performance. More discussion and comparison for similar methods are needed.

2. **Clarity of Visualizations:** The readability of several experimental figures is a concern. In Figure 3 (heatmaps) and Figure 5 (dumbbell plots), it is difficult to discern the precise numerical gains or losses.
    
3. **Marginal Performance Gains:** The improvement of R²M over prior model merging techniques, like CART, appears to be marginal. 
    
4. **Concerns about Practical Utility and Scalability:** My primary concern lies with the practical application of the proposed merging strategy. The experiments partition domains based on broad forgery categories (e.g., FS, FR, EFS) rather than specific generator models (e.g., FSGAN, FaceSwapV2). This raises a crucial question: how should the framework handle the incremental addition of a new forgery method that belongs to an existing broad category? If a new FaceSwap variant emerges, would one retrain the entire FS specialist, or would a new, more granular merging strategy be required? The current setup does not seem to address this realistic scenario.
	
5. **Strange Performance of Comparison Methods:** For instance, the performance of *Specialist-FR* on EFS samples is so low (AUC=0.099). Can the author explain?

### Questions
1. **Fine-tuning Details:** For reproducibility and clarity, it would be beneficial to specify which parameters of the backbone were fine-tuned for the specialist models. For instance, was it only the final linear layer, or were other parts of the network also updated?
    
2. **Lack of Pipeline Diagram:** It would be better to provide a high-level pipeline diagram. A clear visual representation of the R²M process—from task vectors to the final merged model—would significantly aid reader comprehension.
    
3. **Ambiguity of "All-in-one" Baseline:** The training setup for the "All-in-one" baseline is unclear. Was it trained as a single binary (Real vs. Fake) classifier on all forgery data? To further strengthen the experimental results, I would suggest including a baseline where the "All-in-one" model is trained on a multi-class Fake detection task (i.e., classifying each specific forgery type), with the logits for all fake classes then aggregated for binary evaluation. This would provide a more robust comparison.
    
4. **Limited Impact on the Deepfake Detection Field:** While the authors position the work as a contribution to model merging, its tangible impact on advancing the core challenges in Deepfake detection itself seems limited. The problem of generalizing to unseen forgery families, a critical issue in the field, is not substantially improved by this merging approach.

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
This work analyzes the similarity between each specialist detector and the weight‑averaged model in deepfake detection, finding that real features are shared across detectors while fake features differ and can be complementary. Based on this observation, the paper proposes a Real‑aware Residual Model Merging strategy that enables rapid incorporation of new forgery families by preserving the shared real component and merging the residuals from different forgery specialists. Experiments across multiple datasets and protocols demonstrate the effectiveness of the proposed method.

### Strengths
1.The proposed method is straightforward and compelling; the analysis of real and fake feature similarities between each specialist and the weight‑averaged model is particularly insightful.
2.The R2M method is novel and effective: it updates models by retaining a shared real component while composing denoised, norm‑matched fake residuals to enable rapid adaptation to new forgeries.

### Weaknesses
1.Figure 2’s analysis relies on features from older forgery datasets (e.g., DF40), where real and fake images are relatively easy to separate; how would the analysis and the proposed method perform when forgeries are highly realistic and real and fake feature distributions are not well separated?

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
