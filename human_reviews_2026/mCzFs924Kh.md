# Revisiting Out-of-Distribution Detection: Angular Separation Learning as a Powerful and Simple Baseline

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Out-of-Distribution (OOD) detection is a critical safety requirement for deploying deep neural networks in open-world environments. While recent advances increasingly rely on more computationally intensive training methods involving synthetic outliers, contrastive objectives, or specialized loss functions, their gains often come with substantial computational overhead and implementation complexity.
In this work, we revisit the fundamentals of OOD detection and uncover a key flaw in common distance-based detectors: sensitivity to feature magnitude. We show that low-norm OOD samples can appear closer to in-distribution (ID) class centroids than actual ID samples, evading detection.
To address this, we introduce **Angular Separation Learning (ASL)**, a simple and highly effective strategy that applies $\ell_2$-normalization to features before the final classification layer. This modification compels the network to optimize for angular separation, achieving robust feature learning without additional regularization mechanisms, synthetic samples, or costly negative mining.
Through extensive experiments on diverse benchmarks, we demonstrate that ASL not only matches but often surpasses state-of-the-art methods, especially in challenging near-OOD scenarios, while maintaining training efficiency. Our results indicate that a minimalist rethink of standard training can achieve superior OOD performance, prompting a re-evaluation of the complexity-to-performance trade-off in OOD detection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
## Summary
The paper proposes Angular Separation Learning (ASL) as a simple, training-time technique for OOD detection: normalize feature representations before the classification head to emphasize angular decisions. The authors hypothesize that feature-norm is an overlooked factor; they analyze Mahalanobis distance, highlight failure cases with low-norm OOD features and show strong performance on multiple benchmarks. They also compare with post-hoc normalization methods (e.g., Mahalanobis++) underperform ASL.

### Strengths
- **Simplicity.** Very simple algorithm and presented in a clear way.
- **Low overhead, strong results.** A lightweight add-on achieving strong performance.
- **Broad evaluation.** Covers many cases; Layerwise analysis done on Figure 5 is particularly a nice add.
- **Fair comparisons.** Side-by-side with state-of-the-art post-hoc scoring methods, highlighting gains attributable to the training scheme.

### Weaknesses
- **Overstated claims.**  
  - Calling other training-based methods “complex” (line 15) / “exotic” (line 24) is too strong. Please tone down language unless quantitatively/qualitatively supported.  
  - Line 111: [1], [2], [3] are works that already studied Mahalanobis and its corrections. Calling it “under-studied” is not accurate.  
  - Line 407: Real-world practicality is an overstatement. One would not use a model with FPR95 ~ 20% on safety-critical applications.

- **Figure 1 clarity.**  
  Low-norm OOD representations are closer to ID is a strong observation, but I find Figure 1 unclear: how exactly computed, how thresholds selected, which backbone/dataset used, and whether repeated across datasets/backbones.

- **Motivation vs. cosine head.**  
  If the goal is angular learning, a cosine-based classifier (normalized features and normalized weights) is a straightforward alternative. Authors mention applying weight decay to bound weight norms, but do not justify not normalizing weights. Please add an ablation to evaluate:
  - Whether norm cancellation and a truly cosine classifier improve OOD detection.
  - Whether cosine model feature representations align with Section 3.2’s intuition.
  - If cosine performs worse, provide reasoning and supporting experiments if it does not agree with the intuition provided.

- **Low-norm failure analysis.**  
  Beyond Fig. 1, selecting the 95% ID validation threshold, collecting false-positive OODs, and plotting their norm distribution alongside ID validation norms would be an easy validation for the main argument that we need to solve the low norm OOD sample symptom.

- **Related work (post-hoc).**  
  It is a rather crowded line of research. Therefore, the coverage lags the literature. Please consider [4] for strong and simple baselines and extensive evaluation and also [5], [6], [7].

- **Norm as signal vs. hypothesis.**  
  Line 150 cites a work using norm as a discriminative signal, which seems at odds with the main hypothesis. Either show [8] does not apply consistently or cite works aligned with your stance (e.g., [2], [5], [9]).

- **Intuition around Eq. (1).**  
  The discussion around line 198 is confusing. In ideal CE, ID feature and class mean coincide (therefore, distance becomes 0). Interpreting the equation "holding everything else constant" may mislead; many configurations allow high class separation and tight clusters with large norms. The issue seems more about ambiguous CE feature assignments to ID/OOD than "MDS sensitivity to norm" per se; consider analysis grounded in training dynamics (possibly with synthetic settings).

- **Training efficiency.**  
  Include confidence intervals from repeated runs after warmup. The table is concerning as there’s no clear reason LogitNorm should be slower than ASL. Control all hyperparameters (especially batch size) if reporting wall-time.

### Questions
- Figure 3 is aesthetically nice but uses large space; consider rearranging to save space.
- In Figure 1, what metric defines "nearest class mean"? L2 or cosine?
- In imbalanced classes, weight/feature norms may contain priors probabilities if we interpret CE trained models as posterior predictors (~ $p(y) \cdot p(x|y)$). How does normalization behave in this setting?
- Table 3 (CIFAR-100): PALM has the lowest FPR, yet both ASL and PALM are bolded.
- Can ASL be straightforwardly combined with activation shaping (e.g., ReAct, ASH)? It would be interesting to see interactions.
- Since training/optimization is involved, please report confidence intervals with repeated experiments over different random seeds. Any reason this was not done?


## References
[1] Ren et al., 2021. *A simple fix to Mahalanobis distance for improving near-OOD detection*. arXiv:2106.09022.  
[2] Mueller & Hein, 2025. *Mahalanobis++: Improving OOD Detection via Feature Normalization*. arXiv:2505.18032.  
[3] Shi et al., 2013. *Improved relative-transformation PCA based on Mahalanobis distance*. Acta Automatica Sinica.  
[4] Bitterwolf, Mueller, Hein, 2023. *In or out? Fixing ImageNet OOD detection evaluation*. arXiv:2306.00826.  
[5] Demirel, Fumero, Locatello, 2024. *Out-of-Distribution Detection with Relative Angles*. arXiv:2410.04525.  
[6] Liu & Qin, 2025. *Detecting OOD through the lens of neural collapse*. CVPR.  
[7] Ammar et al., 2023. *NECO: Neural collapse based OOD detection*. arXiv:2310.06823.  
[8] Zhang & Xiang, 2023. *Decoupling maxlogit for OOD detection*. CVPR.  
[9] Sun et al., 2022. *OOD detection with deep nearest neighbors*. ICML.

### Soundness
3

### Presentation
3

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
This paper proposes Angular Separation Learning (ASL), a simple modification to standard classification training for Out-of-Distribution (OOD) detection. By applying normalization to feature vectors before the final linear layer, the method enforces an angular decision geometry that alleviates the well-known sensitivity of distance-based OOD detectors (in particular of Mahalanobis detection score) to feature magnitude. The authors demonstrate that this minimalist change yields state-of-the-art or superior performance on several OOD benchmarks, especially for near-OOD cases. The paper argues that small architectural choices can achieve competitive OOD robustness without added complexity.

### Strengths
1.	The method is extremely lightweight and well-motivated by a geometric analysis.
2.	ASL consistently matches or outperforms more complex OOD detection approaches, particularly in the challenging near-OOD regime.
3.	The work prompts a healthy reconsideration of whether the field’s increasing methodological complexity is always justified.
4.	The method aligns the loss used during training with the score employed for InD/OoD separation.

### Weaknesses
1.	The idea of feature normalization and angular margin learning is not new (similar principles exist in SphereFace for example). The main novelty lies in reinterpreting and applying them to OOD detection.
2.	The method shows clear improvements in near-OOD detection; however, a more thorough analysis is needed to assess its effectiveness on far-OOD scenarios and its impact on overall classification accuracy.
3.	The impact of normalization on confidence calibration and threshold is not discussed.

### Questions
1.	Angular separation learning has been previously explored in the literature, notably in *SphereFace: Deep Hypersphere Embedding for Face Recognition* (Liu et al., CVPR 2017). Although SphereFace was not designed for OOD detection, its underlying rationale is closely related, as it also enforces angular-based feature learning on a hypersphere and discusses the Mahalanobis distance for classification. SphereFace should therefore be cited and discussed in relation to the proposed method. In particular, the loss functions of ASL and SphereFace should be explicitly compared and analyzed, as both promote hyperspherical feature learning but differ in how they enforce angular separation (standard normalized cross-entropy vs. multiplicative angular-margin loss).

2.	The mechanics of MDS failure are well-motivated for near-OOD samples. However, losing access to feature norm information may negatively impact both classification accuracy and far-OOD detection. While the accuracies of models trained on CIFAR-10 and CIFAR-100 are reported in the appendix, they should also be clearly mentioned alongside all experimental results, especially those presented in the main body of the paper. Regarding far-OoD detection, the results presented in Fig. 4 show a strong instability with respect to parameter $\lambda$ (on individual datasets) meaning that the method could exhibit poor performance in some particular situations. Could you elaborate on the strategy to choose the value of $\lambda$ that should not be guided by experimental results on particular datasets ?

3.	Could you provide an analysis of the features distribution to visually verify the claimed hyperspherical geometry and the relative positioning of InD and OOD samples?

4.	The estimation of $\mu_c$ and $\Sigma$ is a key factor in the effectiveness of MDS. Could you clarify how these estimates were computed (e.g., number of samples, and whether training, validation, or test data were used)?

5.	Neural Collapse (mentionned in the paper) is typically observed for in-distribution features, where class features collapse toward their centroids on a hypersphere. Given that Angular Separation Learning enforces $\ell_2$-normalization, did you observe a similar collapse behavior around class centroids for in-distribution samples, and how does this interact with the positioning of out-of-distribution features? 

Angular Separation Learning (ASL) is a lightweight and well-motivated approach that demonstrates strong empirical performance, particularly for near-OOD detection. While the method is effective and clearly presented, its novelty is limited, with prior work (e.g., SphereFace) exploring similar angular feature normalization. The analysis of far-OOD performance, feature-space visualization, and hyperparameter sensitivity is incomplete. Overall, the paper provides a valuable incremental contribution to OOD detection, but it primarily reinforces existing intuitions rather than introducing a fundamentally new methodology.

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
3

### Summary
This paper revisits the foundation of out-of-distribution (OOD) detection and identifies a critical weakness of Mahalanobis-based methods — their sensitivity to feature norms. The authors propose Angular Separation Learning (ASL), a minimalist yet effective training strategy that applies $\ell_2$-normalization to features before the final classification layer. This simple modification encourages angular discrimination between classes, effectively mitigating the failure mode of distance-based detectors.

### Strengths
1. The proposed Angular Separation Learning (ASL) involves only a minimal modification—$\ell_2$ normalization of features before the cross-entropy loss—yet consistently surpasses much more complex training schemes (contrastive, synthetic outlier generation, or regularization-heavy methods). This reinforces the key message that complexity is not always correlated with OOD robustness, positioning ASL as a strong and practical new baseline.

2. The paper provides a unifying perspective connecting normalization, angular margin optimization, and contrastive representation learning. Proposition 1 formally shows that ASL implicitly performs prototype-based contrastive learning in angular space, without explicit pair mining or temperature tuning. This theoretical lens could inspire a rethinking of numerous normalization and contrastive frameworks.

3. The experiments are extensive, covering both convolutional and transformer backbones, and include near-/far-OOD as well as covariate shift scenarios.

### Weaknesses
1. The novelty of the proposed method appears somewhat limited, as normalization-based mechanisms for OOD robustness have already been explored in prior works such as LogitNorm (Wei et al., 2022), T2FNorm (Regmi et al., 2024a), and MD++ (Müller & Hein, 2025). It would strengthen the paper if the authors could further highlight the unique aspects and contributions of their approach compared to these existing methods.

2. Proposition 1 provides a qualitative statement about angular separation, but its proof sketch is not rigorous and lacks an explicit link to OOD generalization.

3. In the proposed ASL framework, features are projected onto the angular space via $\ell_2$-normalization before classification. Could the authors further clarify why discarding the feature magnitude and relying solely on angular information is beneficial for OOD detection?

### Questions
Please carefully check the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the out-of-distribution (OOD) detection problem. It proposes that recent work rely on complex training paradigms, which significantly limit the computational efficiency. Moreover, it discover that the existing methods neglect the influence of feature magnitude, which may lead to incorrect detection of OOD samples. To address this problem, it proposes a new method called ASL, which encourages robust feature learning by adding a L2-normalization before the classifier. The proposed method is evaluated on multiple datasets, where the performance achieves SOTA in most cases.

### Strengths
- The motivation of this paper is reasonable. It reveals the reason behind the failure of existing distance-based detection methods by empirical studies.
- The proposed method is simple while effective. This will improve the genealizability of the proposed method.
- The experimental results are strong. Also, the authors have conducted extensive ablation studies and additional analysis to make the proposed method more convincing.
- Theoretical analysis is provided to improve the solidness.

### Weaknesses
- The proposed method combines a feature normalization and a linear classifier. What about directly modifying the linear classifier into a cosine classifier or some other distance-based classifier? Using cosine classifier for OOD detection is already studied by previous works, so what is the difference between previous work and this study?
- The training efficiency is an important strength of this paper, which, however, is only mentioned in the section of experiments. Is it possible to add some explanations or analyses in the section of method to highlight this advantage?
- The font sizes of some figures are inharmonious (e.g. Figure 15). It is better to improve the fonts.

[1] Hyperparameter-free out-of-distribution detection using cosine similarity.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
