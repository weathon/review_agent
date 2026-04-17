# Riemannian High-Order Pooling for Brain Foundation Models

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
Electroencephalography (EEG) is a noninvasive technique for measuring brain electrical activity that supports a wide range of brain-computer interaction applications. Motivated by the breakthroughs of Large Language Models (LLMs), recent efforts have begun to explore Large EEG foundation Models trained on broad unlabeled corpora. However, most advances focus on improving the backbone while neglecting the classification head. Existing models often rely on a single class token, underutilizing the spatiotemporal structure and second-order statistics that are crucial for EEG decoding. We propose Riemannian High Order Pooling (RHOP), a plug-and-play module that injects principled Riemannian statistics into the classifier. RHOP maps each token to a quotient Gaussian jointly encoding mean and second-order information, yielding scale-invariant descriptors. Tokens are then aggregated by estimating a Riemannian Gaussian on the SPD manifold, where the Fréchet mean and covariance are embedded into an SPD descriptor. The resulting normalized vector is fused with the class token for prediction. RHOP is backbone-agnostic and integrates with modern EEG foundation models, \textit{e.g.}, BIOT and LaBraM. Across diverse EEG benchmarks, it improves accuracy and efficiency under full fine-tuning, linear probing, and from-scratch training settings. The code is publicly available at github.com/ChenHu-ML/RHOP.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a plug-and-play classification head module named Riemannian High-Order Pooling , specifically designed for Electroencephalography foundation models. The core thesis is that while state-of-the-art EEG foundation models like BIOT and LaBraM have made significant strides in their feature extraction backbones, their classification stages are underdeveloped. These models typically rely on simple pooling mechanisms  or a single class embedding  for prediction, a process that discards second-order statistics and spatiotemporal structure crucial for EEG decoding. RHOP aims to address this deficiency by injecting principled Riemannian statistics and geometric priors into the classifier, thereby making fuller use of the rich features extracted by the foundation model.

### Strengths
S1: Significant Problem & Principled Methodology
The paper addresses a well-motivated and significant problem. The proposed solution, RHOP, is not an ad-hoc engineering trick but is built on a solid theoretical foundation.

First, Quotient-Gaussian Embedding  achieves scale invariance. The use of quotient spaces to achieve invariance is a classic approach in geometric machine learning. Applying it to normalize covariance matrices into correlation matrices elegantly addresses the challenge of amplitude variations across EEG segments, a more principled solution than simple normalization techniques.   

Second, Riemannian geometry preserves the intrinsic data structure. The long-standing success of Riemannian methods in EEG analysis demonstrates the power of operating on the SPD manifold. By modeling the set of tokens as a Riemannian Gaussian distribution, the method captures not only the central tendency of the features in a geometrically faithful way  but also their dispersion.   

Finally, the iSICE layer serves as a feature refiner. Standard covariance captures all pairwise correlations, which can be misleading due to confounding factors. The inverse covariance  captures partial correlations. By using a sparse inverse covariance, iSICE explicitly models and emphasizes the most direct and robust relationships, which is highly beneficial for dealing with noisy EEG data.   

These three components are not independent but form a logical, progressive hierarchy of statistical refinement. This structure is a core strength of the method, as it demonstrates a deep understanding of the problem by tackling different statistical challenges at different stages of the pipeline.

S2: Comprehensive and Rigorous Experimental Validation
The experimental evaluation is a major highlight of this paper. The authors have gone to great lengths to demonstrate the effectiveness and generalizability of RHOP.

Diverse Benchmarks: Evaluation on four different and widely-used EEG tasks—abnormality detection on TUAB, event classification on TUEV, motor imagery on BCIC2B, and event-related potentials on PhysioP300—shows that the method is not overfitted to a single problem type. These are standard and challenging benchmarks.   

Multiple Backbones: Testing on both BIOT and LaBraM, two state-of-the-art EEG foundation models, proves that RHOP is, as claimed, a truly "backbone-agnostic" module.   

Multiple Training Paradigms: The evaluation across full fine-tuning, linear probing, and training from scratch is comprehensive. The strong performance in the "from scratch" and "linear probing" settings is particularly impressive, suggesting that RHOP can both impose useful structure early in training and effectively extract discriminative information from fixed representations.   

Strong Baseline Comparisons: The comparison against multiple representative global covariance pooling heads is fair and pits RHOP against its most direct competitors.   

S3: Clarity of Exposition & Reproducibility
The paper is well-written, well-structured, and easy to follow. The introduction clearly lays out the motivation. The methods section builds logically from preliminaries to the final framework. Figure 1 provides an excellent visual overview of the entire pipeline. The appendix includes detailed hyperparameters , the Karcher flow algorithm , and a proof for the key embedding theorem , which greatly enhances the work's reproducibility. The authors also promise to release the code upon acceptance, which is in line with academic best practices.

### Weaknesses
While the overall methodology is strong, several key design choices lack sufficient justification or analysis.

- Approximation of the Fréchet Mean: The paper states that for efficiency, the Fréchet Mean is computed using only a single iteration of the Karcher flow. While the desire for efficiency is understandable in a deep learning context, this is a significant approximation. The Fréchet Mean is defined as the minimizer of the variance function, and a single step from an arithmetic mean initialization does not guarantee convergence or proximity to the optimum.

- Sensitivity to Hyperparameters ($k, k'$): The embedding dimensions $k$ and $k'$ are crucial hyperparameters.1 Table 6 shows that performance varies with their setting, but no intuition or methodology is provided on how to choose them or set them for a new task. Were they tuned on a validation set? Is there a theoretical reason why $k=3, k'=3$ is optimal on TUAB?

- Choice of Riemannian-Gaussian Embedding (Eq. 9): The specific method for embedding the Riemannian-Gaussian pair $(P^m, P^c)$ into a single SPD matrix $G$ is drawn from the work of Nguyen. While citing prior work is appropriate, a sentence or two explaining why this particular block matrix form involving the Cholesky decomposition of the covariance is a good choice would be highly beneficial to the reader. What properties does this embedding preserve?

### Questions
See Weakness

### Soundness
3

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
4

### Summary
This paper presents an exploratory attempt to integrate recent large brain foundation models with second-order pooling techniques originally developed in computer vision. The authors claim that their proposed neural layer captures second-order information and global spatiotemporal dependencies from EEGs, and evaluate their approaches on several EEG-BCI scenarios.

### Strengths
Overall, the work is conceptually interesting. It aims to transfer second-order modeling paradigms from vision to EEG-based brain signal analysis. The motivation is creative, and such cross-domain adaptation deserves attention.

### Weaknesses
1. The paper frequently cites SPDNet approaches where SPD matrices are computed as spatial covariance matrices derived from multichannel EEG time series. However, the current work constructs SPD representations from tokens derived from the backbone model on temporal segment EEGs, i.e., covariances across temporal-segment tokens rather than across spatial channels. This formulation deviates from existing literature on Riemannian EEG analysis, where spatial covariance carries physiologically meaningful information about inter-channel connectivity. Since reaction time and task onset vary substantially across trials, temporal alignment is inconsistent, making temporal-segment covariance unlikely to encode stable or discriminative features. The marginal performance gains across datasets and backbones (Tables 1–4) further support this concern. Without a clear justification, the connection between the proposed SPD representation and neural dynamics remains weak.

2. The experimental setup is under-specified. It is unclear whether the experiments are subject-independent or subject-dependent, how many cross-validation folds were used, and what preprocessing pipeline (e.g., filtering, artifact rejection, normalization) was applied. These details are critical, as EEG performance is highly sensitive to preprocessing and evaluation protocols. Moreover, the description of the BCIC2B dataset is ambiguous; it is not stated which BCI Competition dataset (e.g., IV-2a, II-2b, etc.) was used. Given that such datasets are relatively small, large-scale models like LaBraM-Base are highly prone to overfitting. Without explicit regularization or validation strategies, the reported performance may not generalize.

### Questions
1. What is the physiological or theoretical motivation for the covariance of the temporal-segment tokens to carry class-discriminative information? Has any prior literature demonstrated its effectiveness for the EEG-BCI classification?

2. Can the authors clarify the experimental protocol in detail? Were the experiments subject-dependent or subject-independent? How many folds or runs were used for cross-validation? What preprocessing steps were applied to the EEG signals (e.g., band-pass filtering, normalization, artifact removal)?

3. Please specify all experimental settings in all experiments. Regarding the "BCIC2B" dataset, please specify: Which exact dataset version or competition subset was used (e.g., BCI Competition IV-2a or II-2b)? Provide a download link or public reference for reproducibility. What measures were taken to mitigate overfitting, such as early stopping, data augmentation, dropout, or weight decay?

4. Although the EEG-BCI classification inspires the proposed architecture, its design, particularly the covariance of the temporal-segment token and Riemannian high-order pooling, seems more general and not specific to EEG. Could the authors discuss whether their model can be effectively applied to other domains where temporal dependencies and non-Euclidean structure exist (e.g., video, speech, or physiological time series)? If so, what properties of those modalities would make the approach more suitable or interpretable than in EEGs?

### Soundness
2

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
This paper proposes RHOP, a pooling module for EEG foundation models that uses quotient Gaussian embeddings and Riemannian aggregation on SPD manifolds. While the geometric approach is conceptually interesting, the paper suffers from multiple critical flaws: missing the current SOTA baseline (CBraMod in ICLR'25), potential rank deficiency issues, unjustified Gaussian assumptions, and marginal empirical gains.

### Strengths
The method is conceptually interesting and represents a novel approach to token-level aggregation in EEG foundation models. The paper is well-written with clear methodology and good presentation.

### Weaknesses
1.Missing Backbone (CBraMod in ICLR'25): 

The paper cites CBraMod [1] in page 2 but did not evaluate on it. CBraMod outperforms LaBraM, BIOT, and other EEG foundation models, as well as backbone + RHOP proposed in this work, while being substantially lighter in terms of trainable params. Moreover, CBraMod[1] is open‑source and provides pretrained checkpoints, which makes it directly compatible as a plug‑in backbone for RHOP. Ignoring CbraMod as backbone limits the paper's contribution.

2. Rank Deficiency Not Resolved:

The paper does not clearly specify the values of $D$, $N$, $T$, used by each baseline on each dataset. This prevents proper assessment of issues such as rank-deficiency.
When \( D < T \) (common in EEG), the covariance matrix \( $\Sigma_n \in \mathbb{R}^{T \times T}$ \) is rank-deficient, with 
rank($\Sigma_n) \leq D $.
$C_n = D_n^{-1/2} \Sigma_n D_n^{-1/2}\$ preserves rank and therefore cannot convert a semi-definite matrix into a strictly positive definite (SPD) one.
Moreover, in Equation (11), $\mu_n^{(k)} (\mu_n^{(k)})^\top = k \mu_n \mu_n^\top$, leaving the Schur complement as
$S = C_n$. This will leave $Y_n$ in Eq. 11 not SPD, therefore, Riemannian geometry should not be used.  
This severe rank deficiency fundamentally limits the informativeness of the SPD representation.
Particularly, In BCIC2B, where $D=3$, the resulting covariance matrix has very low rank, limiting its capability to capture informative SPD structure.

3. Gaussian Assumption Unjustified:
 
RHOP applies Quotient‑Gaussian embedding to learned token features, not to raw EEG. However, these features are outputs of deep encoders (LaBraM or BIOT), including non-linear activations, normalization layers, and attention mechanisms, making Gaussian distribution highly unlikely, leaving the SPD embedding theoretically questionable. Covariance is not a sufficient statistic for non-Gaussian data, and QGE may discard essential structure.

4. Across several datasets, the improvement of baseline + RHOP over the corresponding baseline is marginal (often less than 1%).

Refs:

[1] Wang, Jiquan, et al. "Cbramod: A criss-cross brain foundation model for eeg decoding." arXiv preprint arXiv:2412.07236 (2024).


I would be willing to raise my score if the authors address any concerns outlined above.

### Questions
See comments above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper highlights a shortcoming of existing foundation models trained on EEG signals: they ignore the informative geometry of the underlying spaces. The paper addresses this by proposing a simple module that can be added to (even pre-trained) existing foundations models to allow them to make use of Riemannian geometry, which is well-known in the BCI community to capture important task-relevant information.

### Strengths
- comprehensive, compelling experiments
- open source code and applications on public datasets
- well written
- clear theoretical motivation
- nice translation of the theory into a practical method

### Weaknesses
- the clear performance gains of the method are nevertheless modest
- the focus is a bit limited (compared to the broader ICLR community), focusing on EEG applications

### Questions
1. are there other applications of this besides EEG?
2. L058: Is it really about scaling, or more about applying/tailoring/tuning?
3. L070: What does CLS stand for?
4. L089 and 094: SPD is an adjective, so what's the missing noun it's modifying here? Matrices? Manifold?
5. L104 and 106: A word is missing after EEG here. Maybe "signals" or "measurements"?
6. L127: Can the authors more precisely and mathematically explain what is meant here (and elsewhere in the section) by "collapses all spatiotemporal tokens into a single global discriptor"? Isn't the proprosed method also doing that via eq. (9), but in a geometry-aware way?
7. L185 and 189: $\mathcal{Q}\mathcal{N}(n)$ isn't a single distribution but the space of all distributions, isn't it? And then eq. (5) is a specific (quotient) distribution that is also a representative of the equivalence class of the nonquotient distributions? So each element $[\Sigma, \mu]$ is an equivalence class of distributions $(\Sigma^\prime, \mu)$ where ${\Sigma}, {\Sigma^\prime}$ have the same corresponding correlation matrix?
8. L194: why not use something more informative and accessible than "↓", like "Appendix F"?
9. L210: is eq. (7) here the same as (4)?
10. Tables 3 and 4: why does the param count for LaBraM-Base+iSICE drop by over 20k here, while all other methods have little or no drop?

### Soundness
4

### Presentation
3

### Contribution
3
