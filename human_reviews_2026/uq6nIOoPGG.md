# Learning Coarse-Grained Representations: An Exploration of Mutual Information via Hyperspherical Density

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 0, 2

## Abstract
We revisit InfoMax for representation learning, using hyperspherical geometry with a non-parametric von Mises-Fisher kernel density estimator and differential entropy. This method is minimal with no asymmetry and trains stably. Results are competitive on smaller datasets such as CIFAR-10, STL-10 and LC25000, but lags behind modern baselines on ImageNet-1000. Experiments show that weakening the global entropy term consistently helps classification accuracy, suggesting that strict mutual information classification favors coarse grouping over fine discrimination.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work deals with self-supervised learning of visual representation, exploring the InfoMax principle with a restricted hyperspherical geometry. The densities to compute the entropy are estimated with the  von Mises-Fisher non parametric kernel density estimator and the mutual information approximated by a difference of the global differential entropy and the local one.

### Strengths
* while several works seem quite close from the proposed one (see weaknesses) it seems that the use of a direct non-parametric density estimation wit vMF and an approximation of the mutual information by a difference of differential entropies has never been proposed. It is theorically new *and* interesting.

* a large part of the paper is dedicated to the ablation of the approach. Three hyperparameters are studied quantitavelly and qualitative results are also reported.

* the full code is provided as supplementary material, ensuring that it will be released to the community to favour reproducibility. The code is particularly well written and structured.

### Weaknesses
* the proposed work should position itself:
  - in relation with [b], where two properties of contrastive representation learning is studied on the Hypersphere.  After their Theorem 1, they relate their work with  feature distribution entropy estimation (via a von Mises-Fisher kernel density estimation) and the InfoMax principle. 
  - in relation with [d] which interpret DINO features (the same as those used in the paper) as a von Mises-Fisher mixture model

* the quantitative evaluation are reported with a ResNet-18 backbone only, except of the proposed approach that is estimated with a ViT-16 backbone. One can regret that this last backbone is not used for baseline approaches.

**minor**:
  - line 030: SSL is not defined, it should be first written on line 024: "Self-supervised learning (SSL) methods..."
  - line 045: ICA is not defined (independent component analysis). By the way, ICA was "grounded" by several principles, including contrast maximization by Comon (1991) and Joint Approximate Diagonalization of Eigenmatrices by Cardoso and Souloumiac in 1993.Regarding the Infomax principle, Bell and Sejnowsky indeed relied on it, but Nadal and Parga showed in 1994 that Infomax was equivalent to redundancy reduction principle, opening the path to its usage for ICA.
  - line 053: MI should be defined previously (mutual information on line 046)
  - line 091: "we an" --> "we can"
  - line 101: "Models trained on LC25000 were trained for 25000 epochs" --> "Models were trained on LC25000 for 25000 epochs"
  - implementation details (line 098-103) may be better placed in section 4. In particular the citation of datasets (currently n lines 107-108) should be before the details currently n line 101-103 
  - baseline method in Table 1, while known, should be cited in the text
  - line 395: the actual published paper should be cited rather than the arxiv report for ( Kalapos and Gyires-Tóth, 2024) [a]
  - while different, it may worth to position the work w.r.t [c]

[a] A. Kalapos and B. Gyires-Tóth, "Whitening Consistently Improves Self-Supervised Learning," 2024 International Conference on Machine Learning and Applications (ICMLA), Miami, FL, USA, 2024, pp. 448-453, 

[b] Wang and Isola (2020) [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://proceedings.mlr.press/v119/wang20k/wang20k.pdf), ICML

[c] Li et al (2024) Probabilistic Contrastive Learning with Explicit Concentration on the Hypersphere, arxiv:2405.16460

[d] Govindarajan et al (2023) DINO as a von Mises-Fisher mixture model, ICLR

### Questions
* why the top-5 accuracy is ont reported for LC-25000 in Table 1 ?
* why the performance on ImageNet-1k is not reported for the baselines in Table1 ? At least from previous papers...
* which K is used for K-NN in Table 1? 
* what is the value of $\kappa$ for the main results of section 4? Are $\alpha$ and $\beta$ set to one (as said line 091)

### Soundness
3

### Presentation
3

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
This paper revisits the InfoMax principle for self-supervised representation learning, proposing a simple implementation based on hyperspherical geometry and non-parametric von Mises-Fisher (vMF) kernel density estimation to compute global and local differential entropies.

### Strengths
Simplicity and conceptual clarity in method.

### Weaknesses
Limited Novelty and Technical Contribution

- The proposed formulation appears to be a relatively direct combination of established components—namely hyperspherical embeddings, von Mises–Fisher (vMF) kernels, and differential entropy estimation.
- The paper does not present a substantial theoretical advancement or novel derivation beyond existing InfoMax- or kernel-density-based self-supervised learning (SSL) frameworks.

Poor Scalability and Weak Empirical Results

- The performance reported in Table 1 is markedly low, contradicting the paper’s claims; even on datasets with a small number of classes, the method underperforms.
- Figures 1 and 2 omit results for several experimental settings, and there is no comparison against standard SSL benchmarks, which undermines the empirical credibility of the work.
- Similarly, Figures 3 and 4 report results only on small or relatively simple datasets, further limiting the strength of the experimental validation.
- The claim that the learned representations are “coarse-grained” is largely descriptive and lacks rigorous quantitative validation.
- No ablation or analysis is provided to substantiate the connection between this phenomenon and mutual-information behavior in a measurable way.

Lack of Theoretical Grounding

- The paper fails to formally justify how the proposed estimator approximates true mutual information.
- From the perspective of mutual-information estimation, the work does not convincingly demonstrate either theoretical rigor or empirical superiority; a comprehensive comparison against existing MI estimators is essential.
- In contrast to prior MI estimators that do not rely on hyperspherical geometric assumptions, the proposed approach is considerably more restrictive.
- The observation that relaxing the global entropy term improves classification lacks analytical explanation and may merely reflect optimization dynamics rather than a genuine information-theoretic insight.

### Questions
- In Equation (4), the formulation no longer corresponds to mutual information when both α and β differ from 1. Could you clarify the rationale for setting these parameters to values other than 1, and how this choice affects the interpretation of the objective?
- What is the basis for claiming that the proposed method learns coarser representations? Could you provide a theoretical explanation of why the proposed loss function encourages such coarse-grained structure, and offer empirical evidence demonstrating that this property indeed emerges in the learned embeddings?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes using a vMF (von Mises–Fisher) kernel as a mutual-information estimator for self-supervised learning.
The authors evaluate the method on four datasets with Grad-CAM, CLS-to-patch attention analyses, and hyperparameter ablation studies.

### Strengths
The paper proposes using a vMF kernel as a mutual-information estimator for self-supervised learning.
The authors evaluate the method on four datasets with Grad-CAM, CLS-to-patch attention analyses, and hyperparameter ablation studies.

### Weaknesses
1. The paper proposes using a vMF kernel as an MI estimator for self-supervised learning but does not explain how it differs from existing estimators such as InfoNCE or what advantages it offers.

2. The proposed loss is algebraically similar to the InfoNCE loss, differing mainly in the number of positive pairs and its interpretation as a vMF-based entropy estimator, which weakens the novelty of the method.

3. The results in Table 1 do not demonstrate any clear advantage of the proposed vMF-based self-supervised method compared with other baselines.

4. Figure 1 does not convincingly show that the trained model focuses on the main object, as the Grad-CAM activation is broadly distributed over non-object regions.

### Questions
Please see weakness

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the InfoMax objective function for self-supervised representation learning. The authors estimate mutual information using a von Mises-Fisher (vMF) kernel density estimator under hyperspherical geometry (i.e. embeddings have unit length). They evaluate their method on several datasets (STL, CIFAR, LC-25000, and ImageNet-1K) using standard data augmentations to form positive pairs. The linear probing results for classification show that the method achieves competitive performance on smaller datasets but falls behind on ImageNet-1K.

### Strengths
1- the motivation for using vMF kernels is clear, and the objective function is minimal and theoretically grounded.

2- good empirical insight from ablations on projector dimension and band width parameter.

### Weaknesses
1- the paper feels incomplete. The direction and main research question are not well defined. It’s unclear whether the goal is to analyze InfoMax behavior or to improve performance.

2- more experiments, analyses, and discussion are needed. For instance, comparisons to alternative entropy estimators or normalization schemes (e.g., stop-grad, whitening) would add clarity.

3- the study of the bandwidth parameter κ closely parallels the temperature parameter in contrastive learning. This connection could be more explicitly analyzed.

### Questions
1- can you analyze how your loss function relates to the alignment and uniformity objectives (Wang & Isola, 2020)? Under similar assumptions, what does your objective correspond to?

2- can you elaborate on the motivation for using vMF kernel estimation instead of InfoNCE or other parametric estimators?

3- is the main goal of your work to achieve strong downstream performance, or primarily to analyze the behavior of InfoMax under certain assumptions?

### Soundness
2

### Presentation
2

### Contribution
3
