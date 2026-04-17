# Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Hierarchical coding offers distinct advantages for learned image compression by capturing multi-scale representations to support scale-wise modeling and enable flexible quality scalability, making it a promising alternative to single-scale models. However, its practical performance remains limited. Through spectral analysis of training dynamics, we reveal that existing hierarchical image coding approaches suffer from cross-scale energy dispersion and spectral aliasing, resulting in optimization inefficiency and performance bottlenecks. To address this, we propose explicit spectral regularization schemes for hierarchical image coding, consisting of (i) intra-scale frequency regularization, which encourages a smooth low‑to‑high frequency buildup as scales increase, and (ii) inter-scale similarity regularization, which suppresses spectral aliasing across scales. Both regularizers are applied only during training and impose no overhead at inference. Extensive experiments demonstrate that our method accelerates the training of the vanilla model by 2.3$\times$, delivers an average 20.65\% rate–distortion gain over the latest VTM-22.0 on public datasets, and outperforms existing single-scale approaches, thereby setting a new state of the art in learned image compression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
They propose explicit spectral regularization (intra-scale frequency regularization for smooth frequency buildup across scales, inter-scale similarity regularization to suppress cross-scale aliasing) — applied only in training with no inference overhead. Experiments show the method accelerates vanilla model training by 2.3x, achieves 20.65% average rate–distortion gain over VTM-22.0 on public datasets, outperforms single-scale approaches, and sets a new SOTA in learned image compression.

### Strengths
1. This manuscript tries to extract disentangled features from images using spectral regularization.

2. The motivation is clear.

3. The results about the performance of the proposed method are convincing in Kodak, CLIC datasets.

### Weaknesses
1. The related details of the proposed method are not described clearly. For example, how to obtain the y-axis of Figure 1 ? What does it mean about the lines with different color of Figure 8?

2. The training of inter-scale regularization as shown in Figure 5 may not be stable since $z_1, z_2$ maybe the noise at the early stage and the corresponding regularization makes no sense.

3. There exists no analysis or discussion about "Different scales converge to their respective frequency bands at different rates" at line 183. It could be better if this manuscript provides some clues.

4. For visualization in Figure 6, it is hard to totally distinguish the frequencies of the details without any aliasing. Besides the demonstration of Figure 6(b) has little color shift compared with 6(c), can the regularization be more effective for color representation?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed two techniques to address the inefficiency in training in hierarchical coding. The first is for intra-scale frequency regularization, which is to guide the training frequencies by progressive truncation. The second for inter-scale similarity regularization, which is to suppress the similarity between neighboring scales (?). Experimental results demonstrate the successfulness of the proposed method.

For the weakness, 1. The inter-scale similarity regularization part needs more explanation. It looks like that Eq.(6) encourages similarity instead of suppressing it. 2. Some figures are not helpful for explaining the method, e.g. Figs 4 and 5.

### Strengths
The authors proposed two techniques to address the inefficiency in training in hierarchical coding. The first is for intra-scale frequency regularization, which is to guide the training frequencies by progressive truncation. The second for inter-scale similarity regularization, which is to suppress the similarity between neighboring scales (?). Experimental results demonstrate the successfulness of the proposed method.

### Weaknesses
1. The inter-scale similarity regularization part needs more explanation. It looks like that Eq.(6) encourages similarity instead of suppressing it. 
2. Some figures are not helpful for explaining the method, e.g. Figs 4 and 5.

### Questions
1.	The inter-scale similarity regularization part needs more explanation. It looks like that Eq.(6) encourages similarity instead of suppressing it.
2.	Some figures are not helpful for explaining the method, e.g. Figs 4 and 5. 
3.	What is the meaning of “time t”?
4.	The inner minimum in Eq.(4) seems not necessary.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles optimization bottlenecks in hierarchical image compression by identifying that standard training suffers from "spectral aliasing" and "energy dispersion" across scales. To resolve this, the authors introduce two training-only spectral regularizers: an intra-scale DCT-based curriculum for progressive low-to-high frequency learning, and an inter-scale similarity penalty to reduce redundancy. This approach accelerates training by 2.3x and sets a new state-of-the-art, achieving a 20.65% average BD-Rate saving over the VTM-22.0 video codec. I like the paper's insightful analysis and strong empirical results. However, its methodological soundness is weakened by a lack of theoretical justification connecting the proposed regularization schemes to their stated goal of spectral separation.

### Strengths
1.The paper provides a novel and insightful spectral analysis, identifying "energy dispersion" and "spectral aliasing" as the root causes of training difficulties in hierarchical compression models. The compelling visualizations strongly support this diagnosis.
2.The proposed regularization techniques are well-designed and directly target the diagnosed spectral issues. The approach is rigorous, logically sound, and adds no inference overhead, making it practical.
3.The method achieves state-of-the-art performance with substantial gains, including a 20.65% average BD-Rate saving over VTM-22.0. The 2.3x training acceleration is a significant practical advantage.
4.The paper is supported by thorough experiments, including extensive SOTA comparisons, detailed ablation studies, and proven generalizability to other architectures, which robustly validate the authors' claims.

### Weaknesses
1.The paper's design, guided by the "frequency principle," assigns low-frequency content to semantically deeper scales and high-frequency content to shallower scales. This creates an apparent tension with the common understanding that more challenging, high-frequency details often benefit from the greater expressive power of deeper network layers. While the empirical results are strong, the paper's methodological contribution would be further strengthened by a clearer theoretical justification for this seemingly counter-traditional design, explaining why assigning the most complex information to relatively shallower structures is advantageous in this context.

2.The paper lacks a theoretical basis for its claim that an L2 latent penalty (Eq. 6) enforces spectral separation between scales. The connection between latent distance and spectral orthogonality is not established. Furthermore, the implementation uses a 1x1 convolution, a feature-mixing operation that could potentially contradict the goal of suppressing spectral aliasing. The authors should clarify the true source of this regularizer's effectiveness.

### Questions
1.Regarding Figure 1:
 Could you provide a more detailed technical explanation of how these heatmaps were generated? Specifically, how is the "spectral overlap" between a scale's contribution and the input image quantitatively defined and computed? The current description in the Appendix is a bit brief for full reproducibility.

2.Regarding Figure 9:
Could you elaborate on the specific design purpose and function of the FSP?
What is the relationship between the FSP and the proposed Inter-Scale Latent Regularization? Do they work synergistically, or are they independent components?
I noticed a similar shortcut structure in a recent related work, AuxT [R1]. Could you clarify the key differences and relationships between your FSP and the auxiliary structure in AuxT?

[R1] Li, et. al, On Disentangled Training for Nonlinear Transform in Learned Image Compression. ICLR 2025.

3.Can the proposed spectral regularization methods be adapted for non-hierarchical (single-scale) compression frameworks?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the optimization challenges in hierarchical learned image compression via spectral analysis. The authors identify two major issues—cross-scale energy dispersion and spectral aliasing—and introduce two regularization strategies to mitigate them: (1) intra-scale frequency regularization through progressive DCT truncation, and (2) inter-scale latent regularization using similarity penalties. Experimental results demonstrate strong performance, achieving a 20.65% BD-Rate improvement over VTM-22.0 and a 2.3× training speedup without adding inference complexity.

### Strengths
1. The spectral interpretation of hierarchical training dynamics provides an intuitive understanding of cross-scale interactions and training instability.
2. Strong empirical results: 1) State-of-the-art compression performance across multiple datasets, 2) Significant 2.3× training acceleration without inference overhead, 3) Robust performance across diverse resolutions (480p–4K).
3. The proposed regularizers are training-only and do not increase inference complexity, making the method easy to deploy.
4. The identification of spectral dispersion and aliasing as key bottlenecks adds valuable understanding to hierarchical compression models.

### Weaknesses
1. The inter-scale regularization term minimizes $ L2(z_{l-1}, Conv(DWT(z_l)))$, which encourages similarity between adjacent scales. However, the text claims the objective is to make features “as distant as possible”. This appears contradictory and should be clarified.
The paper lacks a formal explanation of how minimizing latent similarity mitigates spectral aliasing. The relationship between spatial-domain similarity and frequency-domain decoupling needs stronger theoretical grounding.
2. The model diagram includes the FSP component, but it is not sufficiently discussed in the text. Compared to QARV, it remains unclear how much FSP contributes to convergence speed and rate–distortion performance. FSP seems conceptually close to AuxT [1], yet no detailed comparison or ablation is provided, despite citing AuxT.
3. The reparameterized block structure is mentioned but not thoroughly evaluated and its impact on performance should be quantified.
4. Hierarchical VAE has been previously applied to image compression (e.g., by Yueyu Hu et al. [2-3] ); this line of work should be properly cited and discussed.
5. It would be valuable to analyze whether the proposed regularization strategies could be extended to models such as HPCM or MLIC++, where complex context modeling divides single-scale features into multiple slices. Would regularizing these slices be similarly effective?

[1] Li, Han, et al. "On disentangled training for nonlinear transform in learned image compression." arXiv preprint arXiv:2501.13751 (2025).

[2] Hu, Yueyu, et al. "Learning end-to-end lossy image compression: A benchmark." IEEE Transactions on Pattern Analysis and Machine Intelligence 44.8 (2021): 4194-4211.

[3] Hu, Yueyu, Wenhan Yang, and Jiaying Liu. "Coarse-to-fine hyper-prior modeling for learned image compression." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 07. 2020.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
