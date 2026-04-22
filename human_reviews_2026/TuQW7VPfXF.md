# Soft Equivariance Regularization for Invariant Self-Supervised Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 8

## Abstract
Self-supervised learning (SSL) typically learns representations invariant to semantic-preserving augmentations (e.g., random crops and photometric jitter). While effective for recognition, enforcing strong invariance can suppress transformation-dependent structure that is useful for robustness to geometric perturbations and spatially sensitive transfer. A growing body of work, therefore, augments invariance-based SSL with equivariance objectives, but these objectives are often imposed on the same **final** (typically spatially-collapsed) representation. We empirically observe a trade-off in this coupled setting: pushing equivariance regularization toward deeper layers improves equivariance scores but degrades ImageNet-1k linear evaluation, motivating a layer-decoupled design. Motivated by this trade-off, we propose **Soft Equivariance Regularization (SER)**, a plug-in regularizer that decouples where invariance and equivariance are enforced: we keep the base SSL objective unchanged on the final embedding, while softly encouraging equivariance on an \emph{intermediate spatial token map} via analytically specified group actions $\rho_g$ (e.g., $90^{\circ}$ rotations, flips, and scaling) applied directly in feature space. SER learns/predicts no per-sample transformation codes/labels, requires no auxiliary transformation-prediction head, and adds only **1.008$\times$** training FLOPs. On ImageNet-1k ViT-S/16 pretraining, SER improves MoCo-v3 by **+0.84** Top-1 in linear evaluation under a strictly matched 2-view setting and consistently improves DINO and Barlow Twins; under matched view counts, SER achieves the best ImageNet-1k linear-eval Top-1 among the compared invariance+equivariance add-ons. SER further improves ImageNet-C/P by **+1.11/+1.22** Top-1 and frozen-backbone COCO detection by **+1.7** mAP. Finally, applying the same **layer-decoupling** recipe to existing invariance+equivariance baselines (e.g., EquiMod and AugSelf) improves their accuracy, suggesting layer decoupling as a general design principle for combining invariance and equivariance. Code is available at https://github.com/aitrics-chris/SER.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present a simple yet effective method for introducing an equivariant regularisation constraint into self-supervised ViTs. The goal is to better decouple the traditional approach of simultaneous invariant and equivariant learning on the output embedding space by enforcing intermetiate layers to be equivairant to the input transformations by preserving the spatial structure, while the output vector remains invariant. The approach demonstrates strong invariant performance on downstream linear evaluation tasks, and scales to imagenet level datasets. However, its primary weakness is the lack of a direct evaluation to confirm that the intermediate layers actually learn the desired equivariant properties, and its motivation for avoiding transformation labels is partially contradicted by its own methodology.

### Strengths
- The analysis and subsequent conclusion that enforcing equivariant at the output embeddings is a significant and interesting contribution to the equivariant self-supervised literature. This provides new insights and research directions that the community will find valuable. 
- The method presented is simple yet effective at improving downstream invariant performance. The approach is generalisable, and enforces the desired properties at defined layers within the network. The enforcement of equivariant feature maps at intermediate layers is seemingly correct. 
- The empirical results demonstrate the effectiveness of the method and its ability to scale, where prior works have not yet managed to achieve competitive performance on large-scale datasets. 
- Limitations are presented and discussed
- Code and implementation details are provided for validation and reproduction.

### Weaknesses
**Major:**
- It is mentioned in the introduction one of the drawbacks is that knowing the transformation labels is an inherent drawback, however, the proposed method also requires the transformation group to be known. While not necessarily a weakness in my opinion, this contradicts the authors motivation.
- No evaluation of the equivariant property is performed. For example, how well do the intermediate representations which the regularisation has been applied to and the invariant representations encode the translation information. It cannot be claimed that equivariance has been enforced without some analysis of the inversion property at a minimum. 

**Minor:**
- I would argue that the regularisation is not necessarily soft. The authors are specifically enforcing features to preserve the transformation in the latent space of the intermediate layers via a direct distance metric. This seems like a specific hard constraint.
- The additional computational overhead is presented as a limitation but not quantified.
- It is not clear how the 3DIEBench 3D tranformations are encoded into the equivariance regularisation. Are the these object transformations encoded in the g parameter or are the standard crops and rotations applied?
- See questions for additional minor weaknesses.

### Questions
1. In practice how do you determine which layers to apply the soft-equivariant objective to? For some transformation groups the layers which have the most significant effect may change, therefore in practice this is a heuristic / empirical pre-processing step to identify? How does the layer performance change under different architectures and datasets?
2. You do not want to operate the equivariant objective on the pooled feature vector of the output. Hence the question is, why not just omit the final pooling/mean layer? Also prior works such as Konstantinou, et al., (2025) EquiCaps: Predictor-Free Pose-Aware Pre-Trained Capsule Networks. perform on the feature embedding prior to pooling hence maintaining the spatial component. 
3. How does the model perform if you apply the equivariant regularisation on all layers of the equivariant network? Why perform on just one when you can enforce all features?

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
- Introduces SER, which decouples invariance and equivariance by applying an equivariance loss to intermediate, spatially structured features while learning invariance on the final head. 
- Demonstrates ViT-S/16 pretraining on ImageNet-1k with linear/non-linear evals and robustness suites (IN-Sketch, IN-V2, IN-R, IN-C/P) and an OOD 3D dataset (3DIEBench). Shows consistent gains over MoCo-v3 and positive deltas when plugged into DINO and Barlow Twins; also improves frozen-backbone object detection on COCO.

### Strengths
- The paper is well written and easy to follow
- SER does not require transformation labels or predictors; it leverages known group actions on feature maps with a simple NT-Xent loss, keeping the method lightweight and scalable.
- By excluding crop from set of groups $G$ and keeping it in the invariance path, the method respects group theory while preserving the benefits of strong augmentations for representation quality. 
- Improves linear/nonlinear evaluation on ImageNet-1k across MoCo-v3/DINO/Barlow Twins and shows better robustness on ImageNet-C/P and 3DIEBench

### Weaknesses
- The paper applies rotations, flips, and anisotropic scaling at intermediate feature maps but does not detail how resampling/interpolation, padding patch-grid alignment are handled. On discrete token lattices, 90° rotations are exact but scales are not. The chosen interpolation kernel can materially affect equivariance error. A detailed description and sensitivity study are missing. 
- SER avoids predicting labels, yet it relies on knowing $g_2g_1^{-1}$ from the augmentation pipeline and on having a known group action $\rho_g$ at feature level. This limits applicability when transforms are stochastic/unknown (e.g., random photometric transforms, camera jitter).
- ImageNet Top-1 gains over MoCo-v3 are quite modest, especially in 2-view setting. No confidence intervals or multi-seed statistics are provided, so practical significance remains uncertain.
- By construction, SER cannot handle non-group transforms (e.g., crop, typical color jitter) except via the invariance path, so transformation-aware tasks for photometric changes or crops remain unaddressed.
- The $b_2$ path introduces 90° rotations and removes crop. It is possible that the gains partially stem from the augmentation distribution shift rather than the equivariance loss. One should include the same set of augmentations for MoCo-v3 but without 
 the equivariant loss i.e. equivalently run SER with $\lambda=0$. Its important to disentangle the effect of augmentations and the equivariant loss. 
- There are related papers that put the equivariant objective “earlier” (pre-projector / on spatial features) and keep the invariant objective “later” (post-projector) [1, 2] . The contributions of this work don't seem to be significantly novel compared to existing literature. 

[1] Garrido, Quentin, Laurent Najman, and Yann Lecun. "Self-supervised learning of split invariant equivariant representations." arXiv preprint arXiv:2302.10283 (2023).

[2] Devillers, Alexandre, and Mathieu Lefort. "Equimod: An equivariance module to improve self-supervised learning." arXiv preprint arXiv:2211.01244 (2022).

### Questions
See weaknesses above

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors tackle two aspects of equivariant representation learning. First, is scaling to imagenet pretraining, where most works tend to focus on smaller datasets. The second is to not enforce equivariance at a representational level, but partially through the network. The rest of the network is then trained with the final goal being the usual invariance, leading to invariant representations at the end of the network. To enforce equivariance partially through the network, the authors leverage the spatial structure that is kept to apply the same transformation as done on the input.

### Strengths
- The proposed approach can be integrated with off the shelf SSL methods and architectures, making it widely usable.

- The goal of learning invariant representations, but still enforcing intermediate representations to be equivariant is an interesting take on equivariant SSL in general.

- The scale of the experiments is appreciated, making it closer to state of the art setups.

- The authors demonstrate performance gains over baseline methods and other equivariant methods, with the caveat that here the representations are ultimately invariant.

- Performance gains are observed over the baseline methods across datasets, with some caveats (see weaknesses)

### Weaknesses
1) Line 94: “Our approach requires no explicit transformation labels”. If my understanding is correct, having access to the exact transformation (generated from the labels) is required, which while a different constraint can be even harder to obtain in practice. Perhaps [1] (cited in the paper) should be discussed more, as the authors use a similar idea by applying the transformation to few dimensions of the representations, as defined by the dataset/transformation.

2) Claims of being the first to scale equivariant SSL to ImageNet (abstract, lines 61-63) are incorrect (and unnecessary) as this was already done in [2], training ViT-B models on ImageNet and studying the impact of equivariance on downstream tasks.

3) Unclear loss definitions. $L_{inv}$ is never properly defined, and the choice of negative samples for both  $L_{inv}$ and $L_{equi}$ is unclear. Lines 295-296 indicate that negative pairs are samples from $z$, but then line 300 indicates that “we omit negative pairs sample from the same image as the anchor”. I assume that the latter is correct, but the notation must be made clearer. For $L_{equi}$, how are the negative pairs chosen ? Across the same image or across the batch ? While appendix B.4 clarifies a bit, it should not take this amount of work to understand the loss, which is the core of the paper.

4) The authors mention that cropping is removed from the equivariant sub-batch as it does not form a group, but other commonly used augmentations do not either. The standard protocol of BYOL[3] (used in MoCov3 that is the base of experiments in the paper) includes grayscale, blurring and solarization, all of which definitely do not have an inverse. This would suggest that experiments are performed with suboptimal data augmentations in the invariant batch, which makes the comparison to the baselines more complex/unclear.

[1] Marchetti, Giovanni Luca, et al. "Equivariant representation learning via class-pose decomposition." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

[2] Garrido, Quentin, et al. "Learning and leveraging world models in visual representation learning." arXiv preprint arXiv:2403.00504 (2024).

[3] Grill, Jean-Bastien, et al. "Bootstrap your own latent-a new approach to self-supervised learning." Advances in neural information processing systems 33 (2020): 21271-21284.

### Questions
1) Since $f^{(1)}$ is equivariant, it could be interesting to visualize the feature maps to see how the spatial structure is preserved compared to other approaches/a fully invariant baseline.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes using soft equivariance regularization (SER), where output features use an invariant loss while intermediate features use equivariant regularization in the context of SSL. For transformers, the output features are spatially collapsed and thus in order to use both invariance and equivariance objectives commonly used in SSL, the authors encourage equivariance only in the intermediate representations, which are then passed to an invariant block that also takes [CLS] token as input. The authors show results on ViT-S on ImageNet-1k with linear and nonlinear evaluations, and compare against various SSL baselines.

### Strengths
- Practical way to use transformers for invariance and equivariance objectives in SSL, considers spatial collapse and distinguishes between invertible and noninvertible augmentations.
- Experiments are done at ImageNet scale.
- The evaluation is pretty comprehensive: considers linear and nonlinear evals, robustness (ImageNet-P), and also detection transfer. They consider all the SOTA SSL baselines I'm aware of and also do extensive ablations on the intermediate layer position.
- Does not require transformation labels.

### Weaknesses
See questions.

### Questions
- The method seems to hinge heavily on $\lambda$ to balance $L_{inv}$ and $L_{equiv}$ and also the batch partition ratio. Is the method sensitive to these parameters?
- It's not clear if regularizing intermediate representations do in fact encourage soft equivariance on the output representations. Is there any way to verify this on the output representations?
- I feel that many SSL methods heavily use non-invertible transformations (e.g. crop, color jitter, random grayscale, mixup/cutmix, etc). Partitioning the batch is reasonable, but does the method rely on using a balance of invertible and noninvertible transformations?
- How does SER fit in with MAE style pretraining? Would you say SER is an alternative to masking for learning representations?

### Soundness
3

### Presentation
4

### Contribution
3
