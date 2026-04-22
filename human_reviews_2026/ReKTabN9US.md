# Disentangling Content from Style to Overcome Shortcut Learning: A Hybrid Generative-Discriminative Learning Framework

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Despite the remarkable success of Self-Supervised Learning (SSL), its generalization is fundamentally hindered by Shortcut Learning, where models exploit superficial features like texture instead of intrinsic structure. We experimentally verify this flaw within the generative paradigm (e.g., MAE) and argue it is a systemic issue also affecting discriminative methods, identifying it as the root cause of their failure on unseen domains. While existing methods often tackle this at a surface level by aligning or separating domain-specific features, they fail to alter the underlying learning mechanism that fosters shortcut dependency.
To address this at its core, we propose HyGDL (Hybrid Generative-Discriminative Learning Framework), a hybrid framework that achieves explicit content-style disentanglement. Our approach is guided by the Invariance Pre-training Principle: forcing a model to learn an invariant essence by systematically varying a bias (e.g., style) at the input while keeping the supervision signal constant. HyGDL operates on a single encoder and analytically defines style as the component of a representation that is orthogonal to its style-invariant content, derived via vector projection. 
This is operationalized through a synergistic design: (1) a self-distillation objective learns a stable, style-invariant content direction; (2) an analytical projection then decomposes the representation into orthogonal content and style vectors; and (3) a style-conditioned reconstruction objective uses these vectors to restore the image, providing end-to-end supervision.
Unlike prior methods that rely on implicit heuristics, this principled disentanglement allows HyGDL to learn truly robust representations, demonstrating superior performance on benchmarks designed to diagnose shortcut learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes HyGDL, which tackles shortcut learning in SSL by explicitly disentangling content from style. Guided by an Invariance Pre-training Principle, a student–teacher encoder learns a style-invariant “content direction”; features are then analytically split via projection into orthogonal content and style vectors, which drive a style-conditioned reconstruction decoder. This principled hybrid design improves OOD robustness and competitive PACS/DomainNet results, surpassing heuristic augmentations and revealing MAE shortcut dynamics during pretraining, empirically.

### Strengths
1.This paper is well written, easy to follow.

2.Experimental results are better than MAE and SimCLRv2.

### Weaknesses
1.Experiments in Table 2 are not convincing, HyGDL achieves the inferior performance compared to CycleMAE, DIMAE and BrAD.

2.The proposed HyGDL essentially uses AdaIN as a data augmentation technique during training, all modules in HyGDL are widely explored in previous methods.

3.The assumptions and equations for style/content disentanglement is un-reliable. Content and style are not simply orthogonal; sometimes the content itself becomes part of the style, and the style defines how the content exists. For example, in Pablo Picasso’s Les Demoiselles d’Avignon, the five female figures serve as the painting’s “content,” yet their highly geometric and deconstructed forms simultaneously constitute its distinctive style. The definitions of vc, cA, sA in your paper are without any theoretical basis.

4.AdaIN introduces style bias by oversimplifying style as mean and variance, causing dataset-induced and content-dependent distortions. It often produces uniform stylization and loses fine structural or textural details due to ignored spatial correlations and channel dependencies. Not good for style augmentation.

### Questions
1.Why style images solve the Second-Order shortcut?

### Soundness
1

### Presentation
1

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
This paper presents HyGDL, a hybrid generative–discriminative SSL framework that disentangles content from style to counter shortcut learning. Via self-distillation (style-invariant content direction), orthogonal projection, and style-conditioned reconstruction, it yields more robust, transferable representations and stronger domain generalization on PACS/DomainNet while revealing overfitting dynamics.

### Strengths
1.The writing of the paper is generally clear.
2.The experiments demonstrate the effectiveness of their methods compared to MAE and SIMCLRv2.

### Weaknesses
1.The motivation of this paper is unclear. The descriptions for first-order and second-order shortcuts lack theoretical and experimental support, and why AdaIN solve these shortcut s?

2.The the main insight of this paper is a new data augmentation AdaIN, and the other proposed modules are widely used in previous methods. Lack of novelty.

3.The experiments are not satisfactory, CycleMAE, BrAD and DIMAE achieve a much superior performance compared to proposed HyGDL.

4.AdaIN is a bias style transfer method with similar stylized texture and color across all style. Applying it to SSL constitutes another "first-order shortcut".

### Questions
The equations of vc, ca and sa are un-reasonable, if vc=zs+zt represents the content derection vector, why ca is the projection of zs onto vc?

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
This paper introduces a method to mitigate shortcut learning to achieve domain generalization in a self-supervised learning context (Unsupervised Domain Generalization). The authors propose the use of AdaIN for style transfer of the input image and then use the resulting image pairs as input to a self-distillation component to learn a style-invariant representation. The orthogonal vector to this representation is the style vector. These are used in a self-reconstruction and a cross-reconstruction loss component. The resulting representation is trained solely on the source domains (no ImageNet pretraining) and achieves improved accuracy compared to similarly trained models.

### Strengths
- The motivation and problem that this paper addresses (shortcut learning in SSL) is important and less studied compared to the corresponding problem in the supervised learning setting. Also, the diagnosis of the problem in the introductory experiment supports the motivation well.

- The geometric modelling of the style component via orthogonal projection is interesting and enforces the orthogonality of representations

- The proposed method outperforms the baselines that do not use ImageNet pretraining.

### Weaknesses
The novelty of the paper is relatively limited. It combines a BYOL-style self-distillation approach with a vector embedding and AdaIN decoder reconstruction, although the explicit disentanglement is interesting. Furthermore, it is not clear why the proposed "Invariant Pre-training Principle" is different from applying style augmentation to achieve invariance, which has been extensively used to improve generalization, including the domain generalization problem. Furthermore, the broad claim that a nuisance variable is systematically varied is not supported by the methodology nor the experiments, since these are limited to image datasets, and a single method for varying style (i.e., AdaIN).

Regarding the experimental results, although there are effectiveness gains, e.g., 58.13% accuracy in PACS vs 52.73% for MAE with AdaIN and 51.2% for DARLING (29.25%, 22.62% and 23.91% respectively for DomainNet), these come at a significant increase of computational resources.

Some aspects of the paper are not clearly specified or are assumed known, which in some cases hinders understanding. See the questions below.

### Questions
- The evaluation protocol is not clearly specified. Is this the leave-one-domain-out cross-validation approach, where all source domains are used for training and the left-out domain as test? 
 - What is the architecture of the classification head? 
 - How is the final classifier trained? 
 - Is it linear probing or some other network? 
 - How did you choose the epochs to use for the three stages of training?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes HyGDL, a Hybrid Generative–Discriminative Learning Framework designed to mitigate shortcut learning in self-supervised learning (SSL). The authors argue that current SSL methods (MAE, SimCLR, BYOL) rely on superficial features such as texture, limiting generalization. To address this, HyGDL introduces an Invariance Pre-training Principle and performs explicit content–style disentanglement.

### Strengths
The paper identifies shortcut learning as a key issue in SSL and presents a coherent argument that simple augmentations do not address the root cause. Combining discriminative self-distillation and generative reconstruction is conceptually appealing.

### Weaknesses
The idea of disentangling style and content via orthogonal projection and using AdaIN for style injection is incremental. Prior works such as “Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning” already explored similar ideas. The proposed “Invariance Pre-training Principle” is largely a rephrasing of established domain generalization concepts (e.g., invariance to nuisance variables).

Performance gains are small (e.g., +6% on PACS, still far below CycleMAE).
DomainNet results remain low (29% vs. 47% for CycleMAE+), showing that the approach doesn’t scale.
The additional 40% training cost for modest improvement makes the method impractical.

The comparisons omit stronger SSL baselines like DINOv2 or iBOT. Moreover, results for real-world OOD benchmarks (e.g., ImageNet-R, ImageNet-Sketch) are missing, which undermines the generalization claim.

### Questions
Why is the improvement on DomainNet so marginal despite the 40% higher computational cost?
How does the proposed “content direction” generalize beyond the [CLS] token — does it hold for dense feature maps?
Could similar disentanglement be achieved simply with an orthogonality regularizer or a linear projection layer without the heavy reconstruction pipeline?
Have you compared HyGDL against stronger modern baselines such as DINOv2 or SimMIM under equivalent pre-training budgets?

### Soundness
1

### Presentation
1

### Contribution
1
