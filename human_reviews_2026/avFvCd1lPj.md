# Neural Collapse by Design: Learning Class Prototypes on the Hypersphere

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Neural Collapse (NC) describes the global optimum of supervised learning, yet standard cross-entropy (CE) training rarely attains its geometry in practice. This is due to unconstrained radial degrees of freedom: cross-entropy is invariant to joint rescaling of features and weights, leaving radial directions underconstrained thus preventing convergence to a unique geometry. We show that constraining optimization to the unit hypersphere removes this degeneracy and reveals a unifying view of normalized softmax classifier learning (CL) and supervised contrastive learning (SCL) as the same prototype-contrast principle: both optimize angular similarity to class prototypes, using explicit learned weights for normalized softmax and implicit class means for SCL. Despite this shared foundation, existing objectives suffer from small effective negative sets and interference between positive and negative terms, which slows convergence to NC. We address these issues with two objectives: NTCE, which contrasts class prototypes against all batch instances to expand the negative set from K classes to M samples; and NONL, which normalizes only over negatives to decouple intra-class alignment from inter-class repulsion. Theoretically, we prove that SCL already learns an optimal prototype classifier under NC, eliminating the need for post-hoc typically hours-scale linear probing. Empirically, across four benchmarks including ImageNet-1K, our methods surpass CE accuracy, reach $\ge$95\% on NC metrics, and match NC structure with substantially fewer iterations. Moreover, SCL with class-mean prototypes matches linear-probing accuracy while requiring no training. These results reframe supervised learning as prototype-based classification on the hypersphere, closing the theory–practice gap while simplifying training and accelerating convergence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies why standard supervised learning with cross-entropy (CE) does not always reach the Neural Collapse (NC) geometry, even though NC represents an ideal solution for many classification objectives. The authors point out that the main reason is the unconstrained radial degree of freedom in the feature space.
To solve this issue, they propose training on a unit hypersphere, which naturally removes this degree of freedom. The paper also shows that normalized softmax and supervised contrastive learning (SCL) can be unified as prototype-based contrastive methods under this geometric view.
Based on this, the authors propose two new losses: NTCE (Normalized Temperature-scaled Cross Entropy) and NONL (Negatives-Only Normalization Loss). Both encourage class prototypes to form an orthogonal structure on the hypersphere. Experiments on CIFAR, ImageNet-100, and ImageNet-1K show faster convergence to NC and small but consistent accuracy gains over CE and NormFace.

### Strengths
- The paper gives a strong explanation of why CE fails to reach the NC geometry and how the hyperspherical constraint solves this problem.
- The connection between normalized softmax and SCL is well presented and helps understand these methods in a common theoretical view.
- The proposed NTCE and NONL are easy to implement and directly build on existing normalized contrastive losses.
- The paper evaluates on multiple datasets, compares to CE and SCL, and provides metrics showing faster NC convergence and better class separation.
- The paper is clearly organized, and the results are well illustrated with understandable figures.

### Weaknesses
- While NC metrics improve strongly, the actual classification accuracy only improves slightly. The practical advantage may not be very large.
- Because class imbalance is known to affect prototype geometry and Neural Collapse dynamics, it would be valuable to see results on imbalanced datasets.

### Questions
Many recent works on Neural Collapse and prototype geometry have extended these ideas to self-supervised or semi-supervised frameworks. It would be interesting to know if the proposed NTCE and NONL objectives can also be applied in these areas. Would the geometric properties (e.g., orthogonal prototype structure or faster Neural Collapse) still emerge when label supervision is only partial or noisy?

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
5

### Summary
The paper trains classifier on the unit hypersphere to avoid the radial issues of standard CE. This view ties normalized softmax classifier learning (CL) and supervised contrastive learning (SCL) into one prototype-contrast idea. The authors add two losses that grow the negative set and separate “pull” vs. “push,” aiming to reach Neural Collapse (NC) faster. In experiments, they beat CE/NormFace on accuracy and show quicker NC convergence.

### Strengths
- Paper cleanly shows how normalized softmax CL and SCL are the same prototype-contrast story on the hypersphere, fixing CE’s radial problem.
- Proposed objectives are drop-in and, in practice, speed up NC and often improve accuracy over CE/NormFace.
- Using class-mean prototypes can replace linear probing with similar accuracy and no extra training step.
- Comprehensive convergence analysis .

### Weaknesses
- Missing several recent NC/hypersphere/prototype methods (e.g., fixed/orthogonal/ETF heads, decoupled contrastive) and simple nearest-centroid. Please compare with [1]–[4]. If these works are not relevant, please explain why.
[1]Yang, Yibo, et al. "Inducing neural collapse in imbalanced learning: Do we really need a learnable classifier at the end of deep neural network?." Advances in neural information processing systems 35 (2022): 37991-38002.
[2]Mettes, Pascal, Elise Van der Pol, and Cees Snoek. "Hyperspherical prototype networks." Advances in neural information processing systems 32 (2019).
[3]Shen, Yang, Xuhao Sun, and Xiu-Shen Wei. "Equiangular basis vectors." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[4]Wang, Wenguan, et al. "Visual Recognition with Deep Nearest Centroids." The Eleventh International Conference on Learning Representations.

-The intro ties NC to better transfer learning, robustness, and generalization, but these are not evaluated.

- Experiments use ResNet-18/50. Please add larger CNNs (ResNet-101/152) and ViT-B/L to support the application.

### Questions
Please see the weaknesses. Also, one key contribution is comparing prototypes to samples within a batch (instead of the other way around). What is the computational cost as the batch size or the number of classes/prototypes grows? I will also read the other reviews. (my rating is for the current state of the paper)

### Soundness
4

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
The paper argues that standard cross-entropy rarely reaches the optimal Neural Collapse geometry because of unconstrained radial degrees of freedom. It proposes constraining both features and classifier weights to the unit hypersphere, which yields a unified prototype-contrast view bridging normalized classifier learning and supervised contrastive learning. To mitigate small negative sets and alignment–uniformity coupling, the authors introduce NTCE, which expands the denominator to M in-batch instances, and NONL, which normalizes over negatives only. Empirically, these objectives reach NC structure and metrics much faster while maintaining accuracy. The paper further shows, under unit-norm and balanced-label assumptions, that SCL already learns optimal class-mean prototypes, so replacing linear probing with class-mean classifiers can match linear-probe accuracy without extra training. The paper also discusses limitations for NONL at large numbers of classes and when per-batch class coverage is sparse.

### Strengths
1. It provides a clear and intuitive diagnosis of why standard cross-entropy training rarely reaches the theoretical Neural Collapse, namely, unconstrained radial degrees of freedom;
2. It offers an elegant unifying view that places normalized CE softmax and supervised contrastive learning within a single prototype contrast principle on the unit hypersphere;
3. Under unit norm and balanced label assumptions, it shows that class mean prototypes learned by supervised contrastive learning can replace linear probing while preserving accuracy and saving the hours of computation typically spent on that phase;
4. The proposed NTCE and NONL objectives reach the target Neural Collapse geometry much faster than strong baselines when progress is measured by time to $95$% NC thresholds.

### Weaknesses
## Major
### The benefit is unclear
The paper is premised on the assumption that enforcing neural collapse is causally beneficial. It assumes that because NC was observed in late-stage training (as reported in the original NC paper), one can design an objective to force this geometry, thereby improving generalization or robustness. However, the paper never validates this causal link. It does not test the alleged benefits of NC (e.g., out-of-distribution robustness, adversarial resilience). All evaluations remain within standard in-distribution classification and a set of NC metrics. At present, the paper only shows that its method produces representations that look more collapsed, not that this geometric property is what makes the model better.

### The  stated gap "unconstrained radial degrees of freedom" and the actual method are not fully aligned
The introduction frames the problem as a geometric one: standard cross-entropy suffers from unconstrained radial degrees of freedom. The proposed solution, however, does not stop at normalization (which would fix this). The paper's core technical contributions—NTCE (using in-batch instances as negatives) and NONL (decoupling positive/negative terms)—primarily address contrastive-style optimization inefficiencies (i.e., small negative sets and alignment-uniformity coupling). These are optimization choices, not direct solutions to the stated geometric problem of radial degeneracy. The paper, therefore, starts from a geometric motivation but delivers a recipe that is much closer to standard contrastive learning practice.

### Limited scalability of NONL
The paper's own results show that NONL fails to scale to large-K settings like ImageNet-1K, where its performance drops below the baseline. The authors attribute this to gradient imbalance: when many classes are absent from a batch, their prototypes ($w_c$) only receive negative gradients (repulsion) without any positive signal (alignment). This reliance on sufficient per-batch class coverage is a structural limitation, not a minor implementation detail. This limitation makes it difficult to claim NONL as a general-purpose, large-scale method.

### More negative instances are better is not verified
The core change in NTCE is to replace the denominator's $K$ learned class prototypes (the $\hat{w}_j$ in NormFace) with $M$ in-batch instances (the $u_j$ in NTCE). The paper implicitly assumes that this larger set of $M$ negatives leads to a better contrastive estimate. This assumption conflates the quantity and quality of negatives. The $K$ prototypes are learned, “expert” hard negatives that define the decision boundary. The $M$ random instances are likely dominated by easy negatives. The paper provides no ablation to compare the effect of 'few learned hard negatives' vs. 'many random instance negatives.' Therefore, the superiority of this design choice is an unverified claim.

## Minor
### Inconsistent notation
The paper's notation is inconsistent. On page 3, it states that $W \in \mathbb{R}^{K \times h}$ and that $w_j$ denotes the j-th row of $W$ (a $1 \times h$ vector). However, Equation (1) defines the logit as $z_i^\top w_j$. Given that $z_i$ is an $h \times 1$ feature vector, $z_i^\top$ is a $1 \times h$ row vector. The operation $z_i^\top w_j$ (a $[1 \times h] \times [1 \times h]$ operation) is thus mathematically undefined. This should presumably be $w_j z_i$ (if $w_j$ is a row) or $w_j^\top z_i$ (if $w_j$ is a column). While likely a typo, this inconsistency reduces confidence in the paper's formal precision.

### The main novelty is mostly a unifying reformulation
The paper's strongest contribution is conceptual: it places normalized softmax (CL) and supervised contrastive learning (SCL) into a single, unified 'spherical prototype-to-instance' framework. This is an elegant reformulation. However, both CL on the hypersphere and SCL are, by themselves, well-studied topics. The paper's technical additions are two loss variants (NTCE, NONL) that import ideas from one framework to the other. The actual algorithmic novelty is therefore modest, with the primary contribution being the unified narrative itself.

### Questions
- What concrete evidence do you have that enforcing neural collapse is causally beneficial for robustness or out-of-distribution generalization?
- What is the task scope? Under what conditions or task families would enforcing NC1 be counterproductive (e.g., detection/segmentation/pose)?
- Why prefer many random instance negatives over, for example, a few learned hard negatives (class prototypes or mined hard examples)?
- Better NC does not mean better performance everywhere. In what condition should this method not be used or require modifications?
- To disentangle your contributions, have you considered a full factorial ablation? Your sequential path (Normalized CE $\rightarrow$ NTCE $\rightarrow$ NONL) confounds the denominator enlargement (to $M$ instances) with positive/negative decoupling (NONL). Have you tested a "Decoupled NormFace", applying Negatives-Only Normalization directly to NormFace's $K$-class denominator, to isolate the contribution of decoupling alone?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper is motivated by prior studies suggesting that leveraging the properties of Neural Collapse can improve generalization, by the similarity between cross-entropy loss and supervised contrastive learning, and by the observation that features learned through supervised contrastive learning also form a simplex ETF structure even without linear probing. Based on these insights, the authors propose a modified cross-entropy loss that operates at the sample level within each mini-batch, along with an additional variant of this modified loss. Through experiments, they empirically demonstrate the effectiveness of the proposed approach and its improvement in exhibiting Neural Collapse properties.

### Strengths
- Motivated by the previously observed similarity between cross-entropy loss and supervised contrastive learning [R1], this paper addresses an interesting problem: why conventional classification models fail to exhibit Neural Collapse in practical settings.

- To address this issue, the authors draw inspiration from prior studies on Neural Collapse and its effectiveness in improving generalization across various tasks, and propose a simple yet intuitive modified cross-entropy loss.

- The proposed method is validated through performance comparisons on image classification tasks, and further empirical analysis of the Neural Collapse properties demonstrates that the proposed loss successfully enhances intra-class alignment as intended.

**_reference_**

[R1] Graf et al., Dissecting Supervised Contrastive Learning, ICML 2021

### Weaknesses
**W1 (Novelty)**. Although proving the equivalence between SCL and the prototype-softmax minimizer through Theorem 4.1 is interesting, the proposed use of class-mean prototypes in place of linear probing is not novel. (See *NC3-inspired classifier* [R2])

**W2 (Gradient Analysis of NTCE)**. The proposed method NTCE treats class vectors as samples and, conversely, treats other samples within a mini-batch as class vectors when constructing the loss function. However, an analysis of how the NTCE affects the gradient has not been conducted. Although the authors provide their interpretation and an indirect analysis of the experimental results in (lines 352-357), a more detailed examination is required to make the contribution of the proposed approach more robust

**W3 (Theoretical Analysis of Neural Collapse)**. Although the modified version of the cross-entropy is proposed, this paper only provides empirical verification of the Neural Collapse properties. A corresponding theoretical model (e.g., layer-peeled model (LPM) [R3], unconstrained features model (UFM) [R4]) is not presented, nor is a theoretical analysis of Neural Collapse conducted based on it.

**W4 (Marginal Improvement)**. The performance improvement achieved by the proposed approach is marginal, which raises doubts about whether the research question posed in (lines 46-47) truly addresses an important problem. Moreover, among the datasets used in the experiments, ImageNet-1K—which is the closest to real-world settings—shows almost no performance difference compared to the baseline, and in the case of NONL in Table 1, even lower performance than the baseline. This weakens the evidence supporting the generalization ability of the proposed method.

**_reference_**

[R2] Zhang et al., Neural Collapse Inspired Knowledge Distillation, AAAI 2025

[R3] Fang et al., Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training, PNAS 2021

[R4] Mixon et al., Neural collapse with unconstrained features, SaSiDa 2021

### Questions
**Q1**. Since the proposed method is based on supervised contrastive learning (SCL), wouldn’t it also suffer from batch-size-related issues?

**Q2** (w.r.t **W2**). In the original softmax cross entropy (SCE), the model learns through a push-and-pull effect between samples and class vectors [R4]. However, if the roles of classes and samples are swapped, the pull effect in SCE would change—would this not affect learning? In my opinion, it likely does because the class vectors corresponding to classes that do not appear within a mini-batch do not receive any gradient updates. The impact may become more pronounced as the number of classes increases, since the frequency of each class appearing within a mini-batch decreases, thereby weakening the inter-class separation effect. This might also explain why NTCE > NONL in ImageNet-1K of Table 1.

**Q3** (w.r.t **W3**). How is the LPM (or UFM) analysis conducted under the proposed loss? Neural Collapse was originally observed under conventional SCE, where theoretical analysis showed that NC properties hold when global optimality is achieved in LPM. When other losses (e.g., MSE loss) are used, theoretical verification was also conducted using this framework [R5]. Does the proposed loss satisfy these conditions? Moreover, if it is unclear whether NC actually occurs under the proposed loss, is it valid to empirically evaluate it using NC-based metrics?

**Q4** (w.r.t **W4**). As shown in Table 2, while CE exhibits lower values than the proposed method on other NC metrics, it demonstrates the opposite trend on *Information Theory Metrics*. Doesn’t this suggest that CE has no inherent issue in learning the information necessary for classification and is, in fact, superior to other methods in this regard? If so, is it truly necessary to enforce Neural Collapse in practical classification tasks?

**_reference_**

[R5] Han et al., Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path, ICLR 2022

### Soundness
2

### Presentation
3

### Contribution
2
