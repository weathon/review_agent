# Bures-Isotropy Alignment: Manifold Learning of Generalized Category Discovery

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Generalized Category Discovery (GCD) seeks to discover categories by clustering unlabeled samples that mix known and novel classes. While the prevailing recipe enforces compact clustering, this pursuit is largely blind to representation geometry: it over-compresses token manifolds, distorts eigen-structure, and yields brittle feature distributions that undermine discovery. We argue that GCD requires not more compression, but geometric restoration of an over-flattened feature space. Drawing inspiration from quantum information science, which similarly pursues representational completeness, we introduce Bures-Isotropy Alignment (BIA), which optimizes the mini-batch class-token Gram toward an isotropic prior by minimizing the Bures distance. Under a mild trace constraint, BIA admits a practical surrogate equivalent to maximizing the nuclear norm of stacked class tokens, thereby promoting isotropic, non-collapsed subspaces without altering architectures. The induced isotropy homogenizes the eigen-spectrum and raises the von Neumann entropy, improving both cluster separability and class-number estimation. BIA is plug-and-play, implemented in a few lines on unlabeled batches, and generally boosts strong GCD baselines on coarse- and fine-grained benchmarks, improving overall accuracy and reducing errors in the estimation of class-number. By restoring the geometry of token manifolds rather than compressing them blindly, BIA supplies compactness for known classes and cohesive emergence for novel ones, advancing robust open-world discovery. Code is available at github.com/lytang63/BIA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper addresses a core challenge in the Generalized Category Discovery (GCD) task: characterizing geometric quality degradation. It proposes an innovative method called Bures-Isotropy Alignment. GCD aims to simultaneously identify known categories and discover novel categories from unlabeled data.

### Strengths
Instead of following mainstream approaches to further pursue feature compression, this paper astutely points out the fundamental bottleneck in current GCD research: the degradation of representational geometric quality. It offers a novel and insightful solution from the perspective of "restoring geometry" rather than "compressing features."

The paper successfully introduces the Bures metric from quantum information science into the field of machine learning and cleverly derives that minimizing it is equivalent to maximizing the nuclear norm of the feature matrix.

### Weaknesses
1. The paper emphasizes its connection to the literature on representation collapse, but could delve deeper into the specific differences and connections between BIA and other methods in self-supervised learning aimed at solving similar problems, thus more clearly defining its unique contribution.

2. The paper's core argument is "geometry recovery," but BIA is essentially still an optimization objective that indirectly "recovers" geometry by changing the optimization direction. A more in-depth discussion could explore how this optimization dynamically corrects the geometric structure during training, rather than just post-hoc statistical analysis.

3. The paper mentions that BIA has low computational overhead, but provides no quantitative data (such as the percentage increase in training time or changes in GPU memory usage) to support this claim.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a novel loss function, termed Bures-Isotropy Alignment (BIA) loss, which aims to enhance intra-class representation completeness, while maintaining inter-class separability in the context of Generalized Category Discovery (GCD)—a task involving clustering over both known and unknown categories. The BIA loss is inspired by the insight that when class-token covariance matrices, constructed from class-specific representations, approach an isotropic form, the learned representation space becomes more completed and less susceptible to collapse. Mathematically, the loss is formulated as the nuclear norm of stacked class representations based on the Bures-Isotropy alignment principle. Experimental results show that incorporating the BIA loss into baseline GCD models, *i.e.*, SelEX, SimGCD, CMS, SPTNet, consistently improves accuracy across various datasets and architectures, for both known and unknown categories. Furthermore, the method achieves superior performance in category number estimation, reinforcing its overall effectiveness.

### Strengths
- The motivation and insight that encouraging class-token covariance to become isotropic leads to more robust within-class representations is both intuitive and compelling. The theoretical support using Bures-Isotropy alignment (Section 3.3) is mathematically well-grounded and strengthens the conceptual foundation.

- The process of deriving a simple and practical loss formulation from this motivation is clear and elegant, making the approach easily applicable to future work in the GCD research community.

- The experimental design is thorough and effectively demonstrates both the simplicity and generality of the proposed method.

- Incorporating the BIA loss consistently leads to improved performance in Generalized Category Discovery tasks and category number estimation across various datasets and baseline models, highlighting the robustness and versatility of the approach.

### Weaknesses
I do not have any major concerns with this submission; however, I have several minor suggestions for clarification and completeness.

- In Section 3.4, the paper shows that incorporating the BIA loss during training improves Neumann Entropy, which measures the uniformity of class-specific representation distributions. However, since both BIA and Neumann Entropy are defined in a similar manner—being proportional to the eigenvalues—the observed positive correlation between the two is somewhat expected. It would be helpful to clarify what specific insight or takeaway this section intends to convey beyond confirming this relationship.

- Section 5.4 presents interesting observations, but without sufficient details about the compared methods (e.g., CorInfoMax and VICReg), it is difficult to fully understand or assess the comparison. Please provide additional details or the design motivations of these methods to improve clarity.

- Figure 1 is currently not referenced anywhere in the manuscript. It should be mentioned and discussed in an appropriate place to ensure proper integration into the paper’s narrative.

### Questions
**Is the proposed BIA loss differentiable?** 

In other words, is the process of computing eigenvalues from the covariance matrix differentiable? It appears that it should be, since (i) both matrix decomposition and squared eigenvalue operations, involved in loss calculation, have known derivative forms, and (ii) the paper reports successful training results using this loss. 

I would appreciate clarification on whether the authors have a direct explanation or confirmation regarding the differentiability of the BIA loss.

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
3

### Summary
The paper introduces Bures–Isotropy Alignment (BIA) for generalized category discovery (GCD). It measures the Bures distance from the batch class-token Gram matrix to the identity. BIA improves accuracy and class-count estimation across multiple baselines and datasets.

### Strengths
The paper reframes GCD from “tighter clusters” to restoring isotropy of class-token geometry, and offers a plug-and-play, architecture-agnostic regularizer that targets dimensional collapse and stabilizes the feature spectrum.

### Weaknesses
1. You keep saying the equivalence holds “under a mild trace constraint” and “once row norms are stabilized with LayerNorm,” but the paper never pins down what that actually means. Please spell out the exact assumptions, edge cases where the surrogate can drift from the metric, and give at least a short proof sketch so readers know when it’s safe to use.

2. Compute cost is called “negligible,” and the setup mentions 4×RTX4090, but there are no numbers. Report wall-clock time per epoch, peak GPU memory, and the stability of SVD or PD-root operations as $B$ and $d$ grow, plus failure rates if any. Without that, it’s hard to judge practicality.

3. Section 3.2 builds geometry only on stacked [cls] tokens with $\Sigma_B = ZZ^\top$. That leaves patch-level interactions on the table. For fine-grained tasks, this likely matters. Either show ablations that use patch tokens or argue why a class-token-only view is sufficient. 

4. Gains are not uniform. With unknown $K$ in Table 3, some baselines drop when adding BIA, e.g., CMS on CIFAR100-New −1.6 and Herbarium19-New −1.0. Please analyze when BIA helps and when it hurts, and include stress tests for noise and class imbalance.

5. Robustness ablations are thin. Figure 4 only varies $\lambda$ and feature dimension $D$. Batch size $B$, within-batch class imbalance, and pseudo-label noise all directly change the batch Gram $\Sigma_B$ but are not studied. These should be evaluated.

6. Table 5 compares “representative isotropic feature distribution schemes,” but the theoretical relation to decorrelation/whitening objectives remains informal.

### Questions
1. When does $d_B^2(\Sigma_B,I)\downarrow \Longleftrightarrow |Z|_*\uparrow$ fail?

2. What happens if BIA uses feature-Gram $Z^\top Z$ or includes patch tokens (e.g., pooling or grouped variants) instead of only $[\mathrm{cls}]$? Any trade-offs in accuracy and cost on fine-grained datasets?  

3. Analyze the regressions in Table 3. Are they linked to class imbalance, pseudo-label noise levels, or $K$-estimation errors in CMS?

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
4

### Summary
The paper tackled the setting of generalised category discovery (GCD). It proposes to add a generic loss on the CLS token to maximise isotropy across the batch in the embeddings. The paper shows this can be combined with several exists GCD works to yield fairly consisten improvements across standard GCD benchmarks.

### Strengths
* the method is extremely simple and only has one hyperparameter
* the improvements are consistent
* theoretical analysis is provided and insightful

### Weaknesses
* Figure 1 does not really provide any insight as it remains very high-level.
* Same for Figure 2, it does not provide any new insights
* The idea seems to be very related to VICReg and TIan's uniformity loss. Yet both of these relevant works from reprsentation learning literature are only briefly compared to in Sec 5.4. It's unclear what the setting of choosing (the extremely important!) hyperameters for these experiments were. Note that even if "+VICReg" performed better than the proposed "+bures", it would still be insightful for the ICLR community that an isotropy regulariser seems to be missing for GCD. 
* Similarly it would be good to ablate against other isotropy-encouraging regularisation losses and the dependency on the batchsize is also not analysed
* the hypothesis of "poor eigen-structure, skewed energy distribution, and fragile decision regions, that ultimately impedes category discovery and class-number estimation." is mentioned in the introduction as a key motivation but not quantitatively shown.
* the explanation for improved performance seems to not be super clear. if full-rank manifolds lead to better performance, just using random matrices (which is full rank) should yield good performance.

### Questions
* Why do other isotropy regularisation objective not work as well for the most part? 
* What is the dependency of batch size?
* Why does the method not yield gains on Herbarium?

### Soundness
3

### Presentation
3

### Contribution
3
