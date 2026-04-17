# Towards Understanding The Calibration Benefits of Sharpness-Aware Minimization

- Decision: Accept (Poster)
- Scores: 2, 4, 8, 4

## Abstract
Deep neural networks have been increasingly used in safety-critical applications such as medical diagnosis and autonomous driving. However, many studies suggest that they are prone to being poorly calibrated and have a propensity for overconfidence, which may have disastrous consequences. In this paper, unlike standard training such as stochastic gradient descent, we show that the recently proposed sharpness-aware minimization (SAM) counteracts this tendency towards overconfidence. The theoretical analysis suggests that SAM allows us to learn models that are already well-calibrated by implicitly maximizing the entropy of the predictive distribution. Inspired by this finding, we further propose a variant of SAM, coined as CSAM, to ameliorate model calibration. Extensive experiments on various datasets, including ImageNet-1K, demonstrate the benefits of SAM in reducing calibration error. Meanwhile, CSAM performs even better than SAM and consistently achieves lower calibration error than other approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates why Sharpness-Aware Minimization (SAM) tends to produce better-calibrated neural networks than standard optimizers such as SGD and AdamW.
The authors provide theoretical analysis showing that SAM implicitly regularizes the negative entropy of the predictive distribution, similar to focal loss, and thus discourages over-confident predictions.
Building on this, they propose a simple variant Calibrated SAM (CSAM), which re-weights over-confident examples by modifying the SAM outer loss.
Experiments on CIFAR-10/100 and ImageNet-1K demonstrate that (1) SAM improves Expected Calibration Error (ECE) compared with SGD and common post-hoc methods, and (2) CSAM further improves calibration without harming accuracy.

### Strengths
1. Provides a clean theoretical link between sharpness minimization and confidence calibration.
2. Solid empirical confirmation that SAM reduces calibration error across multiple architectures.
3. The CSAM variant is simple to implement and empirically effective.
4. Clear exposition and well-structured proofs.

### Weaknesses
1. The core theoretical insight—that SAM implicitly maximizes predictive entropy—is conceptually close to prior entropy-regularization or focal-loss analyses; novelty is limited.
2. The proposed CSAM is a minimal heuristic extension rather than a principled optimization method.
3. Missing systematic comparisons against other calibration-oriented training losses such as label smoothing. Although table3 is given but the setting is very limited. Author should also test if the CSAM be compatible with other training loss.
4. Analysis of how calibration evolves during training (claimed in Appendix D) is not deeply discussed in the main text.

### Questions
Would the effect persist if temperature scaling or focal loss were combined with SAM?

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
This study primarily explores the relationship between the probability-based loss output at the point of maximum loss within the perturbation radius 𝜌 and the loss-to-probability mapping under the current model parameters. This analysis establishes a theoretical connection between model calibration and Sharpness-Aware Minimization (SAM), elucidating their underlying mechanisms. Building upon these findings, the authors propose an improved method aimed at enhancing model calibration.

### Strengths
1. Both SAM and model calibration represent important research directions, and exploring their mutual influence holds significant scientific value.
2. The study demonstrates strong completeness by beginning with theoretical foundations to explain their respective roles and subsequently extending the analysis to methodological applications.

### Weaknesses
1. Although Theorems 1, 2, and 3 are presented, they largely reflect variations of similar concepts. Consequently, the theoretical framework appears somewhat repetitive and lacks sufficient depth.
2. The methodological contribution is not particularly innovative, as it mainly combines the existing SAM framework with focal loss. Thus, the degree of novelty—especially from a methodological perspective—may be limited in its ability to inspire readers.

### Questions
1. Table S4 shows improved calibration results when 𝛾=0. Does this suggest that the proposed enhancement does not significantly improve calibration performance? At  𝛾=0, the method reduces to the standard SAM framework.
2. The conclusion states: “We proved that SAM achieves this goal by imposing an implicit regularization on the negative entropy of the predictive distribution during training, which is similar to focal loss.” Could the authors clarify which mathematical expression supports this statement? No corresponding equation appears in the theoretical derivations or references. A more detailed explanation would be appreciated.
3. Since the proposed method is derived from the theoretical analysis, can the assumptions established in the theory be satisfied in practical applications? In other words, do the conclusions derived under these assumptions hold in real-world optimization scenarios?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the calibration benefits of Sharpness-Aware Minimization (SAM) in deep neural networks. The authors provide a theoretical analysis showing that SAM implicitly regularizes the predictive distribution by encouraging higher entropy, thereby mitigating overconfidence. Specifically, they derive a lower bound and building on this insight they propose Calibrated SAM (CSAM), a variant that further enhances calibration by adaptively reweighting overconfident predictions. Extensive experiments on CIFAR, ImageNet-1K, and corrupted datasets demonstrate that both SAM and CSAM achieve significantly lower Expected Calibration Error (ECE) than standard training and existing calibration methods, without sacrificing accuracy.

### Strengths
(1) The paper’s key strength lies in its originality and significance: it provides the first formal theoretical explanation for why Sharpness-Aware Minimization (SAM) improves calibration—linking it to implicit entropy maximization—offering a principled understanding beyond empirical observation. This insight bridges optimization geometry and uncertainty quantification, a valuable contribution to both communities. 

(2) The proposed CSAM variant is a high-quality and practical extension that consistently improves calibration across diverse architectures and datasets, including ImageNet-1K and distribution-shifted settings, without sacrificing accuracy. 

(3) The paper is also clearly written, with well-structured theory, intuitive figures (e.g., reliability diagrams, Hessian analysis), and thorough experiments that effectively support its core message.

### Weaknesses
(1) Missing summation symbol in the ECE estimator (Lines 177–178). The definition of bin accuracy in the Expected Calibration Error (ECE) computation omits the summation over samples in bin. I think the correct expression should be  $\text{acc}(B_i) = \frac{1}{|B_i|} \sum_{z_j \in B_i} \mathbb{I}[y_j = \arg \max f_\theta(x_j)]$.

(2) The theoretical analysis (Lemma 1 and Theorem 1) relies on an assumption about the lower bound of the smallest Hessian eigenvalue along the linear interpolation between $\theta$ and $\theta'$. However, this assumption is only verified on a single model (ResNet-56) and a single dataset (CIFAR-10) in Figure 2. The generality of this assumption—and thus the applicability of the theoretical conclusions—remains unclear for other architectures (e.g., ViTs) or more complex datasets (e.g., ImageNet).

### Questions
(1) The assumption on the Hessian’s smallest eigenvalue (Lemma 1) is only validated for ResNet-56 on CIFAR-10 (Figure 2). Given that the theoretical conclusions rely on this assumption, do the authors have evidence—e.g., from preliminary experiments on ImageNet or ViTs—that this behavior (linear decay of  $k_{min}$ along $\theta$ to $\tilde{\theta}$) holds more broadly? If not, could they discuss potential failure modes where the assumption breaks down and how that might affect SAM’s calibration benefits?

(2) In the ECE estimator (Section 3.1, lines 177–178), the expression for $\text{acc}(B_i)$ appears to miss a summation over samples in the bin. Is this a typographical error? If yes, could it be corrected for formal precision?

### Soundness
2

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
2

### Summary
This paper studies why Sharpness-Aware Minimization (SAM), beyond its known generalization benefits, tends to produce better-calibrated models. The authors show theoretically that SAM implicitly maximizes the entropy of the predictive distribution, discouraging overconfident outputs. Thus, they propose Calibrated SAM (CSAM), a simple variant that further suppresses overconfident examples through a lightweight modification to the loss. Experiments on CIFAR-10/100 and ImageNet-1K confirm that SAM already improves calibration compared to SGD or AdamW, and CSAM achieves the lowest Expected Calibration Error (ECE) without sacrificing accuracy.

### Strengths
## Strengths

* This paper provides a simple and practical variant. CSAM requires only a minor change and no additional computation, yet consistently improves ECE.  
* The experiments are broad, covering both in- and out-of-distribution evaluations across convolutional and transformer-based models.  
* The finding that SAM alone can outperform post-hoc calibration methods like temperature scaling is practically valuable for reliability-sensitive applications.  
* The paper gives a clear mathematical explanation of why SAM tends to yield better-calibrated predictions, connecting it to implicit entropy regularization.

### Weaknesses
## Weaknesses

* The empirical effect has been reported before. The observation that SAM improves calibration is not new—Zheng et al. (2021) showed similar improvements in long-tailed recognition, and the original SAM paper (Foret et al., 2021) also mentioned better reliability qualitatively.
* The theoretical assumptions are strong. The derivations rely on smoothness and bounded-Hessian assumptions, which may not fully hold for complex networks, limiting the generality of the proofs.
* While CSAM consistently reduces calibration error, its advantage over SAM is small.

**References**

* Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR, 2021.  
* Zheng, Z., Yang, X., Wang, Y., Li, Z., Liu, Y., & Zhang, T. Improving Calibration for Long-Tailed Recognition. CVPR, 2021.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
