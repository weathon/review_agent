# DRIFT: Divergent Response in Filtered Transformations for Robust Adversarial Defense

- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
Deep neural networks remain highly vulnerable to adversarial examples, and most defenses collapse once gradients can be reliably estimated. We identify \emph{gradient consensus}—the tendency of randomized transformations to yield aligned gradients—as a key driver of adversarial transferability. Attackers exploit this consensus to construct perturbations that remain effective across transformations. We introduce \textbf{DRIFT} (Divergent Response in Filtered Transformations), a stochastic ensemble of lightweight, learnable filters trained to actively disrupt gradient consensus. Unlike prior randomized defenses that rely on gradient masking, DRIFT enforces \emph{gradient dissonance} by maximizing divergence in Jacobian- and logit-space responses while preserving natural predictions. Our contributions are threefold: (i) we formalize gradient consensus and provide a theoretical analysis linking consensus to transferability; (ii) we propose a consensus-divergence training strategy combining prediction consistency, Jacobian separation, logit-space separation, and adversarial robustness; and (iii) we show that DRIFT achieves substantial robustness gains on ImageNet across CNNs and Vision Transformers, outperforming state-of-the-art preprocessing, adversarial training, and diffusion-based defenses under adaptive white-box, transfer-based, and gradient-free attacks. DRIFT delivers these improvements with negligible runtime and memory cost, establishing gradient divergence as a practical and generalizable principle for adversarial defense.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DRIFT (Divergent Response in Filtered Transformations), a new adversarial defense that breaks the tendency of randomized defenses to share aligned gradients exploitable by adaptive attacks. DRIFT inserts a small ensemble of learnable, differentiable filters before a frozen classifier and trains them to maximize gradient divergence in Jacobian and logit space while maintaining clean accuracy. The authors theoretically link gradient consensus to adversarial transferability and design a joint objective combining prediction consistency, Jacobian separation, logit-VJP separation, and adversarial robustness losses. Experiments on ImageNet with CNNs and Vision Transformers show that DRIFT achieves higher robustness under PGD, AutoAttack, BPDA, and EOT, outperforming baselines like JPEG, BaRT, DiffPure, and adversarial training. Sanity checks confirm no gradient masking, and runtime analysis shows orders-of-magnitude efficiency gains. Overall, DRIFT presents a lightweight, theoretically motivated, and empirically strong defense principle based on gradient divergence rather than masking or purification.

### Strengths
1.The paper introduces gradient consensus as a measurable cause of adversarial transferability and proposes gradient divergence as a defense principle. This framing is intuitive, theoretically motivated, and distinct from existing randomization or purification methods.

2.DRIFT’s learnable residual filters provide a lightweight, differentiable, and architecture-agnostic front-end, requiring no retraining of the backbone. The method is computationally efficient (0.4 ms per image), a clear advantage over diffusion-based defenses.

3.The experiments cover both CNNs and Vision Transformers on ImageNet, testing against diverse adaptive attacks. Results demonstrate consistently higher robustness while maintaining clean accuracy.

4.The inclusion of gradient-norm sanity checks, finite-difference validation, and loss-surface visualization shows the authors are aware of gradient masking pitfalls and have rigorously verified true robustness.

### Weaknesses
1.The link between consensus and transferability (Theorem 3.5) is intuitive but lacks formal rigor; constants are unspecified and empirical validation of consensus metrics (ρ) is missing.

2.Ablation results show improvements, but the contribution of each loss component (LJS vs LLVJP) is not deeply analyzed; no visualization or metric demonstrates the intended gradient decorrelation effect.

3.Comparisons to adversarial training and diffusion-based methods appear under-tuned, and evaluation is limited to ImageNet only, lacking smaller, more reproducible datasets.

### Questions
The paper claims robustness under BPDA and EOT, but the attack strength (number of EOT samples, surrogate choice) is not detailed. Please include robustness curves under stronger settings and alternative BPDA surrogates.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new stochastic adversarial defense method called DRIFT. It identifies gradient consensus, i.e., the alignment of gradients across stochastic transformations, as the root cause of adversarial transferability. DRIFT combats this by learning a small ensemble of lightweight, differentiable preprocessing filters trained to maximize gradient divergence while maintaining clean accuracy. Theoretical analysis links low gradient consensus to reduced transferability. The training loss combines cross-entropy, Jacobian separation, logit-VJP separation, and adversarial robustness terms. Experiments on ImageNet with CNNs and Vision Transformers show substantial robustness gains over adversarial training, diffusion-based purification, and randomized transformations, under both adaptive (BPDA, EOT) and non-adaptive attacks, with negligible computational overhead.

### Strengths
1. This paper reframes adversarial robustness around gradient divergence rather than obfuscation or purification. This is a fresh and intellectually elegant perspective.
2. This paper provides a simple but formal link between gradient alignment and attack transferability, grounded in the geometry of loss gradients.
3. This method applies across CNNs and Vision Transformers without modifying the backbone.
4. This paper is evaluated under strong adaptive attacks (BPDA + EOT, AutoAttack) and validated with sanity checks (finite-difference tests, loss landscape visualization).
5. This paper achieves 10⁴× faster inference than diffusion-based defenses, with minimal memory overhead.
6. Presentation and structure of this paper are clear.

### Weaknesses
1. The analytical part (Lemma 3.4–Theorem 3.5) is intuitive but lacks full proofs or empirical validation of constants; thus, it serves more as motivation than as a formal guarantee.
2. No certified guarantees (unlike randomized smoothing), so the results, though strong, remain empirical.
3. Filters are trained on a subset of ImageNet validation data—unclear if robustness generalizes across distribution shifts or unseen domains.
4. The method assumes a high-quality pretrained base; how it performs on weaker or adversarially trained backbones is unclear.
5. While the method ensures gradient divergence, it does not analyze how filters achieve this (e.g., frequency-domain or spatial patterns).
6. Effects of ensemble size and loss weighting are reported briefly in appendices but could be expanded in the main text.

### Questions
1. How does robustness scale with the number of filters? Is there diminishing return or optimal ensemble size beyond n = 4?
2. Could DRIFT generalize to non-image domains (e.g., NLP, tabular, or time-series data)?
3. How sensitive are results to the loss weights (β_JS, β_LVJP, λ)?
4. If the base model is already adversarially trained, does DRIFT still provide gains, or is its effect redundant?
5. Can the authors show qualitative examples of learned filters to interpret what kind of transformations disrupt gradient consensus?
6. Could DRIFT be combined with randomized smoothing to yield both empirical and certified robustness?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors propose a defense method based on learnable filters. To mitigate the issue of gradient consistency, which makes models vulnerable to transfer attacks, the method reduces the transferability of adversarial examples across different filters while enhancing filter diversity for defense. Additionally, lightweight residual convolutional blocks are introduced within each filter to ensure that they can learn meaningful transformations. Experiments conducted on multiple CNN and ViT architectures using the ImageNet benchmark demonstrate that the proposed learnable filter ensemble method achieves strong defensive performance compared with baseline approaches.

### Strengths
1. The authors propose using a Jacobian separation loss to reduce transferability across filters, along with a logit-space separation loss to enhance filter diversity.

2. To improve filter performance and enable meaningful feature transformations, the authors design residual modules for each filter.

3. Experimental results on CNN and ViT models demonstrate that the proposed learnable random filter ensemble defense outperforms traditional defense methods.

### Weaknesses
1. Theoretical details are insufficient, as the derivations of the inequalities in Lemma 3.4 and Theorem 3.5 are not provided.

2. The total loss comprises four components. However, it remains unclear how the contributions of each loss are balanced during filter training and how the optimal values of the associated hyperparameters are determined.

3. The overall training procedure of the DRIFT filters is not clearly explained. For instance, Algorithm 1 includes a warm-up stage, separates the optimization of individual losses, and maximizes $L_{adv}$ over the index $i$. However, the paper lacks sufficient details regarding the training process and implementation.

4. The experimental comparison is not comprehensive. The paper lacks evaluations against more recent defense methods and does not consider evaluations against the latest attack techniques. Furthermore, some mentioned methods (e.g., ANF, FFR) are not compared in Table 1, which limits the completeness of the comparative analysis.

5. In Section 6.2, the choice of perturbation size (4/255) is not explained, and no experiments are provided to compare different perturbation magnitudes. 

6. The paper lacks ablation studies on the hyperparameters of the proposed loss function to analyze their impact on the experimental results.

7. In Section 6.7, Table 6 presents efficiency comparisons only with the DiffPure method and lacks comparisons with other baseline methods.

### Questions
1. Could the authors provide more details on how the inequalities presented in Lemma 3.4 and Theorem 3.5 are derived? 

2. Could the authors offer additional theoretical justification to explain why the proposed Jacobian loss and logit separation loss outperform other adversarial training methods?

3. Could the authors provide more details on how each loss component contributes to the overall filter training process and how the optimal values for the associated hyperparameters are determined?

4. The authors claim that the proposed method can function as a plug-and-play module. Could the authors provide comparative experiments demonstrating how this defense method integrates with and performs across different model architectures?

5. Could the authors provide more details on how different perturbation sizes affect the performance of the proposed method on CNN and ViT models?

6. Could the authors provide additional experimental results on runtime efficiency, including comparisons with more baseline methods?

### Soundness
2

### Presentation
1

### Contribution
1
