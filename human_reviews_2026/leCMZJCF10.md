# Robust Classification with Noisy Labels Based on Posterior Learning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Deep learning has shown robustness to label noise under specific assumptions, yet its performance under extremely high noise rates remains a significant challenge. 
In this paper, we theoretically demonstrate under what conditions models estimating the posterior probability can achieve high classification accuracy in the presence of extremely strong instance-dependent label noise without performing loss correction approaches. 
To estimate the noisy posterior, we propose a class of objective functions derived from the variational representation of the $f$-divergence. Furthermore, we propose two correction methods to achieve robustness when the algorithm is not intrinsically robust to label noise: one method is implemented during the training process, and the other is performed during inference. 
Finally, we show the validity of our theoretical results and the effectiveness of the proposed methods on synthetic and real-world label noise settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the problem of robust classification under extremely high label noise.
The authors argue that when the noise transition matrix becomes **anti-diagonally dominant** (i.e., when the noise rate $\eta > (K-1)/K$), it is still possible to recover the clean Bayes-optimal classifier by minimizing the noisy posterior. They propose a method called **f-divergence-based Noisy Posterior Learning (f-NPL)** to estimate the noisy posterior $p_{Y_\eta|X}$. They also propose two posterior estimator correction to correct the posterior.

### Strengths
The paper provides an interesting observation: even when the transition matrix is anti-diagonally dominant (i.e., extreme noise $\eta > (K-1)/K$), the classifier that minimizes the noisy posterior coincides with the clean Bayes classifier: $\arg\max_y p_{Y|X}(y|x) = \arg\min_y p_{Y_\eta|X}(y|x)$.

### Weaknesses
1. The proposed method **relies on the knowledge of the noise transition matrix** $T$. When the noise rate is relatively small ($\eta < (K-1)/K$), the transition matrix can be estimated using existing approaches such as Forward or DualT. However, when the noise becomes extremely large ($\eta > (K-1)/K$), these estimators are known to fail, as they assume a diagonally dominant transition structure. Moreover, in the experimental section, the paper evaluates (T) estimated by Forward and DualT **only under small noise rates**. In Figures 2 and 3, the authors report results of the proposed method under extremely high noise rates, but it remains unclear **whether the transition matrix used in these experiments is the ground-truth matrix or an estimated one**, which substantially affects the interpretation of the reported robustness.
2. The multi-class result assumes **asymmetric uniform off-diagonal label noise**. The theorem does not extend to more general class-dependent (e.g., pair flipping label noise) or instance-dependent noise.
3. The practical utility of the proposed method is unclear. Given the reliance on the knowledge of the transition matrix, it is doubtful that f-NPL offers benefits for real-world noisy datasets, where the noise is neither uniform nor the transition matrix is unknown.

### Questions
In Figures 2 and 3, is the transition matrix used in these experiments the estimated one, or the ground-truth matrix? This distinction is crucial, as the proposed correction methods explicitly rely on the knowledge of the transition matrix $T$, and the paper's key claim is the robustness under extremely large noise ($\eta > (K-1)/K$).

Suggestions:

The font sizes on Figures 5 and 6 are small.

### Soundness
2

### Presentation
2

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
The authors propose a new loss function that estimates the posterior distribution using the variational representation of an f-divergence, instead of relying on standard cross-entropy loss.
This framework can be implemented with various types of f-divergences, offering flexibility in modeling.
Furthermore, the authors provide theoretical justification showing that their approach can be applied even under extremely noisy label conditions, demonstrating its robustness in challenging LNL scenarios.

### Strengths
1. The proposed loss function is novel and conceptually elegant. It departs from the conventional likelihood-based formulations and leverages the variational structure of f-divergence to obtain a noise-robust posterior.

2. The experimental performance is outstanding. Notably, the method achieves state-of-the-art results ([1]) on the WebVision dataset without relying on heuristic sample selection or confidence filtering, which are commonly used in recent LNL methods.
This demonstrates a clear advantage over existing robust-loss approaches, both in performance and simplicity.

[1] Manifold dividemix: A semi-supervised contrastive learning framework for severe label noise

### Weaknesses
[Minor]

While the claimed robustness to extreme noise environments is theoretically interesting, it may not be the central practical advantage of this work.
Clarifying the method’s most distinctive strengths such as its generality across divergence families or some factors which improve the overall performance would help strengthen the Introduction and make the paper’s contribution more explicit.

### Questions
(Empirically) are there cases where certain f-divergences yield more stable posterior estimation under high noise rates? or the authors can provide some results depends on different choices of the f-divergences?

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
This paper investigates classification robustness under extreme label noise, particularly when the standard assumption of diagonally dominant noise transition matrices (i.e., relatively low noise rates) no longer holds. The authors demonstrate that when the transition matrix becomes anti-diagonally dominant, meaning the true class label is less probable than incorrect ones, it is theoretically possible to recover the correct label by minimizing, rather than maximizing, the noisy posterior.

This idea is formalized through Theorem 3.1 and Corollary 3.2, which establish that the argmin of the noisy posterior yields Bayes-optimal predictions under symmetric and certain asymmetric noise regimes. Building on this foundation, the authors propose an f-divergence based Noisy Posterior Learning (f-NPL) framework that generalizes cross-entropy (CE) and Active-Passive Loss (APL) formulations via a variational f-divergence representation. Two complementary correction strategies are introduced:
- Objective function correction, applied during training
- Posterior estimator correction, applied at inference time.

Overall, while the paper provides a clean and theoretically well-founded extension of posterior-based robustness analysis to anti-diagonally dominant noise settings, its central insight reversing the prediction rule from argmax to argmin when noise dominance flips is conceptually straightforward once the structure of the transition matrix is known. The proposed f-divergence formulation, though elegant, represents a modest reparameterization of existing APL and f-GAN frameworks. Empirical results are competently executed but largely confirm expected trends. The work is mathematically neat yet conceptually incremental and not sufficiently novel or impactful for ICLR.

### Strengths
- The key idea that when the noise matrix is anti-diagonally dominant, the problem can be reversed (i.e., by minimizing the noisy posterior instead of maximizing it) is novel and extends robustness theory beyond standard noise regimes. The structure is also symmetrically generalizable: if diagonal dominance guarantees correctness of argmax, anti-diagonal dominance guarantees correctness of argmin. This symmetry makes the framework conceptually complete.
- The two bias-correction strategies (objective vs posterior) are mathematically well-grounded and neatly complementary, one adjusts the optimization process, the other the inference step.
- Empirical validation: Experiments convincingly demonstrate that the argmin-based approach indeed recovers correct classifications under extreme noise ($\eta$ > 0.9), matching theoretical claims. Performance against strong baselines (e.g., ANL, RENT, Forward) is solid.

### Weaknesses
-  The paper has limited novelty beyond theoretical inversion. Once the anti-diagonal structure is recognized, reversing the classification rule (argmin instead of argmax) is mathematically straightforward. The theory formalizes this observation elegantly but may not constitute a substantial new insight.

- The paper also has restrictive assumptions: The robustness guarantees rely on knowing (or estimating) the noise transition matrix and on strong anti-diagonal or symmetric structure. In most practical label noise scenarios, such structure is rare or only partially satisfied.

- Experiments primarily confirm synthetic settings crafted to satisfy the theoretical assumptions. The real-world experiments (WebVision) show improvements but not dramatic differences, suggesting limited applicability of the extreme-noise regime theory to natural data.

- The exposition is dense and sometimes overly formal, making it harder to extract intuition. Clearer separation between theoretical novelty and empirical heuristics would strengthen impact.

### Questions
- What happens in the intermediate regime where the transition matrix is neither diagonally nor anti-diagonally dominant? Is there a smooth transition between the argmax and argmin regimes, or does the model’s behavior degrade sharply?
- Many results assume that $\eta$ and the transition matrix are either known or accurately estimated. How sensitive is the f-NPL method (and the argmin decision rule) to errors in these estimates?

### Soundness
3

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
4

### Summary
Summary:

This paper tackles the challenge of robust classification under extremely high label noise. The authors propose f-divergence-based Noisy Posterior Learning (f-NPL), a general framework that estimates the noisy posterior using the variational form of f-divergence. To enhance robustness, two correction methods are introduced. Experiments on CIFAR-10/100, WebVision, and ImageNet demonstrate that f-NPL achieves superior robustness across various noise settings, including extreme noise rates, outperforming existing APL-based and correction-based methods.

### Strengths
1. Strong theoretical contribution – The paper provides new insights into label noise robustness, proving that accurate classification is possible even when the noise rate exceeds the traditional bound η > (K - 1)/K. This extends the theoretical foundation of learning with noisy labels.

2. Unified and flexible framework – The proposed f-NPL framework elegantly unifies posterior estimation and f-divergence optimization, offering a principled way to design robust objective functions that encompass and generalize existing APL-based losses.

### Weaknesses
1. The paper fails to compare with 2025 state-of-the-art methods (e.g., NoiseCLIP from CVPR 2025, UniNL from ICLR 2025, or CoNo from NeurIPS 2024), which have demonstrated superior performance under extreme noise (≥93% on CIFAR-10 with 90% symmetric noise) without requiring the noise transition matrix. By omitting these comparisons, the authors overstate the practical significance of f-NPL, as its reported gains may no longer be competitive, and its dependence on accurate transition matrix estimation remains a critical unaddressed vulnerability in real-world deployment.

2. f-NPL is not original; the variational f-divergence formulation has already been applied to clean supervised classification in Novello & Tonello (2024), and the resulting objective is merely a special case of Active-Passive Losses introduced in Ma et al. (2020). Thus, f-NPL offers only incremental packaging rather than a fundamentally new loss design paradigm.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
