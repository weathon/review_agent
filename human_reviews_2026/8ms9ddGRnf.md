# Unified Theory of Adaptive Variance Reduction

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Variance reduction is a family of powerful mechanisms for stochastic optimization that appears to be helpful in many machine learning tasks. It is based on estimating the exact gradient with some recursive sequences. Previously, many papers demonstrated that methods with unbiased variance-reduction estimators can be described in a single framework. We generalize this approach and show that the unbiasedness assumption is excessive; hence, we include biased estimators in this analysis. But the main contribution of our work is the proposition of new variance reduction methods with adaptive step sizes that are adjusted throughout the algorithm iterations and, moreover, do not need hyperparameter tuning. Our analysis covers finite-sum problems, distributed optimization, and coordinate methods. Numerical experiments in various tasks validate the effectiveness of our methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates variance reduction based methods and provides a general theoretical analysis for such algorithms. The authors introduce a unified assumption that does not require the unbiasedness of the stochastic gradient estimator and develop their proofs based on this assumption. They further propose an adaptive variant that eliminates the need for hyperparameter tuning.  Moreover, they extend their analysis to finite-sum problems and distributed optimization. They also conduct numerical experiments to verify the effectiveness of their methods.

### Strengths
- This work provides a general theoretical analysis for variance reduction based methods, which may appear interesting.
- The authors give several examples showing that the proposed unified assumption can be satisfied by some existing algorithms.
- The analysis is further extended to other problem settings, although the extensions follow straightforwardly from proof of Section 4.
- The paper also includes numerical experiments to support the theoretical claims.

### Weaknesses
- My main concern is about Assumption 1, because after reading the proofs, it seems that the analysis mainly follows from this assumption. By simplifying this assumption, we can get

  $\mathbb{E}[\\| g^{t} - \nabla f(x^{t}) \\|^{2} ]
  < (1 - \rho_{1}) \\| g^{t-1} - \nabla f(x^{t-1}) \\|^{2}+ A \sigma_{t-1}^{2} + B L^{2} \gamma_t^2\\| g^t \\|^{2},$

  which is a more general form of the variance reduction guarantee. For example, the theoretical guarantee of the STORM algorithm is

  $\mathbb{E}[\\| g^{t} - \nabla f(x^{t}) \\|^{2} ]
  < (1 - \beta) \\| g^{t-1} - \nabla f(x^{t-1}) \\|^{2}+ 2\beta^2 \sigma_{t-1}^{2} + 2 L^{2} \gamma_t^2\\| g^t \\|^{2}.$ 

  Therefore, I am concerned about the contribution and novelty of this work. 

- I believe that the overall writing quality of the paper can be improved. For example, I do not agree that “Our Contribution” should appear as a standalone section and the equation in line 780 exceeds the width.

- Although the authors claim that Theorem 3 is truly parameter-free, it seems that the momentum parameter in the variance reduction update may still require manual tuning. How exactly is this parameter determined in practice?

### Questions
- In line 189, the authors write that constants B and C are used to bound the difference with the step size.
   However, I am confused by this statement, because as I mentioned in the Weaknesses, this term also includes $\\|g^t\\|^2$.
- I am also curious whether Assumption 1 still holds under the setting of generalized smoothness [1]. Could the authors clarify whether this assumption can be extended to that case?
- In the distributed optimization, the proposed algorithm is based on EF21. However, EF21 uses contractive compressors only in one direction (from workers to the server).  I wonder whether it is possible to employ contractive compressors bidirectionally as [2].



[1] Convex and Non-convex Optimization under Generalized Smoothness. NeurIPS 2023.

[2] Lower Bounds and Nearly Optimal Algorithms in Distributed Learning with Communication Compression. NeurIPS 2022.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a unified theoretical framework for adaptive variance reduction techniques in stochastic optimization, eliminating the need for hyperparameter tuning related to problem constants like smoothness parameter L. The framework applies to finite-sum problems (e.g., L-SVRG, SAGA, PAGE, ZeroSARAH), distributed optimization, and coordinate methods, achieving optimal rates for non-convex settings. Numerical experiments on logistic regression using the a9a dataset demonstrate the effectiveness of the proposed methods.

### Strengths
1. The proposed Assumption 1 captures the recursive behavior of variance-reduced estimators across diverse settings, allowing inclusion of a wider range of methods.
2. The adaptive step sizes, based on accumulated gradient norms, avoid dependence on unknown constants, making them practical for real-world applications. Asymptotically optimal rates are preserved, and theorems provide clear bounds.
3. The work unifies finite-sum, distributed (with compression to reduce communication), and coordinate methods, with new adaptive variants for each.

### Weaknesses
1. In my view, the paper’s main weakness is that its conclusions are not new. Prior adaptive methods [1,2] have already achieved the optimal convergence rate for finite-sum problems, and this work does not provide new results. More importantly, the proposed proof technique is not very different from earlier papers [1,2]. Although previous work focused on a single variance-reduction method, this paper’s contribution and level of innovation seem more like extending the previously introduced adaptive variance-reduction approach to other similar variance-reduction techniques. If there are aspects of the analysis that truly differ from prior methods, the authors should highlight them prominently.

2. To attain the optimal convergence rate $n^{1/4} T^{-1/2}$, the method still requires knowledge of $n$, which remains a problem-dependent parameter. If the optimal convergence rate could be achieved without knowing $n$, the contribution would be substantially stronger.

3. Running experiments only on small datasets for logistic regression is clearly insufficient. More comprehensive and realistic experiments would undoubtedly strengthen the paper.

4. I might have misunderstood here, but the paper introduces the PL condition and presents a theorem for the non-adaptive case. However, it seems the authors do not provide an adaptive method under the PL condition. If so, what is the purpose of presenting the PL condition in the paper?

---

[1] (NeurIPS 2022) Adaptive stochastic variance reduction for non-convex finite-sum minimization. 

[2] (NeurIPS 2024) Adaptive Variance Reduction for Stochastic Optimization under Weaker Assumptions

### Questions
See the Weakness part.

### Soundness
3

### Presentation
2

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
The paper proposes a unified analysis framework for stochastic variance-reduction methods that drops the traditional unbiasedness requirement on gradient estimators. Within a single assumption capturing recursive error contraction (with constants A, B, C, ρ1, ρ2), the work derives convergence guarantees for finite-sum, distributed, and coordinate methods, and introduces parameter-free, adaptive step-size schedules that do not require smoothness or PL constants. Instantiations include L-SVRG, SAGA, PAGE, ZeroSARAH, EF21, DASHA, SEGA, and JAGUAR. Experiments on logistic regression (a9a) show adaptive variants outperforming theoretical and tuned constant-step baselines and being robust to hyperparameter choices.

### Strengths
1. The analysis encompasses biased and unbiased estimators across finite-sum, distributed, and coordinate methods, filling gaps left by prior unified frameworks.
2. The proposed step-size rule depends only on observable quantities, removes reliance on L or µ, and achieves optimal O(1/√T) nonconvex rates and linear PL rates.
3. Ablations over batch sizes, probabilities, compression levels, clients, and coordinate sketch sizes indicate the adaptive variants consistently outperform tuned constant-step baselines.

### Weaknesses
1. Evaluation centers on a single dataset (a9a) and a simple logistic-regression task, with no large-scale nonconvex/deep-learning benchmarks, non-iid federated settings, or real distributed systems.
2. Although the schedule is “parameter-free,” users still choose α; constants (A, B, C, ρ) governing bounds are estimator-specific; links between theoretical and deployable code are not fully operationalized.
3. Claims about extending unified analyses to biased estimators and being first to provide adaptive distributed VR need tighter differentiation from recent unified and adaptive frameworks; related work discussion could be broadened.
4. Results are mainly iteration-wise curves without wall-clock time, communication/compute trade-offs, multiple seeds with confidence intervals, or statistical tests; the advantage over “tuned” baselines may depend on tuning effort and settings.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
3

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
This paper proposes a stochastic optimization method by unifying variance reduction methods across diverse settings and introducing parameter-free adaptive steps that eliminate hyperparameter tuning.

### Strengths
1. This paper proposes a unified theoretical framework that encompasses both unbiased and biased gradient estimators for variance reduction stochastic methods. 

2. A contribution of this paper is the development of adaptive step size schedules that eliminate the need for hyperparameter tuning.

3. It provides comprehensive convergence guarantees for three critical settings including non-convex optimization, PL condition and adaptive step sizes.

### Weaknesses
1. While Assumption 1 is the paper's theoretical centerpiece, it provides limited guidance on how to calibrate constants for new VR methods.

2. The adaptive step size relies on a hyperparameter, but the paper provides limited insight into its practical choice or theoretical impact.

3. The related work section (Section 2) mentions adaptive methods like STORM (Cutkosky & Orabona, 2019) and Prodigy (Mishchenko & Defazio, 2023) but fails to provide a direct, quantitative comparison of the proposed adaptive VR methods to these baselines.  

4. The experimental results are not convincing. The paper claims applicability to non-convex problems (e.g., neural networks) but only tests convex logistic regression.

5. Several minor typos and notation ambiguities reduce readability and could confuse readers: In Lemma 14 (Section B.1), the statement mentions "Let "—the extra "N" is a typo and should be removed

### Questions
1. While Assumption 1 is the paper's theoretical centerpiece, it provides limited guidance on how to calibrate constants for new VR methods.

2. The adaptive step size relies on a hyperparameter, but the paper provides limited insight into its practical choice or theoretical impact.

3. The related work section (Section 2) mentions adaptive methods like STORM (Cutkosky & Orabona, 2019) and Prodigy (Mishchenko & Defazio, 2023) but fails to provide a direct, quantitative comparison of the proposed adaptive VR methods to these baselines.  

4. The experimental results are not convincing. The paper claims applicability to non-convex problems (e.g., neural networks) but only tests convex logistic regression.

5. Several minor typos and notation ambiguities reduce readability and could confuse readers: In Lemma 14 (Section B.1), the statement mentions "Let "—the extra "N" is a typo and should be removed

### Soundness
2

### Presentation
2

### Contribution
2
