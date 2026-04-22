# Stochastic Gaussian Zeroth-Order Optimization: Improved Convergence Analysis under Skewed Hessian Spectra

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
This paper addresses large-scale finite-sum optimization problems, which are particularly prevalent in the big data era. 
In the field of zeroth-order optimization, stochastic methods have become essential tools. 
Natural zeroth-order stochastic methods primarily rely on stochastic gradient descent (SGD).
Preprocessing the stochastic gradient using a Gaussian vector defines the method ZO-SGD-Gauss (ZSG), whereas estimating coordinate-wise partial derivatives defines ZO-SGD-Coordinate (ZSC).
Compared to ZSC, ZSG often demonstrates superior performance in practice.
However, the underlying mechanisms behind this phenomenon remain unclear in the academic community.
To the best of our knowledge, our work is the first to theoretically analyze the potential advantages of ZSG compared to ZSC.
To facilitate convergence analysis, the quadratic regularity assumption is introduced to generalize the smoothness and strong convexity to the Hessian matrix.
This assumption makes it possible to integrate Hessian information into the complexity analysis.
We provide a theoretical analysis proving the significant convergence improvement of ZSG. Finally, experiments on both synthetic and real-world datasets validate the effectiveness of our theoretical analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the zero-order SGD with gaussian gradient estimation. The authors refine the convergence rates, showing that convergence depends on $\text{tr}(\nabla^2 f)$, instead of $\lambda_{\max}(\nabla^2 f)$. They compare the performance of the analyzed algorithm on quadratic functions, as well as on logistic regression on LibSVM datasets.

### Strengths
1)The authors improve the existing rates for zero-order optimization, that benefits the skewed Hessian spectra, when $\text{tr}(\nabla^2f) \leq d\lambda_{max}(\nabla^2 f)$.

### Weaknesses
1)The authors claim that the terms $P_1(\alpha)$ and $Q_1(\alpha)$ are negligible with the small choice of $\alpha$. However, these terms contain multipliers $\lambda_{\max}^2(\nabla^2 f)d^3$ and $\lambda_{\max}^2(\nabla^2 f)d^3T$, which are frequently large.

2)Unclear writing -- no description of ZSC was given, though, the authors compare the obtained results with it throughout the paper, differences between Theorems 4.5, 4.7, 4.8 are hard to distinguish.

3)Considering $\text{tr}(\nabla^2 f)$ as $\max_{z^t}\nabla^2 f(z^t)$ wuth similar definitions for $\lambda_{\min}(\nabla^2 f)$ and $\lambda_{\max}(\nabla^2 f)$ is a rough estimate. With this analysis, most convex problems might be considered as strongly convex, when $\lambda_{\min} > 0$. 

4)The plots do not contain confidence intervals; however, stochastic methods are considered. Also, more complex setups than LibSVM datasets are missing.

### Questions
1)The derived stepsize depends on $\text{tr}(\nabla^2 f)$ and $\lambda_{\min}(\nabla^2 f)$. How are they obtained in practice?

2)If we access the objective's Hessian during the training process, why do we consider the derivative-free optimization at the first place?

3)Do we demand gaussian distributions in the scheme proposed in Algorithm 1? What if we consider arbitrary (a)symmetric distribution? 

4)Does corollary 4.6 result in sublinear convergence for strongly convex functions? According to Assumption 2.1  $\gamma_l > 0$.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper analyzes **Gaussian-based zeroth-order stochastic gradient descent (ZSG)** and compares it with **coordinate-based zeroth-order SGD (ZSC)**.  
It introduces **Assumption 2.1 : Quadratic Regularity (QR)**, a deterministic Hessian-metric condition generalizing smoothness and strong convexity.  
Under this assumption the authors derive iteration-complexity bounds suggesting that ZSG enjoys milder dimensional dependence (\(\operatorname{tr}(M)\)) than ZSC (\(d\,\lambda_{\max}(M)\)), particularly for skewed Hessian spectra.  
Experiments on synthetic quadratics and logistic regression qualitatively support the idea that Gaussian perturbations help under ill-conditioning.

However, several technical and presentation issues seriously weaken the results.  
Most importantly, **Theorems 4.5 and 4.8 mis-state convergence rates**: the quantities \(Q_1\) and \(Q_2\) depend on the total iteration count T, so the bounds do **not** imply convergence to zero for stochastic functions.  
Combined with ambiguous assumptions and missing discussion of validity domains, the paper’s theoretical claims are overstated.

### Strengths
- Provides a careful deterministic analysis under a clear quadratic-regularity assumption.  
- Offers intuition on why Gaussian perturbations can mitigate poor conditioning.  
- Experiments qualitatively match the deterministic predictions.

### Weaknesses
1. **Mis-stated main theorems (4.5 & 4.8).**  
   The “constants” \(Q_1\) and \(Q_2\) depend explicitly on T through cumulative step-size and variance terms.  
   This destroys asymptotic convergence: the error bound does not vanish as T → ∞.  
   Despite this, the text claims a “sublinear” convergence in the *stochastic* case.  
   The presentation conceals the dependence, giving the impression of a stronger result than actually proved.

2. **Failure to handle stochastic functions.**  
   Because the variance term grows with T, the analysis effectively applies only to deterministic or finite-sum settings.  
   There is no uniform bound on stochastic noise, so the claimed results do **not** establish convergence for genuinely stochastic oracles.  
   The theory should have been presented as deterministic analysis rather than stochastic convergence.

3. **Ambiguity in Assumption 2.1.**  
   The constants \(\gamma_u,\gamma_l\) are written as if they may depend on x,y,z, which would make the inequalities tautological.  
   For the theorems to hold, they must be global constants independent of those points.  
   This appears to be a typographical error that needs correction.

4. **Lack of discussion of when assumptions hold.**  
   The paper should explicitly identify and justify classes of functions satisfying QR (e.g., quadratics, certain regularized GLMs).  
   Beyond the trivial quadratic case, examples are only hinted at and never proved.

5. **Limited novelty and scope.**  
   Algorithmically, ZSG is standard (Gaussian SPSA / NES).  
   Experiments are small-scale and deterministic; no tests on high-variance or nonconvex settings.

6. **Lack of transparency.**  
   By labeling \(Q_1,Q_2\) as constants and not clarifying their T-dependence, the manuscript obscures a fundamental limitation of the analysis.

### Questions
1. Can you formally characterize non-quadratic functions (e.g., logistic or least-squares objectives) that satisfy the QR condition with global constants?  
2. Would a fully deterministic framing (σ = 0) strengthen the paper?  
3. How could the variance term be controlled to extend the results to true stochastic settings?  Maybe use momentum?
4. Can you restate Theorems 4.5 and 4.8 with explicit T-dependence and honest asymptotic interpretation?

### Soundness
1

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper establishes an accelerated convergence rate for ZSG and theoretically analyze the potential advantages of ZSG compared to ZSC. The paper evaluates on both synthetic and real-world datasets and demonstrate the performance of ZSG outperforms that of ZSC.

### Strengths
Novel analysis:
Novel theoretical contribution by being the first to rigorously analyze why ZSG outperforms ZSC in practice

The theoretical gap addressed is important:
The work addresses why ZSG is preferred in practice despite identical O(d) complexity bounds

Clarity:
The paper is generally well written with clear problem setup and algorithmic description. The contributions and stated clearly and the main results are stated precisely with appropriate assumptions.

### Weaknesses
Notations:
There is heavy notation that accumulates through the paper that can make it difficult to parse.


Assumptions do not match motivating examples:
All results assume strongly convex objectives (Assumption 2.1), but the examples used to motivate the analysis such as LLM fine tuning involve non-convex deep learning problems. This means the analysis cannot be directly applied to the examples it states


Experiments:
Some experimental details are sparse such as how are step sizes chosen. The experimental evaluations are very toy settings. However this may be fine because the work mainly fills a theoretical gap. 

Missing comparisons:
The paper does not provide theoretical or empirical comparison to variance reduced zeroth-order methods or adaptive/momentum based zeroth order methods.

### Questions
How should users set the step size when $tr(M)$, $\lambda_\min(M)$ are unknown? 

Is there a way to construct an experiment for LLM fine tuning and other motivating examples?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies stochastic zeroth-order optimization and compares Gaussian perturbation ZO-SGD (ZSG) with coordinate finite-difference ZO-SGD (ZSC). Under a quadratic regularity assumption that lifts smooth/strong-convex conditions to a Hessian-norm form, it proves that ZSG enjoys weaker dimension dependence and faster convergence than ZSC, especially under skewed Hessian spectra.

### Strengths
This paper is easy to navigate: assumptions and notation are stated upfront, the notion of quadratic regularity is introduced with intuition before formal use.  And the Theorems show ZSG attains iteration/query bounds that avoid the explicit factor 𝑑 that appears for ZSC.

### Weaknesses
**Question 1.** The bounds hinge on $\gamma_u, \gamma_l$ and on quantities like $\operatorname{tr}(M), \lambda_{\min }(M), \lambda_{\max }(M)$, which may be unknown or hard to estimate. Thus, practical guidance for choosing $\eta_t$ that depends on these is limited.

**Question 2.** The empirical comparison is primarily ZSG vs. ZSC; other ZO baselines (e.g., two-point random directions with mini-batching/importance sampling) are not reported, making it harder to gauge practical significance.

**Question 3.**  Although the theory targets skewed spectra, experiments do not directly measure Hessian anisotropy on real data (only synthetic constructions), so the claimed mechanism is not empirically verified on those tasks.

**Question 4.** Sensitivity to $\alpha$ and noise assumptions. Theory requires "sufficiently small" $\alpha$ and a bounded variance $\sigma^2$; experiments fix $\alpha=10^{-6}$ without sensitivity analysis, so robustness is unclear.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
