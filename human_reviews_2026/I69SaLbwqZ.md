# Distribution-informed Online Conformal Prediction

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 4

## Abstract
Conformal prediction provides a pivotal and flexible technique for uncertainty quantification by constructing prediction sets with a predefined coverage rate. Many online conformal prediction methods have been developed to address data distribution shifts in fully adversarial environments, resulting in overly conservative prediction sets. We propose Conformal Optimistic Prediction (COP), an online conformal prediction algorithm incorporating underlying data pattern into the update rule. Through estimated cumulative distribution function of non-conformity scores, COP produces tighter prediction sets when predictable pattern exists, while retaining valid coverage guarantees even when estimates are inaccurate. We establish a joint bound on coverage and regret, which further confirms the validity of our approach. We also prove that COP achieves distribution-free, finite-sample coverage under arbitrary learning rates and can converge when scores are i.i.d. The experimental results also show that COP can achieve valid coverage and construct shorter prediction intervals than other baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Conformal Optimistic Prediction (COP), an online conformal prediction algorithm designed to address the over-conservatism of existing methods in non-stationary environments like time series. 

The core innovation of COP is a two-step update mechanism that refines the standard gradient-based adjustment for the prediction set size. After a primary update based on the current miscoverage error, COP applies a refinement step using an estimated cumulative distribution function (CDF) of the non-conformity scores. This estimated CDF acts as an "optimistic hint" about the data's underlying patterns, enabling the algorithm to produce tighter, more efficient prediction sets when such patterns exist. 

Crucially, the method is robust; it maintains provable, distribution-free, finite-sample coverage guarantees even if the CDF estimate is inaccurate. The authors frame COP within the Optimistic Online Gradient Descent (OOGD) paradigm, deriving a novel joint bound on regret and coverage that clarifies its theoretical motivation. 

Experiments on synthetic datasets with distribution shifts and real-world time series from finance, energy, and climate show that COP consistently achieves the target coverage while generating significantly narrower prediction intervals than state-of-the-art baselines, demonstrating its superior efficiency and practical utility.

### Strengths
*   **Originality**
    *   Presents a new synthesis of online Conformal Prediction (CP) with Optimistic Online Gradient Descent (OOGD).
    *   Uniquely frames distributional information as a proactive "optimistic hint" rather than simply replacing the robust reactive update rule.

*   **Quality**
    *   **Comprehensive Theory:** Provides a full suite of rigorous guarantees, including a novel joint regret-coverage bound, essential distribution-free finite-sample coverage, and asymptotic consistency, establishing strong reliability.
    *   **Thorough Experiments:** Validated on a wide range of synthetic and diverse real-world datasets, using multiple base models against a strong set of seven state-of-the-art baselines, convincingly demonstrating its superior performance and general applicability.

*   **Clarity**
    *   The paper is exceptionally well-written, with a clear logical flow that makes complex ideas accessible.
    *   **Excellent Background Explanation:** The background section effectively explains all necessary concepts, providing the context needed to fully appreciate the paper's contribution.

*   **Significance**
    *   Addresses the critical and practical problem of over-conservatism (excessively wide intervals) in online CP, a major barrier to real-world adoption.
    *   The proposed method (COP) is effective, theoretically sound, and simple to implement, giving it high potential to become a standard practitioner's tool.
    *   The theoretical framework introduces new analytical tools that can inspire future research.

### Weaknesses
1.  **Lack of Statistical Significance in Experimental Claims:** While the experiments are extensive, the tables present only point estimates (mean and median) for the evaluation metrics from a single run. This makes it difficult to ascertain whether the observed improvements of the proposed method over baselines are statistically significant or merely due to experimental randomness. For example, a small difference in average interval width between two methods might not be meaningful without a measure of variance. The empirical claims would be substantially strengthened by including standard errors or confidence intervals for the reported metrics, which could be derived from multiple runs with different random seeds.

2.  **Over-reliance on Tables for Presenting Results:** The paper relies heavily on large, dense tables (Table 1 and 2) to display the primary experimental outcomes. Such a presentation format makes it challenging for readers to quickly parse results and identify performance trends across different datasets and models. The paper's readability and impact would be greatly enhanced by using visualizations. For instance, it is suggested that the authors consider replacing or supplementing the tables in the main text with figures like **boxplots**. Boxplots would more effectively illustrate the distribution, median, and spread of the prediction interval widths, while also providing a natural way to incorporate the standard error or uncertainty estimates mentioned in the first point, thereby increasing the perceived reliability and clarity of the experimental validation. The detailed tables could then be moved to the Appendix for reference.

### Questions
**On the Limits of the CDF Estimator's Robustness:**

A major strength of the paper is that COP maintains coverage guarantees even with an inaccurate CDF estimate. However, a poor estimate could still harm efficiency, potentially making the intervals wider than those from a simpler baseline like OGD.

Could you discuss the practical boundaries of this robustness? Is there a point where a consistently misleading CDF estimate (e.g., during a prolonged, adversarial shift) makes COP perform worse than the baseline OGD in terms of interval width?
An experiment on a synthetic dataset with a deliberately miscalibrated or noisy CDF estimator could help characterize this failure mode and provide practical guidance on when the optimistic step is most beneficial.

**On Statistical Significance of Experimental Results:**

The experimental results in the tables are promising, but they report point estimates from what appears to be a single experimental run. To strengthen the claims of superiority, especially when the performance differences with top baselines are small, could you provide measures of uncertainty for the reported metrics (e.g., standard errors or confidence intervals calculated over multiple runs with different random seeds)? This would be crucial for confirming that the observed improvements are statistically significant.

**On the Impact of Noise and Low Predictability in Scores:**

COP's advantage stems from exploiting predictable patterns in the non-conformity scores. This raises questions about its performance in settings where such patterns are weak or heavily obscured.

How does COP perform in high-noise or low signal-to-noise ratio settings, where the underlying predictable structure in the scores is minimal? In such a scenario, does the optimistic update, which is based on a noisy and potentially uninformative CDF estimate, risk making spurious adjustments that could degrade efficiency (i.e., widen intervals) compared to a more conservative baseline like decay-OGD?
It would be very illuminating to include an experiment on a synthetic dataset where the level of i.i.d. noise added to the non-conformity scores is systematically varied. This would help to characterize the performance trade-offs and define the boundaries of where COP's optimism is a clear advantage.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces an online conformal prediction algorithm that incorporates distributional information of non-conformity scores into the update rule. The proposed method leverages an estimated cdf of scores to anticipate predictable patterns and tighten prediction sets while preserving coverage guarantees. By establishing a connection between the proposed method and OOGD, authorss demonstrate a joint bound on regret and coverage.

### Strengths
1- The idea of incorporating distribution-informed optimistic updates into online conformal prediction is a meaningful advancement in this area.

2- The paper provides solid theoretical foundations for both coverage and regret.

3-The paper is well written and easy to follow, with clear motivation and smooth storytelling that connects prior work to the proposed method.

### Weaknesses
Please check the questions.

### Questions
1- How were the hyperparameters chosen for COP?

2- Could the authors include a computational complexity analysis of the proposed method and a comparison with previous baselines, such as ACI?

3- Would authors show empirically whether performance degrades significantly with poor CDF estimation or not?

4- Could the authors clarify whether $q_t$ in line 121  is a type or not? It's not consistent with eq(2)

### Soundness
3

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
This paper proposes Conformal Optimistic Prediction (COP), an online conformal prediction (CP) algorithm that integrates distributional information of nonconformity scores into the update rule. COP leverages estimated cumulative distribution functions to refine the prediction radius dynamically. A joint coverage–regret bound and finite-sample distribution-free coverage have been established.

### Strengths
The paper introduces an elegant refinement step using estimated CDFs. Theoretical contributions include finite-sample and asymptotic coverage guarantees, as well as a joint regret–coverage bound under general learning rates.

### Weaknesses
- Although both empirical and kernel-based CDFs are considered, the impact of different estimators or misspecification on performance and validity is not deeply analyzed.


- The presentation could be improved for greater readability. For instance, abbreviations should be used consistently once their full forms have been introduced. Moreover, Sections 3.1–3.3 are mathematically dense, which may make it challenging for readers unfamiliar with OOGD or online conformal prediction to grasp the underlying intuition behind each step.

- Computation time of the methods in experiments.

### Questions
- How to choose the scale factor in real practice? How sensitive is COP’s performance to the choice of the scale factor?

- In adversarial settings where estimated CDFs are inaccurate, how robust is COP compared with purely adversarial CP methods like SF-OGD?

- How does COP behave under extreme concept drift or abrupt regime changes, where the empirical CDF is no longer representative of recent data?

### Soundness
3

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
3

### Summary
This paper introduces COP, an online conformal method that injects predictable structure via an estimated CDF of nonconformity scores to perform an “optimistic” radius refinement—tightening sets when patterns exist while remaining robust under misspecification. The theory provides a joint coverage–regret bound together with distribution-free, finite-sample coverage under arbitrary learning rates and convergence when scores are i.i.d. Empirically, COP attains target coverage and yields consistently shorter intervals than strong baselines across synthetic shift scenarios and real-world datasets.

### Strengths
The paper is original in recasting online conformal prediction as optimistic online gradient descent, blending a standard feedback step with a CDF guided optimistic refinement to reduce conservativeness. The theory is solid, delivering a joint coverage regret bound, distribution free finite sample coverage for arbitrary learning rates, and convergence under i.i.d. scores, while the presentation is clear with precise setup and actionable pseudocode. Experiments across simulated shifts and multiple real world domains show target coverage with consistently tighter intervals, underscoring practical significance and easy deployability, using simple ECDF or KDE plugins.

### Weaknesses
The central theory relies on an unverifiable same sign assumption and boundedness that may fail under nonstationary or heavy tailed regimes; an assumption free, non asymptotic refinement bound based on observable CDF error would strengthen the claims. The simulations omit heteroscedasticity, variance changepoints, and heavy tails, lack ablations isolating the optimistic step and ECDF versus KDE, and do not report recovery time or conditional coverage. Practical clarity and scalability are under specified: window and bandwidth choices are sensitive, per step cost appears $(O(w))$ time and $(O(w))$ space without specialized data structures, and the fairness of hyperparameter tuning is unclear; adding adaptive selection rules and explicit complexity with wall clock scaling would improve deployability. Minor exposition gaps remain in mapping the trust region to the step size and in indexing consistency.

### Questions
1. Table 2 shows COP attains competitive coverage with narrow intervals, but Proposition 1 hinges on the unverifiable ``same--sign'' assumption.

  (1) Please report how often $\mathrm{sign}(\hat F_{t+1}(\hat q_{t+1}) - (1 - \alpha)) = \mathrm{sign}(F_{t+1}(\hat q_{t+1}) - (1 - \alpha))$ holds on the real-world datasets (e.g., by sliding windows), and correlate this rate with coverage/width. This would clarify whether COP's gains are explained by the assumption or by broader robustness.

  (2) Provide a refinement-step analysis based solely on the observable CDF error 
$\varepsilon_t := \sup_q \lvert \hat F_t(q) - F_t(q) \rvert$ that yields
$$
\text{coverage} \;\le\; \Phi(\varepsilon_{1:T}, \{\eta_t\}, \{\lambda_t\})
$$
regardless of the sign, that is, a non-asymptotic worst-case bound that does not rely on the same-sign assumption. For i.i.d. windows, a DKW-type inequality (and for time series, its mixing/martingale analogues) controls $\varepsilon_t$, aligning the strong empirical results with equally robust theory.



2. The simulation studies assume homoscedastic Gaussian noise. To better assess the robustness of COP's CDF-based refinement, please include experiments with heteroscedasticity, variance changepoints, and heavy-tailed noise (e.g., Student-$t$). Please report coverage and width as well as post-shift recovery time (steps to return to target coverage) across these settings, and summarize observed failure modes.


3. For deployability, could you briefly report the time and space complexity of COP’s per-step update (with windowed ECDF/KDE) and a small scaling plot of per-step wall-clock versus window size $w$? If the implementation is $O(w)$ per step (vs. $O(1)$ for OGD/PID), please state this explicitly; if a data structure is used, please indicate the amortized complexity (e.g., $O(\log w)$) and the memory footprint $O(w)$.

### Soundness
2

### Presentation
3

### Contribution
2
