# Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Uncertainty quantification is crucial in safety-critical systems, where decisions must be made under uncertainty. 
In particular, we consider the problem of online uncertainty quantification, 
where data points arrive sequentially. 
Online conformal prediction is a principled online uncertainty quantification method that dynamically constructs a prediction set at each time step.
While existing methods for online conformal prediction provide long-run coverage guarantees without any distributional assumptions, they typically assume a *full feedback* setting in which the true label is always observed. 
In this paper, we propose a novel learning method for online conformal prediction with *partial feedback* from an adaptive adversary—a more challenging setup where the true label is revealed only when it lies inside the constructed prediction set.
Specifically, we formulate online conformal prediction as an adversarial bandit problem by treating each candidate prediction set as an arm. 
Building on an existing algorithm for adversarial bandits, 
our method achieves a long-run coverage guarantee by explicitly establishing its connection to the regret of the learner.
Finally, we empirically demonstrate the effectiveness of our method in both independent and identically distributed (i.i.d.) and non-i.i.d. settings, 
showing that it successfully controls the miscoverage rate while maintaining a reasonable size of the prediction set.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles online conformal prediction with semi-bandit feedback and adaptive adversaries by reducing threshold selection to an adversarial bandit problem. It designs a loss that trades off deviation from a target miscoverage level with efficiency (tuned by λ) and proves a conversion from bandit regret to miscoverage guarantees. Implementations with EXP3.P and an “unlocking” variant that reuses feedback via monotonicity provide high-probability bounds, and experiments on classification and regression show accurate coverage tracking with compact prediction sets, including under covariate shift.

### Strengths
The paper introduces an adversarial semi bandit formulation for online conformal prediction and reduces threshold selection to an adversarial bandit with a transparent bridge from regret control to miscoverage control, together with an unlocking mechanism that exploits monotonicity to reuse feedback. It offers high probability guarantees and a clear analysis pipeline from loss design to regret bounds to miscoverage guarantees, implemented with practical EXP3.P algorithms. Experiments on classification and regression show close tracking of target miscoverage with strong sample efficiency, consistent with the theory. The writing is clear, and the modular design lets other bandit learners plug in, increasing impact in retrieval, selective prediction, and human in the loop systems.

### Weaknesses
See details in Questions.

### Questions
1. In both experiments, the choice of the hyperparameter $\lambda$ is not specified. How is $\lambda$ selected, and how sensitive are the results to different values of $\lambda$? The paper lacks discussion on this critical design choice.
2. What is the performance of the proposed method under low feedback rates (i.e., when semi-bandit feedback is rarely triggered)? The current experiments do not analyze or discuss scenarios with sparse feedback.
3. Could there exist a more direct threshold-updating mechanism (e.g., standard online quantile tracking or reinforcement learning methods) that also achieves adversarial miscoverage control? Why is the bandit framework preferred here, and what are its concrete advantages over such alternatives?
4. In the abstract (line 025), the word "micoverage" should be corrected to "miscoverage".
5. In line 127, the definition of “size” is unclear. Why is it claimed to be proportional to exp(-$\pi$)? Please provide a reasonable justification or reference.
6. In line 172 (Equation 3), the definition of $a_t(\pi_t)$ appears, but later discussions refer to $a_t(\pi)$. Are they the same quantity? A unified and explicit definition for $a_t(\pi)$ is missing.
7. In line 213 under EXP3.P-CP, the parameters $\delta$and $K$ are used without prior introduction or definition. Please clarify their meanings.
8. In line 225, what is the valid range of the probability distribution $p_t$? It would be helpful to explicitly state its domain or constraints.
9. In Figure 1, the blue curve (MVP) is barely visible or unidentifiable. Please consider adjusting the visualization for clarity.
10. In line 568, there is an unexplained symbol or placeholder "(?)", which appears to be a typo.
11. On the final page of the appendix (Page 18), there are punctuation issues in one of the equations. A thorough proofreading of all punctuation throughout the paper is recommended.

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
The paper explores how regret minimization techniques from bandit and sequential prediction literature can be adapted to online conformal prediction under partial feedback. It establishes a theoretical link by deriving an upper bound on the miscoverage rate in terms of the learner’s regret, enabling the use of these methods in this setting. Building on this, the paper demonstrates the application of the EXP3.P algorithm and introduces a novel variant, EXP3.P-CP with Unlocking, which exploits additional partial feedback and the monotonicity of prediction set construction. For both methods, the paper provides miscoverage guarantees and evaluates their empirical performance on regression and classification tasks.

### Strengths
- The partial feedback setting in online conformal prediction is highly relevant, offering practical value to both the conformal prediction research community and real-world practitioners.
- The paper’s connection to regret minimization methods is particularly compelling, as it unlocks a rich toolbox of established techniques that can be effectively applied to conformal prediction. I am not an expert in this specific area of conformal prediction, and I am surprised this connection has not appeared in the literature before, but to the best of my knowledge, it is novel.
- The paper is generally clear and well-structured, making it easy to follow. The proofs are also well detailed and, I believe, sound.

### Weaknesses
- The experiments are somewhat limited in the sense that the only baselines considered operate under a different setting. This makes it hard to evaluate the effectiveness of the proposed method. Are there no other conformal prediction methods designed for partial feedback, or at least methods that could be adapted to this setting?
- The presentation could be improved as there are a few minor typographic errors and occasional non-idiomatic phrasing (see minor issues below).

### Minor issues

- Line 16: “with fewer data generation assumptions” instead of “with less data generation assumption”
- Line 18: "mature literature" or "established literature" instead of “matured literatures”
- Line 41:  I am not sure “enjoyable” is a good word choice in “enjoyable guarantee”. Maybe just “guarantee” would read better.
- Line 126: “minimizes” should be “minimize”.
- Line 133: The sentence “due to exploiting the achievement of adversarial bandits under an adaptive adversary” is not clear. What does the “achievement of adversarial bandits” mean?
- Line 159: “which we leverage this algorithm as our baseline” does not read well. Probably better to add a full stop and then “We leverage this algorithm as our baseline”.
- Line 208: “method” instead of “methods”.
- Line 209: “to use” instead of “to using”.
- Line 568: Missing reference.

### Questions
1. After Lemma 1, it says that “the corresponding miscoverage rate converges to a desired level $\alpha$”. However, we do not really see that in the results of Figure 1. Why is that?
2. How are the set of candidate thresholds chosen? How does that affect the results in both theoretical and practical terms? I imagine that the resolution of the threshold might affect the convergence of the regret, for example.
3. The results focus on the absolute coverage gap, treating undercoverage and overcoverage equally. Would it be possible to extend the analysis to penalize undercoverage more heavily than overcoverage? Relatedly, EXP3.P-CP with Unlocking appears more conservative than EXP3.P-CP, leading to overcoverage in Figure 1. Is this behavior expected?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new framework for online conformal prediction under partial and adversarial feedback, where the learner only observes limited information about prediction correctness.  Conformal prediction has been connected to adversarial bandit learning by framing conformal set selection as an arm-selection problem. They develop a conversion lemma linking regret bounds to miscoverage guarantees, allowing the use of existing bandit algorithms for online conformal prediction.

### Strengths
- Establishes a clear connection between regret minimization and coverage guarantees in conformal prediction.

- The introduction of the EXP3.P-CP-UNLOCK algorithm effectively adapts the classic EXP3.P framework to the unique structure of conformal prediction feedback.

- Provides regret and miscoverage bounds with detailed proofs.

### Weaknesses
- The review of online conformal prediction is insufficient. Important recent works are missing, including:
   - Angelopoulos et al. (2023), Conformal PID Control for Time Series Prediction;
   - Angelopoulos et al. (2024), Online Conformal Prediction with Decaying Step Sizes;
   - Gibbs & Candès (2024), Conformal Inference for Online Prediction with Arbitrary Distribution Shifts;
   - Zhang et al. (2025), Online Differentially Private Conformal Prediction for Uncertainty Quantification.


- As $m_t$ is an indicator function, the regret formulation may not distinguish between observing or not observing $y_t$, potentially causing information loss. Clarification on how the true label feedback is used is needed.

- A formal theorem explicitly stating the miscoverage rate bound would make the theoretical results clearer.

- Please clarify whether specific techniques are used to address non-i.i.d. or non-exchangeable data.

- The differences from Ge et al. (2025) should be highlighted more clearly in terms of assumptions,method, and theoretical guarantees.

### Questions
- Should it be “minimize the miscoverage rate” in line 128 on page 3?

- There are no comparisons with other recent partial-feedback uncertainty quantification frameworks.

- Please discuss how sensitive the results are to the choice of $λ$ and the discretization level $K$.

### Soundness
3

### Presentation
2

### Contribution
3
