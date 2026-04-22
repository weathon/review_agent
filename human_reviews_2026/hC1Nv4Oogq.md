# How to Retrain Online Models Optimally with Few Updates

- Avg Score: 4.80
- Decision: Reject
- Scores: 8, 2, 6, 4, 4

## Abstract
Retraining is the primary mechanism by which AI models can update their internal parameters in response to evolving environments, yet it is also one of the most costly operations. This raises a fundamental question: *how often must a model be retrained to achieve optimal performance over time?*  We address this problem in the framework of *online learning*, beginning with the classical but foundational case of i.i.d. realizable data. We show that retraining at every step is unnecessary: in most cases, only $O(\log T)$ updates suffice to achieve near-optimal risk, where $T$ is the number of steps. Furthermore, when the *learning curve* decays as $1/t^{\alpha}$ with $\alpha < 1$, as few as $O(\log \log T)$ updates are enough.  We design algorithms that achieve these guarantees, including adaptive methods that remain optimal when $\alpha$ is unknown, and extend our analysis to piecewise-stationary settings with *distribution shifts*. We also establish sharp impossibility results, proving that no universal algorithm exists without prior knowledge of the learning curve.  Together, these results provide the first precise characterization of optimal retraining frequency, bridging foundational theory with practical strategies for scalable AI.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper studies a central question in online learning: how frequently does one need to retrain an online model to maintain near-optimal cumulative predictive performance. Under mild assumptions, the authors establish two notable results:
1. With non-increasing learning curve, $\log T$ updates are sufficient for a schedule optimal up to constant factor
2. With power-law learning curve with $\alpha < 1$, only $\log \log T$ updates are required. It's noteworthy that $\alpha < 1$ aligns with empirical scaling-law observations in large language models, making the result particularly relevant. 

These results are further strengthened by
- An update schedule for case (2) that does not require knowing $\alpha$ in advance
- An extension to a piecewise-i.i.d. setting to address distribution shift
- A no-free-lunch result showing that optimal update rules are impossible without assumptions
- An empirical demonstration of (2).

### Strengths
- The authors study a timely and practically important question through a rigorous theoretical lens. The assumptions align well with empirical results, making the work highly relevant to current practice. 
- The theoretical contributions are strong and surprising. These results are well-complemented by $\alpha$-agnostic update schedules, no-free-lunch theorem, and extension to distribution shift setting.
- The writing and organization are clear, polished, and accessible
- The empirical results nicely corroborates the theoretical predictions that only doubly-exponentially fewer updates are required.

### Weaknesses
- The author does not provide empirical results for Eq. 7, which should be quite easy to collect.

### Questions
- On line 92, the authors point to Algorithm 6. Should this refer to Algorithm 2 instead? 
- Could the authors provide experimental results in a language-model setting? Given the strong motivation from LLM scaling-law behavior, such experiments would significantly enhance the practical relevance and appeal of the work

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the problem of determining the optimal retraining schedule for online learning models under limited update budgets. In a typical online learning setup, the model observes data sequentially and incurs prediction loss at each round. Retraining updates improve performance but are computationally expensive. The authors formalize a setting where only $S$ retraining events are allowed over a time horizon of $T$ steps and analyze the trade-off between retraining frequency and cumulative predictive risk.

The paper focuses primarily on the realizable i.i.d. and piecewise-stationary environments. Under these settings, the authors characterize optimal retraining intervals in terms of the learning curve decay rate. They show that for non-increasing learning curves, only $O(\log T)$ retraining events are sufficient to match the performance of continuous retraining within a constant factor. For power-law learning curves $L(t) \sim t^{-\alpha}$, they further prove that only $O(\log \log T)$ retrainings are needed to achieve near-optimal risk. The paper also proposes adaptive algorithms that do not require prior knowledge of the learning curve exponent and extends the analysis to piecewise-stationary distributions. Theoretical guarantees are supported by small-scale experiments.

### Strengths
1. The paper provides rigorous mathematical analysis and establishes sharp retraining-frequency guarantees.
2. The results offer interesting theoretical insights regarding the relationship between learning curves and update schedules.
3. The proposed algorithms are backed by solid proofs and demonstrate provable adaptivity to unknown parameters.

### Weaknesses
The primary concern lies in the core assumptions that motivate the problem. In practice, retraining is typically driven by distributional changes rather than simply the accumulation of more data from an unchanged distribution. However, the paper assumes i.i.d. or piecewise-stationary environments, where retraining due to distribution shift is either unnecessary (in the i.i.d. case) or explicitly handled by assumed segmentation (piecewise-stationary case). This disconnect raises concerns about the practical relevance of the problem formulation.  

As a result, the work appears to build an elegant theoretical framework atop assumptions that do not match the settings where retraining is most critical in real deployments. Although the analysis is technically sound, the paper's motivation does not convincingly justify why optimizing retraining schedules under stationary or near-stationary distributions yields meaningful practical benefit.

### Questions
1. Can the authors clarify concrete real-world scenarios where retraining frequency matters under stationary or piecewise-stationary distributions? 
2. In practice, how would the proposed methods generalize to environments where model drift is caused by gradual changes in the distribution rather than clean piecewise segments?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of optimal retraining frequency for online models, balancing predictive performance against the high computational costs of updating.The authors demonstrate, within an i.i.d. realizable online learning framework, that full retraining at every step is unnecessary, as $O(\log T)$ updates can achieve near-optimal risk. Furthermore, the paper provides algorithms that achieve these guarantees, including adaptive methods for unknown learning curve parameters and extensions to piecewise-stationary environments.

### Strengths
This paper makes an interesting contribution to the literature on online learning and continual learning. The theoretical characterization of retraining frequency is a valuable result, establishing that near-optimal risk can be achieved with exponentially fewer updates (e.g., $O(\log T)$ or $O(\log \log T)$) than a full update schedule, depending on the learning curve's decay rate.

A particularly compelling aspect of this work is the development of "risk-driven" update rules. The idea of monitoring the empirical risk to dynamically trigger updates is an essential and practical concept. This approach is a key strength of the paper and warrants the emphasis the authors have placed on it.

### Weaknesses
While the paper's contributions are noteworthy, there are several areas where it could be significantly improved. The comments below are offered as constructive suggestions.

* **Clarity of Introduction:** The paper would benefit from a revised introduction. The motivation for the problem, as well as a clear overview of the paper's specific goals and structure, could be articulated more clearly. Highlighting the "risk-driven" update mechanism as a core conceptual contribution earlier in the paper could also strengthen its positioning.
* **Formulation of Optimality:** The current formulation of "optimality" is based on minimizing the *number* of updates (S) for a given time horizon T. A significant suggestion is to reformulate the problem to account for the *cost* of retraining. As the authors allude to, the cost of retraining is not constant; it typically grows with the amount of data (*t*). An optimal schedule should ideally be defined by a total computational budget, not just a fixed number of updates.
* **Extension on Cost-Performance Trade-off:** Following the previous point, a valuable extension would be to analyze the trade-off curve plotting total computational cost (derived from an optimal schedule) against predictive performance as the total budget is varied.
* **Reliance on Strong Assumptions:** The main analysis relies on strong assumptions about the learning environment (e.g., i.i.d. realizable data, known learning curve shapes). The paper then moves to settings where these assumptions are relaxed, but the analysis feels somewhat heuristic.
* **Lack of Empirical Validation:** The most significant weakness of the current submission is the near-total lack of empirical validation. The theoretical claims are interesting, but they require empirical support to be fully convincing.
    * In settings where the paper's assumptions *are* met, experiments should be provided to demonstrate that the proposed schedules (e.g., from Theorem 2 or 3) are indeed optimal in practice.
    * In settings where assumptions are *not* met (e.g., unknown environment parameters), the robustness of the proposed methods must be tested.
    * The two main algorithms proposed (the adaptive algorithm for unknown $\alpha$ mentioned in the text and Algorithm 2 for piecewise-stationary data) currently lack sufficient empirical backing.
* **Suggested Application:** A very interesting application and extension would be to apply these scheduling approaches to a language modeling task. Even a small-scale experiment, testing the cases with both known and estimated exponents, would significantly strengthen the paper's practical relevance.
* **Minor Points:** The authors might consider a more descriptive title. The final conclusion section also appears unfinished and requires revision.

### Questions
1.  **Regarding Theorem 3:** The theorem statement provides a risk bound, but it does not explicitly define the schedule used to achieve it. The proof appears to rely on a specific schedule construction. Could the authors clarify in the theorem statement that this bound is achieved *by* a specific schedule, or clarify if it holds more generally?
2.  **A question on notation:** In Algorithm 2, line 5, the update condition uses both `log T` and `ln T`. Is the use of the natural logarithm (`ln`) intentional, or is this a typographical inconsistency? A similar question stands if "Algorithm 1" (which was not visible in the provided PDF) has a similar inconsistency, as noted by the reviewer.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work investigates the need for model retrains over time through the lens of online learning.
Specifically, they show that schedules with $O(\log T)$ retrains are sufficient to be within a constant factor of the
optimal risk (i.e., mean loss over time). This result is based on clean connections to the ``doubling trick'' in
online learning. The authors give tighter risk bounds in the case of certain
sublinear learning curves (e.g., power law in Theorem 3). Finally, the authors extend their analysis somewhat to
piecewise-stationary distributions with distribution shift and give some numerical results (Figure 2).

### Strengths
- This work provides a precise characterization of the trade-off between retrain frequency and predictive risk.
- Tighter analysis for sublinear learning curves of the form $O(1/t)$ and $O(1/t^\alpha)$, i.e., Theorem 2 and Theorem 3.
- Starts to study aspects of distribution shift, which is the most practical case.
- Algorithm 1: Adaptive algorithm for power law learning curves if exponent $\alpha$ isn't known.
- Nice concluding no-free lunch theorem (Theorem 7), showing that some prior information is needed to be competitive.

### Weaknesses
- Limited discussion about how this retrain schedule framework and the results relate to previous work.
- While the theoretical framework and results are clean, this manuscript could benefit from strong experiments.

### Questions
**Questions**
- [152] As $t$ increases we have access to more data. How (if any) can this
  work generalize to non-uniform retrain costs? Intuitively, the first rounds
  could be cheaper retrains.

**Misc**
- [130] Suggestion: Missing reference for learning under nonstationarity: "Learning Rate Schedules in the Presence of Distribution Shift" [Fahrbach et al., ICML 2023]

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides the first precise characterization of optimal retraining frequency for online models, demonstrating that retraining at every step is unnecessary. It proves that $O(\log T)$ updates are sufficient for general i.i.d. data, and a doubly-exponentially small $O(\log \log T)$ updates are sufficient when the learning curve follows a common power-law decay.

**Problem formulation**

The paper models online learning with a fixed retraining budget of $S$ oracle calls over a time horizon of $T$ steps. At each update time $\tau_s$, a predictor $h_s$ is trained, and this predictor is used for all predictions until the next update at $\tau_{s+1}$. The objective is to design the update schedule $\tau_1, \cdots, \tau_S$ to minimize the total cumulative risk, $Risk_{S,T} = \sum_{t=1}^{T}l(h_{s(t)}(X_{t}),Y_{t})$, where $s(t)$ is the index of the most recent update.

**Main results**

The main theoretical results show that for any non-increasing learning curve, $S=O(\log T)$ updates achieve a risk within a constant factor of the optimal full-update (S=T) risk. Furthermore, if the learning curve follows a power law $L(t) \sim 1/t^\alpha$ (with $\alpha < 1$), only $S=O(\log \log T)$ updates are needed to match the optimal full-update risk.

**Technique/algorithm**

The paper first proposes "fixed-design" schedules, such as a geometric "doubling trick" ($\tau_i = 2^{i-1}$), that depend on the general shape of the learning curve. To handle unknown environments, it develops "risk-driven" adaptive algorithms (like Algorithm 1) that monitor empirical losses to estimate the learning curve's parameters or detect distribution shifts (Algorithm 2), allowing the schedule to adapt and remain optimal.

**Experiment sumamry**

Experiments on the MNIST dataset validate the theory by comparing a full-update (single SGD step per round) baseline to the proposed $O(\log \log T)$ schedule. The results confirm that this "$\Phi$-schedule," using doubly-exponentially fewer updates, achieves a cumulative risk that is within a small constant factor ( e.g 3-4) of the baseline.

### Strengths
The paper formulates retraininig in the online learning framework and provides some theory.

### Weaknesses
The theory is largely expected from existing online learning technique.
Also the paper models retraining as a single "oracle call," which is experimentally treated as a full, computationally massive retrain on the entire data prefix, rather than a more practical incremental update.

### Questions
.

### Soundness
2

### Presentation
2

### Contribution
2
