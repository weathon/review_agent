## Human Reviewer 1

### Summary
Summary:
The main contribution of this paper is demonstrating how partially calibrated forecasts can be used in decision making. The authors apply a robust optimization perspective: They assume that the forecaster $f$ satisfies a weak notion of calibration called $\mathcal{H}$-calibration under the (unknown) true distribution. Then, they construct the set of distributions (which can be reduced to a set of candidate conditional expectations) under which the forecaster $f$ satisfies $\mathcal{H}$-calibration. By assumption, the true distribution must lie in this set. They propose to find the decision rule that maximizes the worst-case utility over the uncertainty set – which provides robust performance guarantees under the true distribution. They provide a characterization of the optimal robust policy under the uncertainty set. They find that when $\mathcal{H}$-calibration corresponds to notions of decision calibration, the optimal robust policy simply best responds to the forecast.

### Strengths
Overall, I found this paper to be clear and well-written.
To the best of my knowledge, this is the first time I have seen the connection between "partial" calibration and robust optimization -- I believe this paper proposes a novel question and framework, and also provides a clear exploration of the solution.
This is a strong contribution to the calibration literature and a very nice application of robust optimization.

### Weaknesses
I have questions about how the connection to this framework and decision calibration should be interpreted: To the best of my understanding, decision calibration is the “necessary” and “sufficient” condition on probabilistic forecasts required to recover the expected utility (under the true distribution), e.g. Zhao et al, 2021, Sahoo et al, 2021. 
Is it the case if $\mathcal{H}$ corresponds to a set of test functions for the appropriate notion of decision calibration for our decision task, then every distribution (or conditional expectation) in the ambiguity set $\mathcal{Q}$ yields the same expected utility (which is the true expected utility)? 
Furthermore, does this explain the “collapse” of the optimal minimax decision rule to the decision rule that treats the forecasts “as-is” when $\mathcal{H}$ corresponds to a decision calibration notion? 
If this intuition is correct, it may be worth highlighting for readers in the main text to explain where the optimal minimax policy (for decision calibration classes) comes from.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
* The paper studies decision-making with partially calibrated predictors. Particularly, the paper studies $\mathcal{H}$-calibration, a relaxation of full calibration, which also generalizes several existing calibration notions. 

* It shows that best-responding to a decision-calibrated predictor is minimax-optimal, giving a simple, practical rule for decision makers.

* With weaker calibration (moment/bin-wise), it derives the tractable best-response policy.

* Empirics illustrate the robust rule's behavior vs. a naive plug-in policy.

### Strengths
* I really like the justification for decision calibration. The exponential sample complexity has been a problem for full calibration. The paper proves that, once satisfied, simple best response remains minimax-optimal. While full calibration is not efficiently achievable, this result is actionable for practice: decision makers need only best-respond; forecasters should focus on the concrete decision calibration objective. 

* Bridges theory and empirical: connects common training artifacts (e.g., MSE first-order moments, binning) to usable robustness guarantees.

### Weaknesses
* The paper would benefit from a sensitivity analysis. For example, how does decision calibration loss contribute to the loss of minimax-responding decision makers? I would imagine a linear bound. 

* The paper would also benefit from a discussion of the efficiency of decision calibration, given the motivation of the exponential sample complexity for achieving full calibration. 


* Typos and minor issues:
  - Line 71: it's -> its
  - Line 203, citation format should be \citealp
  - Line 475, Section number missing
  - Line 132: a notable exception is Hu & Wu (2024) who do give a weaker notion than full calibration. The notion in the paper is equivalent to full calibration once the error is 0.

### Questions
I do not have outstanding questions, see my comments in weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper studies the calibration of predictive forecasts, inspired by ideas such as multicalibration.
They consider the idea of "$\mathcal{H}$-calibration", which states that a forecaster $f : \mathcal{X} \to [0,1]^d$ is calibrated if $\mathbb{E}[h(f(X)) \cdot (Y - f(X))] = 0$ for all $h \in \mathcal{H}$.
The authors analyze optimal decision-making under $\mathcal{H}$-calibration, for varying choices of $\mathcal{H}$.
A key result is that, once $\mathcal{H}$ is "rich enough" -- which they characterize by a notion of "decision calibration", weaker than the usual full calibration -- the calibration guarantee becomes already strong enough to the point of recovering the optimal decision properties of full calibration.
Optimal decision rules are also derived for some weaker choices of $\mathcal{H}$.

### Strengths
The paper tackles the important problem of calibration, and in particular that of achieving tractable yet powerful notions of calibration.
The result regarding the "decision calibration" choice of $\mathcal{H}$ is quite interesting, though I'm unsure whether decision calibration is truly achievable in practice.

Overall, I believe the results are a meaningful contribution to the field, from both a theoretical point-of-view (characterization of different $\mathcal{H}$s and their optimal decision rules) and a practical point-of-view (identification of certain practical notions of calibration and optimal decision rules).

### Weaknesses
My major concern is regarding the assumption that the utility be linear in the outcome $y$ (stated in Remark 2.1; line 210/211). It seems really strong, and I'm slightly unconvinced on how realistic it is.
Could the authors give examples besides simple expectations of (i) utilities satisfying this, and (ii) utilities *not* satisfying this? In particular, I am interested in whether there are any meaningful utilities other than simple expectations that will satisfy this.
It would also be nice to have a brief explanation on why this is necessary in the paper (as far as I can tell, this is used in the proof of Theorem 3.1 so that both Sion's minimax theorem and convexity of the adversarial problem hold; I would appreciate it if the authors could confirm or correct me). \
(It's also worth mentioning that perhaps this assumption should be in a `\begin{assumption} \end{assumption}` environment, rather than in a remark.)

I am also slightly unsure as to the actual novelty of the practical choices of $\mathcal{H}$, and in particular the general cases when (i) $\mathcal{H}$ is a finite dimensional subspace; and (ii) self-orthogonality under squared loss.
Nonetheless, I believe the paper stands on its own even if these indeed have limited novelty.
(In the case where $h$ has $X$ as the input, I am sure that these are not new, especially the finite dimensional case; but when $h$ has the forecast $f(X)$ as input I am less sure.)

Overall, this leads me to a score of 'borderline accept'. \
**Why not higher:** while I'm fairly confident with the papers contribution re. decision calibration, I am less sure of how meaningful its contributions on the weaker forms of calibration actually are. I'm also somewhat concerned with the strength of the linearity assumption on the utility. \
**Why not lower:** The paper is sound and has interesting results on a relevant problem. Its conclusions can be of broader impact, both theoretical and practical. This makes me lean towards acceptance.

These aside, I only have some minor comments:

- Line 044/045: are the observed outcomes $Y$ really in $[0, 1]^d$? I would have expected them to be in something more like $\{1, \ldots, d\}$. As far as I can tell this intended to be like a slightly more general form of "one-hot encoding" of outcomes in $\{1, \ldots, d\}$ (which would be more consistent with e.g. $E[Y | f(X) = v] = v$); am I correct?
- Lines 324-326, "Decision calibration is a minimal, task-specific threshold at which robust decision making and plug-in best-response coincide, [...]": is it really clear that there is no $\mathcal{H}$ smaller than decision calibration that still enables optimality of the plug-in best-response? I ask because of the use of "minimal" here.
- Typos / writing / formatting:
    - Line 063/064: large language models are generally neural networks, so "from neural networks to large language models" does not make much sense
    - Line 088: "informally speaking this family" -> "informally speaking, this family"
    - Line 166: "for every $\mathcal{H}$" -> "for every $h \in \mathcal{H}$"
    - Line 210: "Throughout this Paper" -> "Throughout this paper" (capitalization)
    - Line 475: broken Section `\ref`

### Questions
- Could the authors discuss the applicability of the linearity assumption, and elaborate on its necessity?
- Could the authors clarify the novelty of the analyses on the practical choices of $\mathcal{H}$?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4