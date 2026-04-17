# Non-Asymptotic Analysis of (Sticky) Track-and-Stop

- Decision: Accept (Oral)
- Scores: 6, 8, 4, 6

## Abstract
In pure exploration problems, a statistician sequentially collects information to answer a question about some stochastic and unknown environment. The probability of returning a wrong answer should not exceed a maximum risk parameter $\delta$ and good algorithms make as few queries to the environment as possible. The Track-and-Stop algorithm is a pioneering method to solve these problems. Specifically, it is well-known that it enjoys asymptotic optimality sample complexity guarantees for $\delta \to 0$ whenever the map from the environment to its correct answers is single-valued (e.g., best-arm identification with a unique optimal arm). The Sticky Track-and-Stop algorithm extends these results to settings where, for each environment, there might exist multiple correct answers (e.g., $\epsilon$-optimal arm identification). Although both methods are optimal in the asymptotic regime, their non-asymptotic guarantees remain unknown. In this work, we fill this gap and provide non-asymptotic guarantees for both algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a non-asymptotic analysis of the performance of the track-and-stop and sticky track-and-stop algorithms for pure exploration tasks under fixed confidence. The results are non-asymptotic, meaning they hold for confidence levels that do not necessarily converge to zero. The main results are given in Theorems 1 and 2, where the sample complexity upper bounds are stated in an implicit form. More explicit expressions are provided in Appendix G.

### Strengths
•	The problem of pure exploration in moderate-confidence regimes is notoriously challenging and worth investigating—see “The Simulator: Understanding Adaptive Sampling in the Moderate-Confidence Regime” by Simchowitz et al., COLT 2017, for relevant discussions (this paper should likely be cited and discussed). Deriving guarantees in this regime is non-trivial, and this appears to be what the authors have successfully achieved here.
•	The upper bounds, although implicit, seem to be more interpretable and user-friendly than those derived in existing works such as Degenne et al. (2019).

### Weaknesses
•	There is no comparison with existing results in moderate-confidence regimes (e.g., Degenne et al., 2019). This is an important drawback—the paper focuses on the track-and-stop algorithm, but it remains important to compare its performance with other approaches.
•	There is no numerical experiments! How should we know whether the new upper bounds are reasonable?

### Questions
•	I find it hard to believe that no assumption on the mapping $i^\star$ is required. Is the analysis valid for any function, even without regularity assumptions? Does the characteristic time $T^\star(\mu)$ become infinite when regularity is absent?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper revisits the popular pure-exploration algorithms, track-and-stop and sticky track-and-stop. These algorithms are known to have optimal performance in the asymptotic regime, i.e., when the error probability $\delta \to 0$. In this work, the authors analyze these classical algorithm in the finite time-regime and provide non-asymptotic bounds on expected stopping time of these algorithms as a function of $\delta$ and $T^{\star}(\mu)$, the problem specific hardness.

### Strengths
The paper provides non-asymptotic bounds for these popular pure exploration algorithms, thereby addressing a gap in our understanding of the finite-time performance of these algorithms. 

The analysis seems rigorous and is clearly explained. Overall, the paper is good and I like it.

### Weaknesses
I don't see any obvjous weakness in the paper, but I have questions as outlined in the next section.

### Questions
From the expression of $T_0(\delta)$, it seems that $T^{\star}(\mu) \log(1/\delta)$ becomes the dominant term only for very small values of $\delta = \mathcal{O}(e^{-K})$. For values larger than this, the term $KT^{\star}(\mu) \log(\log(1/\delta))$ is dominant. This implies that the optimality still holds for only sufficiently small $\delta$. I agree that it common in non-asymptotic analyses to have a threshold only beyond which the results. However, $\delta = \mathcal{O}(e^{-K})$ this choice seems really small for any practical purpose. Ofcourse, this also not because of any leading constants as I have just denoted the order. Can the authors elaborate more on this?

How large would you typically expect $T_{\mu}$ to be, or equivalently how small would $\epsilon_{\mu}$ be? Is there some structure on $\mathcal{M}$ that would allow us to have more explicit bounds on $\epsilon_{\mu}$? This might help shed light on the sub-optimality of the current bounds.

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
2

### Summary
This paper studies the non-asymptotic performance of the Track-and-Stop (TaS) and Sticky-Track-and-Stop (S-TaS) algorithms for best-arm identification. The authors derive upper bounds on the expected number of arm pulls required for both algorithms, using a carefully constructed sequence of events leading to stopping. The main contribution lies in providing finite-time performance guarantees for these two algorithms, complementing their known asymptotic optimality results.

The paper should be rejected due to two main issues:
(1) insufficient contextualization with respect to closely related work, and
(2) a lack of motivation for analyzing the vanilla versions of the algorithms rather than their improved variants.

**Main argumentation**
While the paper is technically sound and written with care, it does not clearly articulate what new insight is gained from analyzing the vanilla (S-)TaS algorithms, given that closely related works already provide non-asymptotic analyzes for more stable variants. Without a stronger justification of novelty or comparison with those results, the contribution remains poor.

### Strengths
- The paper is well written and clearly structured.
- The assumptions are well motivated, and the mathematical arguments are rigorous and accompanied by clear intuition.
- The work contributes to a deeper understanding of the stopping behavior of TaS-type algorithms in finite-sample regimes.

### Weaknesses
- *Relation to Degenne et al. (2019) and Barrier et al. (2022) is underdeveloped*

Both works introduce algorithmic variants of Track-and-Stop that address its early-phase instability and already provide non-asymptotic analyses.
The paper does not sufficiently motivate why analyzing the vanilla TaS (and S-TaS) provides new insight beyond those results.

The authors claim that the work of Degenne et al. (2019) requires to solve a “more challenging optimization problem,” but do not specify what distinguishes their setting or why this difference matters here.

The discussion of Barrier et al. (2022) is minimal. The paper should clarify in what sense its bounds differ from or improve upon theirs.

- The results are said to complement the observed empirical good performance of TaS, yet no empirical evaluation is provided to assess the tightness or practical relevance of the derived bounds.

- The introduction of S-TaS as a solution to "general multiple-answer problems" is insufficiently motivated. It is unclear what new problem instances this algorithm targets or why these instances are practically or theoretically important.

### Questions
- Why is it valuable to analyze the vanilla Track-and-Stop, given that stabilized variants (Degenne et al., Barrier et al.) already offer non-asymptotic guarantees and often perform better empirically?
- What are the “general multiple-answer problems” mentioned, and why can’t previous approaches be extended to handle them?
- In what scenarios would one prefer using TaS or S-TaS over the existing improved variants?
- Have you evaluated whether your non-asymptotic bounds align with the empirical stopping times observed in practice?

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the non-asymptotic behavior of the Track and Stop algorithm, along with its variant, Sticky Track and Stop, which handles cases with multiple correct answers. This is a heavily theoretical paper. While Track and Stop has been around for several years and its empirical performance has been repeatedly validated showing it to be a strong and hard-to-beat baseline, its non-asymptotic analysis has been known to be challenging. This paper serves as a strong addition to the field, with its primary contribution being the non-asymptotic analysis for both the standard algorithm and its 'Sticky' variant.

### Strengths
1. The paper is well-written, with a clear structure, strong motivation, and a thorough discussion of related work. Overall, it is very easy to follow.
2. The theoretical result is a welcome addition to this field, as a non-asymptotic analysis for Track and Stop has been long-missing.

### Weaknesses
I did not find any obvious weaknesses. However, due to time constraints, I was unable to review the proof details thoroughly.

### Questions
For the 'Sticky' variant, which addresses multiple correct answers, a comparison to the findings in "Revisiting simple regret: Fast rates for returning a good arm" (ICML 2023) would be insightful. That work, though in a fixed-budget setting, discovered that the identification speed can be significantly faster (potentially at a $1/M$ rate, where $M$ is the number of good arms). It would be interesting for the authors to discuss whether a similar, weaker, or analogous result is possible in the fixed-confidence setting. This is currently difficult to determine, as the complexity bounds for Sticky Track and Stop in this paper are not expressed in an explicit form that allows for an easy comparison.

### Soundness
3

### Presentation
3

### Contribution
3
