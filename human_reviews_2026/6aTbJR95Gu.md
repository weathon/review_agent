# Parameter-Free Variance Reduced Zeroth-Order Optimization for Non-Convex Problems

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
eroth-order optimization has become a vital tool for solving black-box learning problems where explicit gradients are unavailable. However, standard zeroth-order methods typically require careful tuning of algorithmic parameters such as the smoothing parameter and step size, which limits their practicality. In this paper, we propose PF-VRZO(Parameter free variance reduced zeroth-order methods), a novel parameter-free variance-reduced zeroth-order optimization framework for nonconvex finite-sum problems. Our method only requires minimal input information—problem dimension $d$ and sample size $n$—and adaptively adjusts the smoothing and step size parameters during the optimization process. We develop two algorithmic variants based on coordinate-wise and random-direction gradient estimators, respectively. We establish non-asymptotic convergence guarantees showing that PF-VRZO achieves  function query complexity of $\widetilde{\mathcal{O}}(d\sqrt{n}\epsilon^{-2})$ for finding stationary points. Additionally, we conduct experiments on non-convex phase retrieval and distributional robust optimization to validate the effectiveness of our method.
To the best of our knowledge, PF-VRZO is the first parameter-free zeroth-order algorithm that incorporates variance reduction techniques tailored specifically for nonconvex optimization problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes PF-VRZO, a parameter-free, variance-reduced zeroth-order framework for nonconvex finite-sum problems. It gives two variants (coordinate-wise and random-direction) that combine SPIDER-type variance reduction with adaptive step size $\gamma_t$ and adaptive smoothing $\mu_t$.

### Strengths
Precise wording and clear organization. The paper defines notation early, states assumptions explicitly, and keeps terminology consistent across sections. Algorithms, lemmas, and theorems are numbered and referenced cleanly, which makes the logical flow easy to follow.

Motivation is well articulated. The authors clearly diagnose the tuning sensitivity of zeroth-order methods and motivate combining variance reduction with adaptive step size and smoothing. The problem setting, design goals, and why the proposed choices address concrete pain points are laid out without ambiguity.

### Weaknesses
**Question 1**  It would help to state clearly what "parameter-free" means here.  The coordinate variant incurs $O(d)$ function calls.
 
**Question 2**  Positioning relative to prior work on reducing dimension constrain. Some ZO methods mitigate dimension effects. A brief note on what this work does better-overall query complexity under comparable assumptions, robustness/constant factors, or ease of use-would help readers understand the advantage.

**Question 3**  Distinctiveness beyond ZO-SPIDER. Since the recursion resembles SPIDER, it would be useful to highlight the distinctive elements and show they matter-perhaps with a small ablation against a tuned ZO-SPIDER under the same query budget and a short proof pointer indicating where standard SPIDER would fail without your $\mu_t$ design.

**Question 4**  When should practitioners choose coordinate or random directions?

### Questions
And some minor issues:
1. Missing spaces after punctuation, e.g., “dimension d,without relying…” → “dimension d, without relying…”
2. Mixed notation of “non-convex” and “nonconvex”; use one form consistently (preferably nonconvex).
3. In figure captions: “Improperly tuned Parameters” → “Improperly tuned parameters.”
4. References to algorithms (e.g., “ZO-SGD,PF-VRZO,ZO-SPIDER”) lack spacing; should be “ZO-SGD, PF-VRZO, ZO-SPIDER.”

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PF-VRZO (Parameter-Free Variance Reduced Zeroth-Order), a parameter-free optimization method for non-convex and finite-sum problems in black-box settings. The proposed method obtains the optimal query complexity for non-convex optimization problem.

### Strengths
The proof sketch is presented clearly. It builds upon the framework of the previous work SPIDER and effectively incorporates the properties of the zeroth-order optimization method.

### Weaknesses
1. The proof process is somewhat difficult to follow. For instance, in the proof of Lemma B.2 and Line 379, it is unclear how the authors derive the bound for $x_s - x_{s-1}$. In addition, there are several typos. For example, in Line 906, the author should check the exponent of T in the second term; the definition of $\Phi(T)$ given below is inconsistent with that in Line 900—the second term should be $\mathcal{O}(n^2)$ as stated in the proof.
The authors should carefully check the proof and correct typos to enhance readability.

2. The experimental evaluation is not sufficient. The input data dimensions in both datasets are relatively small. I suggest that the authors include additional experiments on larger or more diverse datasets, to better demonstrate the scalability and generalization of the proposed method.

### Questions
Since the proof is not entirely clear to me, my main concerns are related to the theoretical analysis:
- In the proof of Lemma B.2 and Line 379, it is unclear how the authors derive the bound for $x_s - x_{s-1}$;
- It is also not clear how Lemma A.7 is applied to obtain the inequality in Line 942;
- how does the parameter learning rate affect the proof analysis

In Line 63, the authors demonstrate that 'As acknowledged by the authors, extending this result to the nonconvex setting is nontrivial'. Could the authors elaborate on how their proof sketch differs from spider, and clarify which part of the extension to the nonconvex setting is nontrivial?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors studied parameter-free algorithms for stochastic nonconvex optimization in the zeroth-order setting. They proposed PF-VRZO, a parameter-free variance-reduced zeroth-order optimization method, which requires only the problem dimension and sample size as inputs. The method adaptively adjusts both the smoothing parameter and the step size, eliminating the need for manual hyperparameter tuning. It introduces two algorithmic variants, leveraging coordination-wise and random-direction estimators respectively. The authors prove non-asymptotic convergence guarantees, showing PF-VRZO reaches a function-query complexity of $\tilde{\mathcal{O}}(d\sqrt{n}\epsilon^{-2})$. They further validate the method empirically on non-convex phase retrieval and distributionally robust optimization.

### Strengths
- The idea of parameter-free zeroth-order nonconvex optimization is interesting and novel.
- The method removes the need for tuning smoothing and step-size parameters, which is an advantage in practice.

### Weaknesses
- The paper is not well written. For example, Section 3.5 mostly consists of lemmas / theorems, but lacks explanation about their significance and connection. Also, there are many typos / formatting issues (please see below).
- The experiment results are not very promising. The proposed method does not consistently outperform existing methods across different datasets and metrics.

### Questions
- Questions
  - Section 2 is very compact. Is it necessary to put the content in a separate section?
  - In Remark 1 and 3, how to derive the $\mathcal{O}(1+ n/n)$ complexity?
  - In the experiments, the random variant performs better than the coordinate variant in terms of sample complexity. But why does the coordinate variant perform better in terms of running time?
- Suggestions
  - The captions of the figures are short. I suggest the authors provide more explanation about the figure in the caption.
  - It would be better to have some texts explaining Algorithm 1.
  - The palettes of the figures are not consistent. I suggest the authors use the same color for the same method across different figures.
- Typos
  - Line 60: (Ren & Luo, 2025) -> Ren & Luo (2025)
  - Line 185: (Gao et al., 2018) -> Gao et al. (2018)
  - Line 189: unbiased estimator of \hat{\nabla} -> unbiased estimator of \nabla
  - Line 192: random-direction -> Random-direction

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies zero-th order optimization of a smooth non-convex objective that is written as finite sum of n smooth terms.
The proposed algorithm is parameters free and implement variance reduction.
The authors prove that time-averaged norm of the gradient converges below \eps in a number of stochastic steps that is of order at most d*n^{1/2}/\eps^2 (where d is the ambient dimension and n is the number of terms in the optimization.

Recent work on parameters free zeroth order optimization was limited to convex functions (but also proved stronger results, namely convergence of the cost).

### Strengths
1. Zeroth order, parameters free optimization is of course a well motivated area of study.
2. The results appear to be novel and sound hence filling a gap in earlier literature.

### Weaknesses
1. The algorithm seems a variant over earlier parameter-free papers (Ivgi et al 2023; and follow up work on first order method also cited in the present paper), as combined with smoothing techniques for gradient estimation using zero-th order methods (eg Duchi et al 2015 and follow up work, Ren Luo 2025)

2. Also for what concerns the analysis, the paper seems closely related to proof techniques from earlier literature on zeroth order methods (in particular Ji et al 2019).

3. I understand that proving that the average gradient converges to zero has become the standard form of results in this subcommunity. However, I am still not sure why one should care about this from a machine learning perspective, since the algorithm can converge on a suboptimal minimum.

### Questions
The first questions are related to the points mentioned above:

1) Can you please spell out the algorithmic innovation. Is it the specific choice of \mu_t? 
If so I would at least explain upfront  where this choice comes from in the analysis.

2) Similarly for the algorithm analysis. Is there a new proof component with respect to earlier proofs?

### Soundness
3

### Presentation
2

### Contribution
3
