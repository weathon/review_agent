# Approximation of the Gompertz trend with a multilogistic function

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 2

## Abstract
The paper deals with the comparison of the Gompertz function and the logistic function. We show that the Gompertz trend can be approximated with high accuracy by a sum of three logistic functions (multilogistic function). Two of them are increasing, and one is decreasing. We use second-order logistic wavelets to estimate the parameters of the multilogistic function.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes approximating the Gompertz function by a sum of three logistic (sigmoid) functions. The parameters of these logistic components are identified using the Continuous Wavelet Transform with logistic wavelets. 
The approximation achieves a very close numerical approximation (Rsquared approx 0.999985) for a specific Gompertz trend with fixed parameters. A potential interpretation of the three components as representing different growth phases (acceleration, inhibition, and reinforcement) is proposed. No theoretical proofs or general results are provided. The paper demonstrates the idea empirically on one synthetic example.

### Strengths
1. Using logistic wavelets to approximate a Gompertz trend is, as dar as I can see, original.
2. The exposition is well structured and easy to follow.
3. I agree that the result that three logistic terms yield such a high-quality approximation could be useful for applied modeling.
4. The approach might extend to other S-shaped growth models or data-driven growth processes. So this example could be the basis for a more comprehensive project.

### Weaknesses
1. The scope is limited. Only one specific Gompertz function is tested. The method’s behavior for general parameters or noisy data is not explored. It is vaguely claimed that the approach generalizes, but a formal statement is missing.
2. No mathematical guarantees are given, e.g., no proof that three logistic terms are needed for such an error, or, more interestingly, how the approximation error behaves as more terms are added.
3. It is unclear if the wavelet based approach finds the best three teem approximation.
4. The “interpretation” of the three logistic waves (growth, inhibition, boost) is speculative rather than analytically grounded.
5. No concrete application, where the three term approximation is used to some advantage, is presented.
6. The work seems too narrow and descriptive for a top-tier conference like ICLR.

### Questions
1. Can the authors provide theoretical guarantees or error bounds for approximating a Gompertz function with n logistic terms?
2. Is the three-term representation optimal in some sense?
3. How does the approximation quality scale with the number of logistic components? (For more than 3.)
4. Can the method be generalized to Gompertz functions with arbitrary parameters or to noisy empirical data?
5. What would be the advantage of the wavelet-based parameter identification compared to a direct regression fit (e.g., LASSO)?
6. Are there practical applications or case studies where this method provides a measurable advantage?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper deals with showing that the Gompertz trend can be approximated with high accuracy
by a sum of three logistic functions (multilogistic function), and utilize second-order logistic wavelets to estimate
the parameters of the logistic function.

### Strengths
The paper would have had a strength if the contributions were indeed novel, however it has been long known that a Gompertz function is itself a logistic function, so estimating it with a sum of other logistic functions is already well known, routine and doesn't tell us anything new.

I strongly believe the authors of this paper are not human.

### Weaknesses
The paper would have had a strength if the contributions were indeed novel, however it has been long known that a Gompertz function is itself a logistic function, so estimating it with a sum of other logistic functions is already well known, routine and doesn't tell us anything new.

I strongly believe the authors of this paper are not human.

### Questions
Nil

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper provides a demonstration that a Gompertz curve can be approximated, with high accuracy, by a multilogistic function consisting of three logistic functions.

### Strengths
The main strength of the paper is its succinctness and depth of presentation.

### Weaknesses
The main weakness of of the appear is its possible lack of relevance to the ICLR community. Although there are uses of Gompertz curves in machine learning the authors make no effort to draw out these connections or illustrate the relevance of their work to a machine learning community.

### Questions
What is the relevance of this work to the machine learning community?

### Soundness
3

### Presentation
2

### Contribution
2
