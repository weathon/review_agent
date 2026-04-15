# Uniform Localized Convergence and Sharper Generalization Bounds for Minimax Problems

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Minimax problems have achieved widely success in machine learning such as adversarial training, robust optimization, reinforcement learning. Existing studies focus on minimax problems with specific \text{algorithms} in stochastic optimization, with only a few work on generalization performance. Current generalization bounds almost all depend on stability, which need case-by-case analyses for specific \text{algorithms}. Additionally, recent work provides the $O(\sqrt{d/n})$ generalization bound in expectation based on uniform convergence. In this paper, we study the generalization bounds measured by the gradients of primal functions using the uniform localized convergence. We relax the Lipschitz continuity assumption and give a sharper high probability generalization bound for nonconvex-strongly-concave (NC-SC) stochastic minimax problems considering the localized information. Furthermore, we provide dimension-independent results under Polyak-Lojasiewicz condition for the outer layer. Based on the uniform localized convergence, we analyze some popular \text{algorithms} such as the empirical saddle point (ESP), gradient descent ascent (GDA) and stochastic gradient descent ascent (SGDA) and improve the generalization bounds for primal functions. We can even gain approximate $O(1/n^2)$ excess primal risk bounds with further assumptions that the optimal population risks are small, which, to the best of our knowledge, are the sharpest results in minimax problems.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper derives generalization bounds for learning with minimax objectives, assuming strong convexity of the dual problem and smoothness of the objective function. It establishes uniform convergence bounds for primal function gradients and provides convergence bounds for the empirical primal function to the primal risk function under a PL condition. The paper also applies these findings to algorithms like empirical saddle point, gradient descent ascent, and stochastic gradient descent ascent.

### Strengths
The paper presents a high-probability analysis for minimax problems which is more challenging than bounds in expectation.
It studies several popular algorithms such as empirical saddle point, gradient descent ascent, and stochastic gradient descent ascent.

### Weaknesses
some comments are listed below

1.  the novelty of the proof techniques is not clear to me. Most of them are just incremental compared with the existing work

2. The proof seems to have issues.  For example, the problem is in Eq (27). It only holds if $x^*$ is the projection of x to the set of solutions. That is if the problem does not have a unique solution, this x^* should be different for different x. Then one cannot apply Lemma 11 since Lemma 11 only holds for a fixed $x$.

Then Eq (24) does not hold since it is derived by applying Lemma 11 for all $x$

### Questions
See above

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Post rebuttal comment: I would raise the score to 5. 

===================
This paper investigates stochastic minimax optimization, focusing on high-probability generalization performance gauged by primal function gradients. It extends prior work by replacing Lipschitz continuity requirements with Bernstein conditions in NC-SC setting, broadening the SC-SC setting to PL-SC setting, demonstrating the generalization bounds for GDA and GDmax in SC-SC setting, and generalizing results from minimization problems to minimax cases where the optimal error is low.

### Strengths
The paper extends prior work by replacing Lipschitz continuity requirements with Bernstein conditions in NC-SC setting, broadening the SC-SC setting to PL-SC setting, demonstrating the generalization bounds for GDA and GDmax in SC-SC setting, and generalizing results from minimization problems to minimax cases where the optimal error is low.

### Weaknesses
The paper lacks a cohesive narrative, as the extensions presented are not tightly interconnected and could be considered in isolation. Specifically, Section 5 appears unrelated to the preceding sections, making the overall structure disjointed. This fragmentation complicates the assessment of the paper's technical contributions.

### Questions
1. How do the extensions contribute to a unified narrative on the generalization of minimax optimization?
2. What constitutes the paper's key technical contribution, and why is it challenging?
3. How does Theorem 6 differs from existing research, specifically [Lin et al., 2020]?
4. Why is the generalization error analysis limited to the SC-SC setting for GDA and SGDA, rather than extending to the NC-SC setting?
5. Given that the PL condition is a natural extension of strong convexity, how does the analysis in Section 5.1 differ from that in the SC-SC setting? Is a minor modification of SC-SC results sufficient?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors study generalization bounds for minimax problems. They provide high-probability bounds for the gradient of the primal function under a nonconvex strongly concave setting. They also provide a dimension-independent result if the outer layer satisfies PL condition. They apply their results to existing algorithms and establish sharp bounds for excess primal risk.

### Strengths
The authors have presented good theoretical results. The principal contribution appears to be the introduction of a high-probability variant (Theorem 1) of the generalization error bound, originally proposed by Zhang et al. (2022) in expectation. Additionally, Theorem 3 offers a dimension-independent bound, albeit with some additional assumptions.

### Weaknesses
The paper was challenging to follow due to its unclear presentation and the presence of cumbersome sentences. These issues disrupted the flow of the text and, on occasion, led to confusion.

- I encountered the term, "almost" $O(\cdot)$, in the paper, but I'm uncertain about its exact meaning. Does this imply that the order of certain elements is not calculated with high accuracy?
- Certain remarks in the paper are challenging to comprehend due to their presentation. For instance, Remark 5 suggests that specific quantities are small in Theorem 3 and (8), but it does not specify their orders or the precise conditions under which they are considered small.
- In Remark 8, it is stated that $\phi(x^*) = O(\frac{1}{n})$ is a common occurrence. Can you provide examples to illustrate this assertion?

It is not my intention to imply that these claims are untrue, but it would greatly enhance the paper's quality if they were substantiated mathematically, especially considering the predominantly theoretical nature of the contributions in the paper.

### Questions
Please check weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work, authors introduce a new framework of uniform localized convergence based on Bernstein condtion, which allows them to derive sharper high-probability generalization bounds for these problems. Their approach relaxes some standard assumptions, such as Lipschitz continuity, and provides dimension-independent results under certain conditions.

### Strengths
First high-probability convergence result compared to existing literature. Second, the paper relaxes some restrictive assumptions in the literature.

### Weaknesses
The Bernstein condition introduces an intriguing alternative to the usual bounded gradient assumption, but as far as I can see, here the new condition is more like a bounded (higher-order) moment setting while it is required to hold for any moment.
1. I'm intrigued by the role of $n$ here in your Assumption 3. Should we interpret it as the sample size? If that's the case, it would require practitioners to verify a prohibitive number of moments to ensure compliance with the assumption, and additionally, to define the parameter $B$.
2. Are there more examples, beyond the one in Remark 2, where the Bernstein condition is met but not the bounded gradient norm? It would be beneficial to see this condition applied in more varied contexts.

In comparison with Zhang et al., 2022, the authors claim a more refined result, yet this advantage is only evident only when $||x-x^*||\leq 1/n$, which basically suggests the point $x$ must be exceedingly close to the optimum.

3. Regarding the potentially very large sample size, I concerned that the feasibility of such a condition seems questionable, especially with large sample sizes, potentially diminishing the practical relevance of the results.
4. In the NC-SC setting, the optimal point $x^*$ may not be unique (should be a set $X^*$), while here the authors implicitly assume the uniqueness, I think a better notation can be $\text{dist}(x, X^*)$.

In conclusion, the paper ambitiously addresses a broad spectrum of minimax challenges. While the primary contributions could be articulated more clearly, and the practical application seems somewhat elusive. But I am open for further discussions. Thank you.

### Questions
See weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
