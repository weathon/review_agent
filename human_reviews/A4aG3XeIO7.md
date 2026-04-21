# Tuning-Free Bilevel Optimization: New Algorithms and Convergence Analysis

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Bilevel optimization has recently attracted considerable attention due to its abundant applications in machine learning problems. However, existing methods rely on prior knowledge of problem parameters to determine stepsizes, resulting in significant effort in tuning stepsizes when these parameters are unknown. In this paper, we propose two novel tuning-free algorithms, D-TFBO and S-TFBO. D-TFBO employs a double-loop structure with stepsizes adaptively adjusted by the "inverse of cumulative gradient norms" strategy. S-TFBO features a simpler fully single-loop structure that updates three variables simultaneously with a theory-motivated joint design of adaptive stepsizes for all variables. We provide a comprehensive convergence analysis for both algorithms and show that D-TFBO and S-TFBO respectively require $\mathcal{O}(\frac{1}{\epsilon})$ and $\mathcal{O}(\frac{1}{\epsilon}\log^4(\frac{1}{\epsilon}))$ iterations to find an $\epsilon$-accurate stationary point, (nearly) matching their well-tuned counterparts using the information of problem parameters. Experiments on various problems show that our methods achieve performance comparable to existing well-tuned approaches, while being more robust to the selection of initial stepsizes. 
To the best of our knowledge, our methods are the first to completely eliminate the need for stepsize tuning, while achieving theoretical guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies parameter-free algorithms for solving BLO, where the lower level problem is strongly convex and the upper level problem can be nonconvex. The authors propose two tuning-free algorithms that achieve the nearly same rate of the state of the art without knowing the parameter. The stationarity measure is the standard hyper gradient norm. Numerical experiments are conducted to show the effectiveness of the proposed methods.

### Strengths
1.This paper combines the parameter free methodology with BLO, which is a good direction to explore.

2.The convergence rates achieved by proposed methods are nearly optimal.

### Weaknesses
1. Under the strong convexity of the lower level problem, the BLO is similar to single level problems. This is due to the fact that there is a unique solution $y^*(x)$ for the lower-level problem given an arbitrary upper-level variable $x$, and then the BLO is reduced to the single level problem $\min_{x}\phi(x)=f(x,y^*(x))$. Even further, the gradient of $\phi$ can be computed, as presented in Sec. 3.1 of this paper. Given this, the authors should point out the novelty of the techniques used in this paper compared with single-level parameter-free methods.

2. Second-order information is needed for each iteration, which is computationally expensive. The authors may discuss how to approximate the used hessian matrix or a furture direction to develop methods that are both tuning-free and hessian-free.

### Questions
Non.

### Soundness
3

### Presentation
3

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
In this paper, the authors introduce two tuning-free algorithms, D-TFBO and S-TFBO to solve bilevel problems. D-TFBO employs a double-loop structure with stepsizes adaptively adjusted by the "inverse of cumulative gradient norms'' strategy. S-TFBO features a simpler fully singleloop structure that updates three variables simultaneously with a theory-motivated joint design of adaptive stepsizes for all variables. The authors demonstrate that D-TFBO and S-TFBO respectively require $\mathcal{O}(1/\epsilon)$ and $\mathcal{O}(1/\epsilon \log^4 (1/\epsilon))$ iterations to reach an $\epsilon$-accurate stationary point. The methods are the first to eliminate the need for stepsize tuning while achieving theoretical guarantees.

### Strengths
1. The paper is well-written and is easy to read.

2. This paper introduce two tuning-free algorithms, D-TFBO and S-TFBO to solve bilevel problems, which eliminate the need for stepsize tuning while achieving theoretical guarantees.

3. The complexity bounds of the proposed method (nearly) match their well-tuned counterparts using the information of problem parameters.

### Weaknesses
1. My primary concern is the update mode of the proposed algorithms. Many tuning-free algorithms do not require prior knowledge of the total number of iterations $T$ (e.g., [1]). However, this is not a significant drawback for me, as addressing bilevel problems with tuning-free algorithms in this context seems to be new.

2. The literature review should comprehensively compare the complexity results and the corresponding problem settings with other methods in a table.

[1] Marumo, Naoki, and Akiko Takeda. "Parameter-free accelerated gradient descent for nonconvex minimization." SIAM Journal on Optimization 34.2 (2024): 2093-2120.

### Questions
1. I do not understand the proposed observations in Remarks 1 and 2. The authors could discuss these observations in greater depth, such as by providing more detailed explanations in the main theorems and experiments.

2. I would like to know whether the proposed algorithm can work without the prior knowledge of the total number of iterations $T$ (cf. Weakness 1).

3. I would like to observe the progression of the loss function over time in the experimental results.

4. For other questions, please see the weaknesses above.

### Soundness
3

### Presentation
3

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
The paper introduces tuning-free bilevel optimization algorithms to eliminate the need for prior knowledge of problem-specific parameters. Theoretical convergence is derived for methods and experiments are provided.

### Strengths
The paper provides a detailed convergence analysis for the algorithms.

### Weaknesses
While I understand the importance of addressing tuning-free bilevel optimization, the main technical novelty of this work is unclear.

### Questions
1. How would the methods extend to more general bilevel problems?
2. Could you update citations from arXiv preprints to published versions where available?
3. How does the performance of these tuning-free methods compare directly to well-tuned bilevel optimization algorithms in terms of convergence rate?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper has proposed two tuning-free algorithms for stochastic bilevel optimization, D-TFBO and S-TFBO, which eliminates the need for stepsize tuning that depends on problem-specific parameters. D-TFBO follows a double-loop structure, while S-TFBO utilizes a fully single-loop approach with a joint design of adaptive stepsizes. Convergence rates for both methods are established, and numerical results on data hyper-cleaning, regularization selection, and coreset selection validate the effectiveness of the proposed algorithms.

### Strengths
Parameter tuning is challenging in bilevel optimization, as bilevel problems typically involve more hyperparameters than single-level learning. This paper presents two versions of tuning-free methods with both theoretical guarantees and experimental validation, making it a valuable contribution to the bilevel optimization community. 

Theoretical guarantees for the proposed methods are solid, and the numerical performance demonstrates a clear advantage, showcasing the effectiveness of these approaches.

### Weaknesses
This paper addresses only deterministic bilevel optimization, leaving it unclear whether the proposed technique is robust in stochastic settings.

### Questions
This paper proposes two versions of tuning-free bilevel algorithms. It appears that the single-loop algorithm offers better gradient complexity, while the double-loop algorithm demonstrates superior empirical performance. Could the authors comment on the reasons for this discrepancy—such as whether it stems from a less tight convergence rate for the double-loop method or something else? Additionally, guidance on when to choose the double-loop versus single-loop version would be helpful for practioner.

### Soundness
3

### Presentation
3

### Contribution
3
