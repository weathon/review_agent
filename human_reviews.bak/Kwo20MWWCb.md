# An Asynchronous Bundle Method for Distributed Learning Problems

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
We propose a novel asynchronous bundle method to solve distributed learning problems. Compared to existing asynchronous methods, our algorithm computes the next iterate based on a more accurate approximation of the objective function and does not require any prior  information about the maximal information delay in the system. This makes the proposed method fast and easy to tune. We prove that the algorithm converges in both deterministic and stochastic (mini-batch) settings, and quantify how the convergence times depend on the level of asynchrony. The practical advantages of our method are illustrated through numerical experiments on classification problems of varying complexities and scales.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents a bundle method for asynchronous distributed optimization. By increasing the server-side complexity, it boosts the performance. The algorithm operates under unknown (bounded) delays, while favorable is the effort for automated parameter tuning.

### Strengths
+ Useful algorithmic contributions (especially embodying parameter auto-tuning).
+ Rigorous analysis (incorporating cases of practical interest such as inexact server solutions and mini-batch processing).
+ Well-written and clear paper (especially commendable is that the authors provide insights for all choices as well as their analysis).
+ Rich, well-presented, and convincing experiments.

### Weaknesses
- Dubious scalability in large systems ($n \gg 1$).

Others--outweighed by the strengths--are that the contribution is an extension of previous works and that the code is not presently shared.

### Questions
Questions:

1. Your analysis in (7) applies across a subsequence ($\exists k\ge K$). Can this be strengthened? For example, can you make assertions for the convergence of $\min_{j\le k} \{F(x_j) - F(x^\star)\}$? How does it compare with the literature?
Can you also comment on the reason for the discrepancy between (7) and (8) in terms of the same (the latter applies $\forall k$).
2. Can you include the adaptive tuning aspect ($\hat{L}_i$) in the analysis, or provide a remark sketching the same?
3. Can you please comment on means to boost scalability in large systems ($n \gg 1$)?


Suggestions:

- Please include the cost of solving the master problem ($O(nmd)$) in the main paper. 
- Can you rephrase the statement in lines 362-365? It seems a bit sloppy for your fluent presentation style. 
- Please explain the reasoning in line 458 (why in this case the master problem is more challenging).
- Please elaborate on "potentially discarding outdated information" in line 532 (in relation to dynamic bundle size selection).

Editing:

- line 55: "insensitive" does not seem accurate and can be misleading; please rephrase.
- line 88: counterpart -> counterparts.
- lines 108-111: please rephrase to prevent any misperception of a contradictory statement (gradients provide local; in contrast, gradients provide global).
- I suggest numbering the equation in line 207. 
- line 256: the central server for each worker $i$ -> each worker $i$ (to prevent any confusion that the server has access to user data).
- line 295: delete "weak".
- line 314: rephrase "gracefully".
- line 490: no tuning -> no tuning in these experiments (to avoid overstating).

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript describes a asynchronous bundle method for distributed optimization. The results are presented well and the manuscript is well-written and clear. However, I have the following concerns that somehow limit the scope of the results.

1) Novelty
 
Bundle methods have been around for quite some time and have been also around in the machine learning community, since the 2000s (there is e.g. a well-cited paper by A. Smola). Similarly, the fact that bundle methods can be used for asynchronous parallel optimization has also been discussed several years ago (around the 2010s). Hence, the conceptual novelty of the work is very limited. 

On the technical side, early works seem to have focused on non-smooth optimization (which arguably is also the setting where bundle methods are very adequate for), while the current work looks at (essentially) strongly convex composite objectives that have a smooth and non-smooth part. In contrast to earlier results, the manuscript claims sublinear/linear convergence to a neighborhood of the solution (8). However, the current statement of Theorem 4.5/Theorem 4.10 does not prove linear/sublinear convergence of the iterates, but rather for a subsequence of the iterates. 
As such, the theorems do still allow for an arbitrarily slow convergence of the iterates (the variable k can be arbitrarily large). The usefulness of deriving a convergence rate for the subsequence is unclear to me, unless this is used as an intermediate step to argue that the entire sequence converges at the given rate (which is not done).


2) Experimental evaluation

The experiments are limited and consist of (regularized) logistic regression with only a very small set of workers (n=9). This does not seem convincing to a machine learning audience for two reasons: i) the community would be interested in deploying the method at scale (meaning several 1000 workers). ii) the community would likely use more elaborate models (NNs, transformers, etc) for classification. 
Can the authors argue about scalability of the method with respect to the number of workers? How does the method empirically perform on nonconvex problems?

--

Minor questions, discussion:
- Can the authors comment on the role of the variable "m" in the convergence results Theorem 4.5/Theorem 4.10? As far as I see, the variable doesn't play a role in the results.

- Is there a good way to reduce the memory overhead? As is, the central server needs to store O(mn) gradients, where n denotes the number of workers. This seems really impractical for large scale problem instances.

### Strengths
see above

### Weaknesses
see above

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a new asynchronous optimization method for solving distributed convex problems. The proposed method approximates each local function using a piece-linear model constructed from a fixed-window of past updates. The main advantage of the proposed method is that it does not assume a maximum time delay and takes performs regularization at the server. The experiments demonstrate the effectiveness of the bundle method in asynchronous settings and interestingly shows a significant improvement over standard approaches.

### Strengths
The authors provide a strong argument for their methodology. It seems intuitive that using past updates can help estimate the local function

### Weaknesses
The reviewer wonders if the authors can provide more detail on how the weighting is chosen. It seems that the weighting should be chosen also based on the time-delay

### Questions
The reviewer wonders if the authors can comment if this applies to settings where each worker has an individual time-dela, $\tau$.

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
The authors in this paper propose an asynchronous bundle method to solve distributed learning problems with parameter server architecture. The algorithm constructs a piecewise linear model to approximate the local loss of each worker and uses this model to compute the next iterate. This method enjoys a more accurate approximation of the objective function, compared to other first-order asynchronous algorithms. The authors prove that their algorithm can converge to a neighborhood of the solution for both deterministic and stochastic settings. In addtion, they show how the convergence results depend on several key parameters, such as upper delay bound $\tau$, approximation error $\delta$ and noise variance $\sigama_1^2, \sigama_2^2$. They also report some numerical simulations to validate the effectiveness of their proposed algorithm.

### Strengths
- The proposed algorithm seems novel to the reviewer, and is fast and easy to tune in practice.
- The authors provide convergence guarantee for their algorithm and clearly show that how some key parameters (e.g., $\tau$, $\delta$, $\sigma_1^2$, $\sigma_2^2$) affect the convergence rate and steady error term.
- The paper is clearly structured and seems technically correct.

### Weaknesses
- There are no proper comparisons of theoretical results of the baseline algorithms: i.e., to DAve-RPG[1] and PIAG [2].
- The data heterogeneity analysis is missing. It is unclear how the proposed algorithm will perform in the case of strong data heterogeneity  in the numerical experiments.
- The scalability of the proposed algorithm is not well addreesed; note that only a system with 9 workers is considered in the numerical experiments. 

[1] Konstantin Mishchenko, Franck Iutzeler, Jer´ome Malick, and Massih-Reza Amini. A Delay-tolerant Proximal-Gradient Algorithm for Distributed Learning. Proceedings of the 35th International Conference on Machine Learning, 2018.

[2] Xuyang Wu, Sindri Magnusson, Hamid Reza Feyzmahdavian, and Mikael Johansson. Delay-Adaptive Step-sizes for Asynchronous Learning. Proceedings of the 39th International Conference on Machine Learning, 2022.

### Questions
- The reviewer would like to know how their algorithm improves the complexity compaerd to the baselines DAve-RPG[1] and PIAG [2].
- How can the upper bound of delay $\tau$ be guaranteed by setting bundle size $m$ in their practical experients, or in ther words, how is the value of $\tau$ related to the value of $m$ in practice?
-  How does the proposed algorithm scale in terms of the node number? Can the authors provide some numerical experiments to illustrate this?
- What’s the limitation of the third one in Assumption 4.9?

### Soundness
3

### Presentation
3

### Contribution
3
