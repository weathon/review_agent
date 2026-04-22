# DR-Submodular Maximization with Stochastic Biased Gradients: Classical and Quantum Gradient Algorithms

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
In this work, we investigate DR-submodular maximization using stochastic biased gradients, which is a more realistic but challenging setting than stochastic unbiased gradients. We first generalize the Lyapunov framework to incorporate biased stochastic gradients, characterizing the adverse impacts of bias and noise. Leveraging this framework, we consider not only conventional constraints but also a novel constraint class: convex sets with a largest element, which naturally arises in applications such as resource allocations. For this constraint, we propose an $1/e$ approximation algorithm for non-monotone DR-submodular maximization, surpassing the hardness result $1/4$ for general convex constraints. As a direct application of stochastic biased gradients, we consider zero-order DR-submodular maximization and introduce both classical and quantum gradient estimation algorithms. In each constraint we consider, while retaining the same approximation ratio, the iteration complexity of our classical zero-order algorithms is $O(\epsilon^{-3})$, matching that of stochastic unbiased gradients; our quantum zero-order algorithms reach $O(\epsilon^{-1})$ iteration complexity, on par with classical first-order algorithms, demonstrating quantum acceleration and validated in numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies a quite general optimization problem, namely maximizing a function F on [0,1]^n under some contraints defining a feasible set $\cal C \in [0,1]^n$. The function is DR-submodular. The feasible set is convex. Such a problem is typically solved using gradient ascend, and there is a huge literature on these techniques.

There are 3 contributions.

1. Extending the Lyapunov framework, to allow gradients to be imprecise, with a bias and a noise.
2. Proposition an 1/e approximation for non-monotone DR-submodular maximization over a convex set. The novelty is the assumption on a largest feasible point. This overcomes a 1/4 upper bound on the general optimization problem.
3. Providing a quantum algorithm, based on the improved quantum Jordan algorithm from 2023, which has the same performance guarantees as its classical counterpart but improves in cubic iteration time.

Some performance guarantees are proven, and experiments are conducted with standard benchmarks.

My background is too weak in this area to judge the results.

### Strengths
The domain is important and has a huge literature. The paper has 3 contributions, all of which seem central and important. The paper has a theoretical and a practical side.

### Weaknesses
The paper seems to be hard to follow for an outsider.

### Questions
- at some moment it would be good to say what DR stands for (diminishing return)
- Page 3 line 127. I think you mean $\textbf{x} \vee \textbf{y} = (\max\\{x_i, y_i\\})_{i\in [d]}$
- Also here dimension is d and later it is n
- Page 3, line 150. I could not understand the difference of a bias and a noise. Is the noise consistent, in the sense that n is a deterministic function? What is the domain of $\xi$? What is known to the algorithm? Does it know the functions $b,n$ and the parameter $\xi$ or only some of them? Does the algorithm knows the assumed bounds $m,\eta_b, M, \eta_n$?
- Page 4 line 205. It wasn't clear to me before that the function x(t) is an algorithm.
- Page 5 line 228. What does it mean to maximize two values? Or do you mean to maximize the maximum of the two values?
- Page 6 line 277: which we have newly introduced -> Potential author name revealing
- Page 8 line 418 Lipschiz -> Lipschitz 
- Page 10. Some acronyms should be uppercase, i.e. in curly brackets in the bibtex file. Such as DR, SGD.

### Soundness
3

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
2

### Summary
This paper studies continuous DR-submodular maximization under stochastic biased gradient oracles. It extends Du's Lyapunov framework to handle bias and variance in gradient estimators. Based on this, it provides approximation algorithms under three constraint classes: general convex, down-closed convex, and convex sets with a largest element. The paper develops zeroth-order algorithms, where the classical version achieves $O(\epsilon^{-3})$ while the quantum version achieves $O(\epsilon^{-1})$ iteration complexity, matching the performance of classical first-order methods.

### Strengths
- Extending Du's Lyapunov framework to stochastic biased gradients seems interesting. The framework explicitly characterize the effect of bias and noise, and the resulting analysis looks useful beyond this problem setting.
- The constraint set given by a convex set with a largest element is well-motivated. Bridging the convex and down-closed settings leads to a provable $1/e$ guarantee.
- The paper shows that quantum gradient estimation can close the gap to first-order methods in ratio and complexity.

### Weaknesses
- The experiments are limited to $d=3$ due to simulating quantum algorithms in classical computers. To improve empirical evidence, larger-scale tests would make the claim of quantum acceleration more convincing.

### Questions
- How tight is the $1/e$ approximation for convex sets with a largest element? 
- Would it be possible to provide alternative constructions for $a(t)$ and $b(t)4 that possibly improve the current setup?

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
3

### Summary
This paper investigates the problem of continuous DR-submodular maximization under the practically relevant yet theoretically challenging setting of stochastic biased gradients. The authors extend the Lyapunov framework, traditionally developed for exact or unbiased stochastic gradients, to handle gradient estimators that contain both bias and noise, thereby characterizing their effects on convergence and approximation guarantees. They further introduce a new class of constraints, namely convex sets with a largest element, that naturally arise in resource allocation and similar applications. Under this setting, the paper proposes a $1/e$-approximation algorithm for non-monotone DR-submodular maximization, which surpasses the known $1/4$ hardness bound for general convex sets. Building upon this framework, the authors design both classical and quantum zero-order algorithms, showing that the quantum version achieves the same approximation ratio with only $O(\varepsilon^{-1})$ iteration complexity, demonstrating a quantum acceleration compared with classical zero-order methods that require $O(\varepsilon^{-3})$. Numerical experiments on quadratic and coverage-type DR-submodular functions validate the theoretical results, showing that quantum algorithms converge faster and achieve comparable solution quality to classical first-order methods. Overall, the paper provides a unified theoretical and algorithmic treatment of DR-submodular maximization in the presence of biased gradients and connects classical optimization with emerging quantum techniques.

### Strengths
The paper extends the Lyapunov-based analytical framework to accommodate stochastic biased gradients, a setting that more faithfully represents real-world learning and optimization scenarios where gradient estimates are noisy and biased. The work also successfully integrates quantum computation into continuous submodular optimization, showing that quantum zero-order algorithms can match the convergence rate of classical first-order methods, offering a clear demonstration of quantum speedup. The results are rigorously proven, the methodology is well grounded in prior literature, and the experimental findings, though small in scale, corroborate the theoretical analysis.

### Weaknesses
The paper currently lacks experimental results on runtime performance, which makes it difficult to assess the practical efficiency of the proposed algorithms.

Moreover, it would be beneficial if the authors could provide additional real-world examples or application scenarios to better justify the practical relevance of the studied problem, since DR-submodular maximization has so far attracted more attention from the theoretical community rather than from most of the ICLR audience.

Minor comment: In Figure 1, the legends in the third and sixth subplots appear partially covered by light-colored text, suggesting a plotting or rendering issue that should be corrected for clarity.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors consider the problem of DR-submodular maximization.  They consider a few combinations of monotone/non-monotone functions over the hypercube and classes of convex constraints (general, down-closed, largest element).  The authors first propose an extension of a Lyapunov framework from exact to stochastic and biased gradients.  The authors consider a new constraint setting (largest element) for the non-monotone setting and obtain an improved approximation ratio (over using a general convex region based method).  The authors also show significant improvements in complexity for the value oracle setting using a quantum algorithm for gradient estimation.  Lastly, the authors run several experiments to demonstrate the improvements of the quantum based method.

### Strengths
- The authors present a new approximation ratio for the case of non-monotone DR-submodular maximization over convex constraints with a largest element.
- The authors show that there is a notable convergence speedup using quantum algorithms for gradient estimation (for the value oracle setting).
- The authors extend the Lyapunov framework for DR-submodular maximization to handle stochastic and biased oracles.
- The authors include experiments that show improved convergence for quantum algorithms in the value oracle setting.

### Weaknesses
- I had some uncertainty about the extent of technical challenges and novelty for some parts
    - For the Lyapunov extension to handle stochastic biased gradients, it was unclear to me from the description in the main section what technical challenges were encountered compared to past works.   Could the authors summarize the challenges?
        - in line 047 (Du, 2022) is cited as having a unifying Lyapunov framework unifying many previous methods.  That reference is not brought up in Section 3 for the framework, though the authors are upfront that they generalize a previous framework to handle stochastic (and biased) gradients.
        - Du’s work applied the Lyapunov framework for DR-submodular, and though that work presumed exact oracles, for convex optimization at least has there been Lyapunov based approaches that handled stochastic and biased (or at least stochastic) gradients? If so, are there unique challenges in extending the Lyapunov framework for DR-submodular problems from exact to stochastic (and biased) gradients? 
    - For the quantum section, the authors adopt a quantum algorithm for gradient estimation for DR-submodular maximization similar to past works in convex optimization.  In light of past works (both from classic gradient estimation in DR-submodular works and wrt Augustino et al 2025 for quantum methods in convex optimization), were there some key steps that were particularly challenging?  


### Minor
- Table 1 I’d suggest listing DR-max approx. bounds (and complexities) for stochastic first order based methods for reference.  

- Non-monotone max with largest element is new setting considered, so appx ratio is first, but not clear  how tight it may be (no lower bound)

- Theorem statements referencing algorithms were imprecise, 
    - eg line 360 “returned by quantum algorithms satisfies” – do the bounds in Theorem 2 hold for any quantum algorithms, or the (single) specific algorithm adopted from van Apeldoorn et al 2023?  
    - Theorem 3 line 377 “there are some quantum zero-order algorithms achieving” 
- Fig 1 the axes’ fonts are too small

### Very Minor
- “Lipschitz” not “Lipschiz”
- line 407 “with [A]ssumption[s] 1 …”
- line 366 “The query complexity of the value oracle” should that be the complexity of the algorithm?

### Questions
- Do the complexity bounds depend on the dimension?
- I found Section 4 on quantum acceleration hard to follow.  I am not familiar with quantum methods.  I have some uncertainty about the specific set up and some uncertainty about whether the impressive complexity results using quantum algorithms are purely of theoretic interest or if in the (near) future there could be the potential for real-world use.
    - From line 354, just to confirm, the set up is identical in terms of the environment (the (biased) value oracle)?  A learner that has access to a quantum computer can use quantum Jordan algorithms to achieve the speedup in terms of query complexity over a learner that only has access to classical computers, but the environment itself and how they interact with the environment is identical?
    - Could the authors remark on example situations where there could be a practical benefit in terms of total run-time?  Eg Figure 1 is measured in terms of iterations.  How would that map to clock time?  I understand that in the experiments the authors were simulating a quantum algorithm on a classical computer, so there could be large overhead.     
        - For readers unaware of how much overhead, just looking at Fig 1 results it might be tempting to consider using a simulated quantum algorithm even for a smaller number of iterations if the overhead is low enough.  What were the (rough) run-times?
        - Would the authors be familiar enough with current quantum computers on the market if they would be big enough to be used for this type of problem? If not, is there an educated guess for how close quantum computing is to the scale even to be used for the small $d=3$ experiments here?
        - For current/near future quantum systems, is there a rough sense for how long each iteration (for the quantum Jordan algorithm) might take in clock time?

### Soundness
3

### Presentation
2

### Contribution
2
