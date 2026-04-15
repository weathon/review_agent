# Constrained Bayesian Optimization with Adaptive Active Learning of Unknown Constraints

- Decision: Reject
- Scores: 1, 3, 5, 6

## Abstract
Optimizing objectives under constraints, where both the objectives and constraints are black box functions, is a common scenario in real-world applications such as scientific experimental design, design of medical therapies, and industrial process optimization. One popular approach to handling these complex scenarios is Bayesian Optimization (BO).
In terms of theoretical behavior, BO is relatively well understood in the unconstrained setting, where its principles have been well explored and validated. However, when it comes to constrained Bayesian optimization (CBO), the existing framework often relies on heuristics or approximations without the same level of theoretical guarantees.
 
In this paper, we delve into the theoretical and practical aspects of constrained Bayesian optimization, where the objective and constraints can be independently evaluated and are subject to noise. By recognizing that both the objective and constraints can help identify high-confidence **regions of interest** (ROI), we propose an efficient CBO framework that intersects the ROIs identified from each aspect to determine the general ROI. The ROI, coupled with a novel acquisition function that adaptively balances the optimization of the objective and the identification of feasible regions, enables us to derive rigorous theoretical justifications for its performance. We showcase the efficiency and robustness of our proposed CBO framework through empirical evidence and discuss the fundamental challenge of deriving practical regret bounds for CBO algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a novel approach for combining active learning of unknown constraints with Bayesian optimization (CBO). The active learning criterion identifies regions of interest (ROI) based on confidence bounds on the output of the constraint function, and subsequently performs BO (UCB specifically) within the ROI. Theoretical analysis of the proposed algorithm is presented. Lastly, CBO is compared to other relevant algorithms, showing favorable results on synthetic and realistic tasks.

### Strengths
- __The methods section__ is clearly presented piece by piece, which makes it simple to digest.
- __Related works__ appear thorough, and cover relevant works from both the constrained BO and active learning litterature.
- __Results__ span from highly synthetic (aimed at showing traits of the algorithm rather than pure performance) to more realistic tasks.

### Weaknesses
To summarize, I believe this paper has substantial flaws both in terms of the novelty of the method, the theoretical analysis, and overall presentation. Unfortunately, this results in a paper that is far from publishable in its current state. Specifically:

- __The active learning component__ (the difference between the confidence bounds) is oddly presented and is not novel. The authors recognize this, but fail to recognize that the active learning method harks back to at least MacKay (1992) and has been widely applied since. Global variance reduction should not, in my opinion, be presented as a novel aspect of any algorithm. Moreover, it should be presented as Global variance reduction, or ALM (Active Learning MacKay), or otherwise, but _not_ as a difference between confidence bounds. This simply disguises the method as something seemingly more complex than it is. 

- __Theoretical analysis__ contains substantial flaws, starting with Assumption 3:

  ___Assumption 3__ Given a proper choice of $\beta_t$ that is non-increasing, the confidence interval shrinks monotonically. (...)_.
This is flawed for two reasons:

  1. It appears trivial (beta is non-increasing, so the confidence interval will naturally be smaller as beta shrinks). _If there is a nuance to this assumption that I am missing, I encourage the authors to enlighten me._
  2. Assuming a non-increasing $\beta_t$ goes against a decade of BO convention, set by Srinivas et. al. (2009) that beta should scale as $\mathcal{O}(\log t)$.
Moreover, the subsequent Lemma 1 and Theorem have numerous un-introduced quantities ($K, \pi_t$, and appears to run contrary to Assumption 3. In Theorem 1,

 $\beta_t =2 \log( 2K+1)|D_{X_t}|$ 
where $D_{X_t}$ should be increasing in the data, making $\beta_t$ increase over time - thus not non-increasing. Admittedly, $D_{X_t}$ is the _intersection_ between a discretization of the search space and the data, which clearly contains only the data, at most. The non-increasing property of $\beta$ is re-stated in the Appendix, suggesting it is not a mistake.

The theoretical results are difficult to properly assess due to these apparent issues and ambiguities, so I will refrain from commenting on the correctness of the proofs in the appendix until these issues are clarified. 

- __The algorithm__ _"Maximize the acquisition values from different aspects:"_ is informal, and line 11 is not well-defined. What does the $\text{argmax}$ over $\mathcal{G}$ (but not $x$) mean? Moreover, $x_{g, t}$ in line 12 has not been obtained from any prior step. Should the $g_t$ in line 11 really be $x_{g,t }$? As such, it is still not clear to me what the resulting acquisition function is, (but I believe it is the $\text{argmax}_{g, x}$ over all constraint and BO acquisitions, jointly.

- __The figures__ are difficult to parse due to the odd color choices (bright grey and dark orange on orange in Fig. 1, tiny legends everywhere).

- __Competing methods__ do not seem to be properly implemented. SCBO (Eriksson and Poloczek, 2019) builds on TuRBO, one of the more robust BO methods around. Seeing it fail on most tasks is a warning sign, and suggests to me that there are implementation flaws. If the authors believe it has been correctly implemented and used, I would suggest they motivate its poor performance in their specific setting.

__Minor__: 
- Enumeration of contributions is generally a nice-to-have, and adds clarity.
- Hyperlinks are missing completely. Consider adding these.
- I would argue that the CBO acronym is occupied by Gardner et. al. (2014). While I don't believe there is a rule governing acronyms, I would strongly suggest for the authors to change it.
 

__Additional References__:

David MacKay. Information-based objective functions for active data selection. _Neural Computation_, 1992.

### Questions
In addition to the weaknesses (which certainly poses a few questions), I am curious about the following:

- The constraint functions $g$ are assumed to be continous-valued. Is this conventional, and is the method ammenable to non- continous (i.e. binary output) constraints?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This study presents a solution to the constrained Bayesian optimization problem with theoretical analysis. The method involves defining high-confidence regions of interest (ROIs) for both the objective and constraints. Following this, an acquisition function is constructed to simultaneously optimize the objective and identify feasible regions. The effectiveness of the proposed algorithm is showcased through numerous experiments.

### Strengths
It is uncommon to find a paper on constrained Bayesian Optimization that includes theoretical analysis. This paper explores both coupled and decoupled settings, with the decoupled setting detailed in the appendix.

### Weaknesses
Unfortunately, the paper lacks clarity and precision, with several confusing and potentially erroneous arguments.

1. The high-level approach is inefficient for the following reasons:

   a. $x_t$ can be picked as $x\_{\\mathcal{C}\_k,t} = \\text{arg}\\!\\max\_{x \\in U\_{\\mathcal{C}\_k,t}} \\alpha\_{\\mathcal{C}\_k,t}(x)$, so it may belong to $L_{\mathcal{C}_{k'},t}$ for some $k' \neq k$. In this case, it raises the question of why the objective and constraints are evaluated at an infeasible input.

   b. In equation (5), when $\text{LCB}_{f,t,\max} = \infty$, we basically pick the most uncertain $f(x)$. However, there may existing uncertain $f(x)$ with a very low evaluation. Then, why do we need to pick this input if we are interested in large $f(x)$? 

2. Here are some arguments that might be incorrect:

   a. The paper asserts that existing constrained Bayesian optimization lacks theoretical analysis. However, the work titled "No-Regret Bayesian Optimization with Unknown Equality and Inequality Constraints using Exact Penalty Functions" by Lu and Paulson (2022) provides theoretical analysis.

   b. The value of $\beta_t$ is defined based on $\hat{X}_t$ but $\hat{X}_t$ is defined based on $\beta_t$. Therefore, $\beta_t$ cannot be defined in Theorem 1. 

   c. In the proof of Theorem 1 (proof of Lemma A.1): Lemma 5.4 of Srinivas et al. (2009) is not applied correctly. Lemma 5.4 is intended for *points selected* in Srinivas et al. (2009). However, in this paper, it is applied to the sequence of $x_{g,t}$ for all $g \in \mathcal{G}$ and $t$. It is important to note that not all $x_{g,t}$ are *selected points*: At each iteration only 1 input $x_{g,t}$ of a function $g \in \mathcal{G}$ is picked as the candidate to evaluate.
   
3. There are several unclear points: 

   a. The paper may need more arguments to explain why the approach of Takeno et al. (2022) "violates the theoretical soundness".

   b. Theorem 1 does not address feasibility. Furthermore, Srinivas et al. (2009) analyze the regret of *some input* (for example $x_t$). While this paper introduces a definition of regret (or reward) of inputs, it does not use this definition in the theoretical analysis. This raises questions about the significance of Theorem 1 in terms of the algorithm's achieved reward (or regret).

   c. The abstract highlights a setting where "the objective and constraints can be independently evaluated", yet the paper primarily focuses on the coupled setting where they are evaluated together. The decoupled setting is briefly explained in the appendix without any supporting experiments.

   d. The paper introduces $\epsilon_C$ for theoretical analysis but does not utilize it in the algorithm. If the algorithm is used with $\epsilon_C = 0$, the theoretical analysis loses its significance as $T \ge \infty$.

   e. The abstract mentions a discussion on "the fundamental challenge of deriving practical regret bounds for CBO algorithms", but this discussion cannot be located in the paper."


4. There are other issues such as

   a. The chosen performance metric is simple regret. However, in practical applications, identifying the input with minimum regret is unknown. Thus, the paper does not address what input Bayesian Optimization recommends as the optimal solution.

   b. The experiment section is quite short. Surprisingly, in the Rastrigin-1D-1C, the discrete search space consists of $1000$ points but the proposed algorithm reaches the optimum with $2000$ iterations. Furthermore the noise is small $\mathcal{N}(0,0.1)$ compared to the function range $[-40,0]$. This efficiency is questionable.

   c. Assumption 2 does not hold for equality constraints.

5. There are some typos in the paper:

   a. In equation (5), $\infty$ should be replaced with $-\infty$.

   b. Before equation (1), GP\supscript(()

   c. In the appendix, there are several typos: thoeretical, defination, Cauchy-Schwar.

   d. Both $\epsilon_k$ and $\epsilon_C$ refer to the same thing.
   
   e. The label of the proposed algorithm in Figures 3 and 4 should be COBALT instead of CBO.

   f. It's worth noting that the name COBALT coincides with an existing Bayesian Optimization work titled 'COBALT: COnstrained Bayesian optimizAtion of computationaLly expensive grey-box models exploiting derivaTive information' by Paulson and Lu (2021).

### Questions
Please answer to the aforementioned weaknesses.

### Soundness
2 fair

### Presentation
2 fair

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
This paper proposes a new algorithm for constrained Bayesian optimization. The acquisition function for the objective is a modified UCB, and the acquisition function for the constraints are marginal variance. In each iteration, the algorithm selects a query point by maximizing the maximum over the objective's and constraints' acquisition functions. The authors have proved an upper bound on the number of samples to find an $\epsilon$ confidence interval containing the global maximum. Empirically, they have shown that the proposed method achieves superior convergence speed compared with other constrained Bayesian optimization baselines.

### Strengths
- The theoretical result seems to be the first non-asymptotic sample complexity guarantee for constrained Bayesian optimization.
- According to the plots in Figure 3 and Figure 4, it looks like the method is often much better than the baselines when the constraint is more challenging (where the feasible set is a small fraction of the domain).

### Weaknesses
- Some important experimental details are missing.
    - It is not clear how to numerically check if $LCB_{f, t, \max}$ is finite or not. To do this, one needs to check if the intersection of $S_{C, t}$ is nonempty, where each super level set $S_{C, t}$ is a non-convex set. It is unclear from the paper how the intersection is computed.
    - It is unclear how the hyperparameter $\beta_t$ is set in the experiments. The constant $\beta_t$ balances the exploration and exploitation and has huge impact on the practical performance.
- The exact condition on the discretization $\tilde D$ is unclear in the statements of Lemma 1 and Theorem 1. According to Lemma 1, $\tilde D$ can be an arbitrary discretization of the domain as long as it contains the global optimum. However, I don't quite think this can be true. For example, the discretization of Srinivas et al. (2009) has to be large enough (exponential) to roughly cover the entire domain. Otherwise, the concentration inequality only holds inside the discretization $\tilde D$ and cannot be generalized to the entire domain.
- The proof relies on an additional assumption that the confidence interval has to shrinks monotonically, which is somewhat non-standard in the analysis of BO algorithms. Though, the author claims they can apply a proof technique by Gotovos et al. (2013) even if this assumption is violated.

### Questions
- The Assumption 3 needs $\beta_t$ to be non-increasing. However, subsequently $\beta_t = 2 \log (2 (K + 1) |\tilde D| \pi_t / \delta)$ which is increasing in $t$, given that $\pi_t$ has to grow superlinearly (typically quadratically). Can you comment on this discrepancy?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes COBALT, a constrained Bayesian optimization algorithm. It models the unknown objective and constraints with Gaussian processes. Regions of interest (ROIs) are defined for the objective and each constraint based on confidence intervals. Acquisition functions are defined on the ROIs. The next point maximizes the combined acquisition functions over the ROI intersection, enabling an adaptive tradeoff between learning constraints and optimization. Theoretical analysis shows the confidence interval around the global optimum shrinks below a threshold after enough iterations. Experiments demonstrate COBALT efficiently finds the global optimum compared to existing methods, benefitting from active constraint learning. In summary, COBALT introduces a principled constrained Bayesian optimization framework, leveraging ideas from active learning and Bayesian optimization via acquisition functions and ROIs for the adaptive constraint-objective tradeoff.

### Strengths
This paper presents an original contribution to the field of constrained Bayesian optimization. The key strength is the principled integration of ideas from both Bayesian optimization and active learning of constraints to develop the novel COBALT algorithm. Defining separate acquisition functions on regions of interest for the objective and constraints enables explicit, adaptive tradeoff between exploring constraints vs exploiting the objective. This combination of existing methods with new problem formulations results in an algorithm with theoretical guarantees on optimizing unknown black box functions subject to unknown constraints. Experiments demonstrate superiority over current state-of-the-art techniques on challenging synthetic and real-world optimization tasks.

### Weaknesses
While the COBALT algorithm represents a significant advance, there are a few areas where the work could be strengthened:
* Empirical evaluation is limited to 6 relatively low-dimensional tasks. Testing on more high-dimensional real-world problems would better showcase scalability.
* The theoretical analysis relies on several assumptions that may not always hold in practice, such as the independence of the GPs and the existence of a feasible solution.

### Questions
* The theoretical analysis relies on assumptions like GP independence and feasible solutions. Could these be relaxed? What are the barriers to deriving more general guarantees?
* What is the computational complexity of COBALT? How does it scale with dimensionality and constraints?
* Have you tested on any high-dimensional (D>10) or large-scale real-world problems? This could better showcase scalability.
* Is there a principled way to set the β tradeoff parameter? Sensitivity analysis could elucidate its impact.
* How does performance depend on the quality of the GP model of the constraints?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
