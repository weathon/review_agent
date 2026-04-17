# Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Scientific discovery is increasingly constrained by costly experiments and limited budgets, making efficient optimization essential for AI for science. Bayesian Optimization (BO), while widely adopted for balancing exploration and exploitation, suffers from slow cold-start performance and poor scalability in high-dimensional settings, limiting its effectiveness in real-world scientific applications. To address these challenges, we propose LLM-Guided Bayesian Optimization (LGBO), the first LLM preference-guided BO framework that continuously integrates the semantic reasoning of large language models (LLMs) into the optimization loop. Unlike prior works that use LLMs only for warm-start initialization or candidate generation, LGBO introduces a region-lifted preference mechanism that embeds LLM-driven preferences into every iteration, shifting the surrogate mean in a stable and controllable way. Theoretically, we prove that LGBO is not perform significantly worse than standard BO in the worst case, while achieving significantly faster convergence when preferences align with the objective. Empirically, LGBO achieves consistent improvements across diverse dry benchmarks in physics, chemistry, biology, and materials science.  Most notably, in a new wet-lab optimization of Fe–Cr battery electrolytes, LGBO reaches \textbf{90\% of the best observed value within 6 iterations}, whereas standard BO and existing LLM-augmented baselines require more than 10 iterations. Together, the results suggest that LGBO offers a promising direction for integrating LLMs into scientific optimization workflows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces LLM-Guided Bayesian Optimization (LGBO), a novel framework for integrating LLMs into the BO loop. The core contribution is a "region-lifted preference" mechanism, where the LLM provides continuous guidance by suggesting promising regions in the search space. This guidance is incorporated as a stable mean shift in the GP surrogate model, leaving the covariance structure intact. The authors provide theoretical regret bounds demonstrating that the method is safe in the worst case (i.e., with misaligned guidance) and can significantly accelerate convergence when the LLM's guidance is accurate. The framework is evaluated on several "dry" scientific optimization benchmarks and a "wet-lab" experiment, showing improved performance over standard GPBO and a prior LLM-augmented method, LLAMBO.

### Strengths
- The core idea of using an exponential lift on a discretized region, and showing its equivalence to a GP mean shift, is novel and technically sound
- Theorem 1 is useful, as it formally establishes that the framework is robust to poor/misleading LLM guidance (worst-case) and can provably accelerate convergence when guidance is informative
- The wet experiment on Fe-Cr battery electrolyte optimization is a great example to demonstrate the method's applicability to real scientific discovery problems beyond simulated benchmarks

### Weaknesses
- The paper motivates the work by citing the poor scalability of BO in high-dimensional settings. However, the experiments are conducted in search spaces of relatively low dimensionality (i.e., all experiments are conducted in low-dimensional settings with $d \leq 7$). Hence, the effectiveness of the proposed method in a truly high-dimensional problem is not sufficiently demonstrated
- The choice of baselines is far from adequate. The authors cite LLINBO and ReasoningBO as more systematic "LLM-in-the-loop" frameworks, yet fail to include them in the comparison. Comparisons with other related LLM-driven BO approaches, such as BOPRO [1] and CAKE [2], are also missing
- To the best of my understanding, the LLM's guidance is limited to a single point or a hyper-rectangular region. I believe this format may be too simplistic for real-world problems where promising regions could be complex (e.g., non-convex, disjoint, or like a manifold). Hence, it is not clear if this method can be applied to problems with more complex geometries

[1] D. Agarwal et al., "Searching for optimal solutions with LLMs via Bayesian optimization," ICLR, 2025.

[2] R. C. Suwandi et al., "Adaptive kernel design for Bayesian optimization is a piece of CAKE with LLMs," arXiv preprint arXiv:2509.17998, 2025.

### Questions
- How does LGBO handle optimization landscapes where the promising regions are have complex geometries that cannot be well-approximated by this format? Does this structural constraint limit the framework's applicability?
- How do we translate the LLM's confidence score into the guidance strength $\lambda$ and the region's properties (e.g., radius in point mode)? Since the parameter $\lambda$ is critical to the regret bounds and practical performance, the authors should provide more discussion on its selection, tuning, or sensitivity
- Based on my experience, a strong mean shift in an incorrect region without a corresponding increase in uncertainty might overly encourage the acquisition function to exploit that area. Since the proposed mechanism only shifts the surrogate's mean while leaving the covariance unchanged, could this lead to premature convergence if the LLM is confidently wrong?

### Soundness
2

### Presentation
3

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
This paper introduces LLM-Guided Bayesian Optimization (LGBO), a new framework that continuously integrates LLM preferences into the BO process. Unlike prior work that uses LLMs only for warm-start initialization or candidate proposal, LGBO employs a region-lifted preference mechanism that adjusts the GP surrogate mean based on LLM-specified regions of interest. The authors show that this modification maintains theoretical rigor, preserving covariance structure and provide regret bounds demonstrating that LGBO performs no worse than standard BO in the worst case and converges faster when LLM preferences align with the true objective.

Empirically, LGBO is evaluated on four dry scientific benchmarks (LNP3, Cross-barrel, Concrete, HPLC) and a new wet-lab experiment on Fe–Cr battery electrolytes. Across all tasks, LGBO outperforms both standard GP-based BO and LLMABO, showing faster convergence, higher final performance, and lower variance. Ablation studies further confirm that performance gains arise from the continous preference integration rather than initialization or random region lifting.

### Strengths
- Novel mechanism for incorporating LLM guidance into BO: the region-lifted preference is mathematically principled and computationally tractable. Making UCB-type bound natural to derive. 
- This paper is well-motivated, with a wide range of real-world scientific discovery problems tested in the experiments. 
- The methodology in this paper provides new insights about how **continuously** incorporating LLM preference could stabilize BO convergence compared to LLMABO, which integrates LLM preference indirectly, with robust ablation study.
- The framework is general and modular, compatible with standard GP surrogates and acquisition functions.

### Weaknesses
**The writing can be improved a lot** 
- The definitions and assumptions are not stated in the main paper. The definitions of $||\cdot||$ in Line 293 and $R_T$ in Line 302 are missing, and the assumption for GP regression in the RKHS setting is not clearly stated in the main paper.
    
- The statement _“running GP-UCB on residual labels”_ in Line 295 is vague. Moreover, how it is equivalent to executing the proposed LGBO algorithm may be unclear to readers at first glance.
    
- The definition of _alignment_ being equivalent to $c > 0$ in Theorem 1 is not clearly stated and explained. Its definition appears only in the proof of Theorem 1 in Appendix A.
    
- In addition, there is a major typo in Theorem 1. According to the proof,  
    $c = \frac{\langle f - \tau, g \rangle}{||f - \tau||\cdot||g||}$,  
    while in the main text it is written as  
    $c = \frac{\langle f - \mu, g \rangle}{||f - \mu||\cdot||g||}$.  
    This is misleading, since $\mu$ is typically the GP mean, whereas $\tau$ is related to the lifting function.
    
**Theoretical result is not consistent with the proposed method.** Theorem 2 only describes the algorithm’s behavior when the mean adjustment is a fixed  
    $g(\cdot) = \sum_{i=1} a_i  k(x_i, \cdot)$ through out all iterations 
    whereas the proposed algorithm proposes a different $g$ through the LLM at each BO iteration. Thus, the theoretical results only partially explain the performance of LGBO.
    
**Lack of baseline comparison.** Only two baselines are included, and among them, only LLAMBO is an LLM-based BO method. Other LLM-integrated BO methods mentioned in the related work, such as ColaBO and LLINBO, are not included. Therefore, the statement about the instability and potential divergence of ColaBO in Line 128 is not well-supported by evidence.

### Questions
1. Why $a_g$ are chosen to be greater than or equal 0? ( why don't we define a general LLM-based adjustment, not just lifting potential points, but also penalizing undesirable queries by allowing negative $a_g$s?)
2. The novelty of the region-lifted preference is not emphasized until conclusion (by searching the key word "novel"). Could the authors re-confirm this is a novel idea?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces LLM-Guided Bayesian Optimization (LGBO), a framework that integrates priors from large language models (LLMs) into Bayesian Optimization (BO) for scientific experimentation. Unlike previous approaches that use LLMs only for the initialization of BO exploration or candidate generation, LGBO continuously incorporates LLM-derived region-lifted preferences into every iteration of the optimization loop. The authors show that this mechanism effectively shifts the Gaussian process (GP) surrogate’s mean without affecting its covariance, allowing semantic LLM guidance while retaining BO’s statistical guarantees. Theoretical results prove that LGBO is not worse than standard BO in the worst case and can achieve faster convergence when LLM guidance aligns with the true objective. Experiments on four “dry” scientific benchmarks (LNP3, Cross-barrel, Concrete, HPLC) and one “wet-lab” Fe–Cr battery optimization demonstrate consistent acceleration and improved stability over GPBO and LLAMBO baselines.

### Strengths
- Novel integration: Proposes a principled and theoretically grounded method to embed LLM preferences directly into the BO surrogate, moving beyond heuristic or warm-start use of LLMs.

- Theoretical formulation: Provides formal regret bounds showing bounded degradation under misalignment.

- Evaluation: Including both simulation and real-world experiments, showing convincing empirical improvements across diverse domains.

- Reproducibility: Clearly describes prompt templates, datasets, and experimental protocols; theoretical and implementation details are provided.

- Scientific relevance: Demonstrates the potential for LLMs to accelerate experimental optimization in time-limited domains.

- Stable framework design: The region-lifted preference mechanism elegantly maintains GP structure and stability, avoiding instability issues typical of preference-based methods.

### Weaknesses
- Evaluation baselines: Evaluation omits other preference-based or human-in-the-loop BO methods (e.g., ColaBO, Preferential BO with human experts), which could contextualize LGBO’s relative contribution.

- Dependence on prompt engineering: Performance and robustness may depend strongly on prompt quality and LLM capabilities, but sensitivity analyses on prompt design are limited.

- Scalability questions: Experiments focus on low- to medium-dimensional tasks (≤6 variables); it remains unclear how LGBO scales to high-dimensional or multi-objective settings.

- Computational cost: Continuous LLM querying at each iteration may incur significant computational or latency overhead; this is not quantified or mentioned in the paper. 

- Interpretability of LLM Guidance: While the framework embeds preferences mathematically, the semantic validity or interpretability of the generated regions is not thoroughly analyzed.

- Limited real-world task diversity: Only one wet-lab experiment is included; broader real-world validations would strengthen generality claims. It is also not clear why these particular tasks were selected for the evaluation.

### Questions
- What is really the motivation for choosing preference optimisation for this setting? It is not immediately obvious to me. Better motivation as context for the approach would be helpful.

- The region-lifted preference is formalized as an exponential mean shift. Could similar effects be achieved using other functional forms (e.g., linear or kernel-weighted lifts), and what motivated the specific exponential choice?

- Since LGBO queries the LLM at every iteration, what is the computational or latency overhead compared to traditional BO? Could lightweight surrogates or cached reasoning traces mitigate this cost?

- How sensitive are the results to the exact prompt design or LLM choice? While Appendix B provides structured prompts, have you quantified performance variance across alternative phrasing or reasoning styles?

- Have you analyzed the semantic quality of the LLM-suggested regions—do they align with known scientific heuristics or physical laws?

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
This paper proposes a way to integrate prior information in large language models into Bayesian optimization.
At its core, the method is based on a simple idea.
It prompts the language model to generate a promising region in the search space.
Then, the proposed method updates the GP mean function so that the mean function has higher values in the promising region.
Experiments on benchmark test functions across several areas show promising results against conventional BO.

### Strengths
1. This paper works in the intersection of Bayesian optimization and large language models, and shows how to effective utilize the prior information stored in LLMs.
Those prior information is often hard to encode into conventional Bayesian optimization by kernel designs.

1. Using the lifting functional to encode the prior information from the language model makes sense to me.
Though I have not checked the math, the part that the functional in the exponential form turns out to shift the mean function makes sense to me, as it is similar to exponential tilting for Gaussian distributions.

### Weaknesses
1. The interpretation of Theorem 1 seems to be over claimed.
The correct interpretation should be weaker than what's claimed in the paper.
The theorem assumes the lift is given in advanced, and **does not change** during BO.
However, the method proposed in this paper actually utilizes language models interactively in that the lifted region and or points get updated in each iteration of BO.
Thus, if the language model's prediction is bad, the regret bound could be unbounded.

1. Many important technical details are missing.
    - How to set the guidance strength parameters \\(a_g\\)?
    Do you estimate them during GP model fitting with maximum likelihood, or do you set them to fixed numbers?
    - How the discretization \\(x_g\\) is chosen in a lifted region?
    - How does the confidence scores generated by LLMs affect the lifting functional?

### Questions
1. What's the exact definition of misaligned lift in Theorem 1? Isn't the misaligned case the same as \\(c = 0\\), i.e., small cosine similarity?

1. Line 293: Is the norm in theorem 1 the RKHS norm?
If so, it would be better to make it explicit.

1. Line 162: "Intuitively, \\(p(\cdot)\\) here denotes a probability distribution...".
Shouldn't it be \\(\rho\\)?

### Soundness
3

### Presentation
3

### Contribution
3
