# SNOV: A Scalable Near-global Optimal Verifier for   Neural Networks under Large Perturbations

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Neural networks achieve remarkable performance across domains, yet their deployment in safety-critical settings is limited by robustness concerns. Formal verification offers guarantees but faces a trade-off: complete verifiers scale poorly, while incomplete verifiers either yield loose lower bounds or miss counterexamples due to local optima. We propose a hybrid verifier within a branch-and-bound (BaB) framework that tightens bounds from both sides: an NLP-based upper bound (via complementarity constraints) rapidly rejects unsafe instances, while a relaxation-based lower bound (e.g., $\beta$-CROWN) certifies safe ones. When early stopping is not triggered, the procedure converges to an $\epsilon$-tight interval  ($\underline{\ell},\bar{u}$) localizing the true optimum $f^\star$. To improve efficiency, we introduce warm-started NLP solves with low-rank KKT updates and a pattern-aligned strong branching strategy that accelerates lower-bound tightening. Experiments on MNIST and CIFAR-10 show that our method (i) produces substantially tighter upper bounds than PGD across perturbation radii, (ii) achieves per-node solves with polynomial-time complexity, and (iii) delivers large end-to-end speedups over MIP-based verification, further amplified by warm-starting, GPU batching, and pattern-aligned branching.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript introduces SNOV, a framework that combines $\beta$-CROWN lower bounds with a NLP solver for upper-bounds, all within the Branch-and-Bound (BaB) algorithm, enhanced by slope-guided warm starts and low-rank KKT updates. Some preliminary results related to verifications of neural networks are presented.

### Strengths
Although the BaB framework is well established for neural network verification, the authors aim to enrich it with engineering heuristics, such as slope-guided warm starts and low-rank KKT updates, that can reduce the overall computational cost.

### Weaknesses
At present, the paper does not convincingly showcase modeling capabilities on substantive application cases, and the theoretical developments seem largely incremental relative to known CROWN/BaB hybrids. The experimental section lacks the breadth and controls needed to support the efficiency claims (comprehensive baselines, ablations, and scaling studies). Clarity also suffers from repetition, ambiguous formulations, and language issues. 

Substantial improvements could be made by addressing both the general comments and the specific remarks listed below.

### Major points

- Lack of consistency: there are some objects that are not denoted in the same way throughout the paper: feasible set is sometimes $\mathcal{X}$, sometimes $\mathcal{C}$, sometimes $\mathcal{B}$; the radius of the ball is $\gamma$, $r$, $\epsilon$ or $\epsilon_B$; the linear function of the last layer is $c$ or $\xi$. There’s inconsistent use of bold versus regular text. Some sentences are very informal, e.g., page 3, lines 149–150, or page 7, lines 360–362. Moreover, some objects or notations are used without being properly defined, e.g., per-neuron slopes, unstable fractions, strong-branching score, etc. Overall, the paper is extremely hard to read and follow, and presentation needs to be improved.

- Literature review is quite incomplete. Not all exact verification methods are MIP-based, and for example, SAT/SMT methods are never discussed. Similarly, approximate methods encompass a much larger number of works (abstract interpretation, zonotope domains, SDP relaxations) than the ones mentioned in Section 3.2.

- Experimental setup is not sufficient for deriving rigorous conclusions. Indeed, it is based on verifying **one single instance**, and the detailed network architecture—crucial for understanding the true problem size—is never mentioned. The paper also lacks a systematic analysis of how performance scales with perturbation magnitude.

- Computing exact optimal values $f^{*}$ with “exact or high-fidelity solvers” remains highly non-trivial in high-dimensional settings. No details are provided about this, despite this value being used to measure absolute and relative gaps.

### Minor points

- There should be a clear distinction between a given function, say $f$, and its evaluation at a particular point $x\in\mathbf{R}^n$, i.e., $f(x)$.
- If $\gamma$ is a real number, there is need to write things like $||\gamma||_{\infty}$. This is related to the lack of consistency point.
- The first part of Section 6 includes quite extensive solver-configuration details (e.g., sort_domain_interval=1) that are not essential for understanding the paper and should be moved to the Appendix. This would free space to describe the network architectures and core notions more clearly.

### Questions
1. Which exact $\beta$-CROWN variant is used? A short algorithmic description would help.

2. Why does splitting change the KKT system with rank at most $3$ in your formulation?

3. What is the exact "strong-branching score" used (formula, weights, normalization)?

4. In what sense exactly does the framework go beyond known BaB + LiRPA frameworks? Can you isolate a lemma or proposition that is new (not just a rephrasing) and explain its impact on tightness or complexity?

5. How do your results aggregate over larger sets (e.g., $100$–$1000$ test inputs)?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In the past years, neural network verifiers based on linear relaxations have been shown to be able to scale to verify large neural networks when combined with efficient branching heuristics. However, these methods struggle to provide tight certificates when perturbation radii are large. This work proposes a hybrid verification approach which combines convex-relaxation-based bound propagation methods to obtain lower bounds with more precise nonlinear-programming solvers for obtaining upper bounds. Both methods run in parallel and exchange information, enabling better branching decisions, fast warmstarts in the NLP solver and accelerated convergence to the true optima. The experimental evaluation demonstrates the capability of the proposed method which achieves a precision comparable to that of MIP solvers in significantly less time.

### Strengths
- The verification of neural networks is an important research topic
- Tackling the inability of bound-propagation-based verifiers to scale to large perturbation radii is a valuable contribution
- The experimental results are impressive, showing that the proposed method achieves MIP-like precision in significantly less time
- The authors introduce formulations which also enable the approach to be extended to transformers

### Weaknesses
- My biggest concern about the paper is the empirical evaluation. In neural network verification, we are generally interested in whether a network is robust or non-robust on a particular input for a given perturbation. There is little benefit in being able to obtain very precise bounds on the specification if loose bounds are already sufficient to prove that a property holds. Looking at the results, it seems that a number of experiments are done for cases where even the true lower bound is $<0$, this means that a counterexample exists. A cheap algorithm such as projected gradient descent could just be run to obtain counterexamples, this is done by any neural network verification tool before bound propagation is even started. Comparing the performance of different bound propagation methods in such cases doesn't make a lot of sense. I understand that PGD might not always find a counterexample, but with such large perturbations and lower bounds well below zero it should not be hard to find counterexamples. The only case where verification is actually possible (true lower bound $>0$) is that shown in Table 2. However, the cheap lower bounds obtained by $\alpha$-CROWN here are already sufficient to verify robustness, hence, SNOV does not provide any benefits in this case and is significantly slower than $\alpha$-CROWN and even slower than MIP. A proper evaluation should run SNOV against other state-of-the-art verifiers (such as GCP-CROWN [1]) on established benchmarks (such as those from [2]) and compare the verification time as well as the number of verified instances.
- Although the general concept of the SNOV verifier makes sense to me, there are a number of details on the algorithm (e.g. the rank-3 warm-starts) that are missing in the paper, therefore I am unable to assess whether the overall algorithm is correct. The authors also repeatedly mention that the NLP yields "incumbents and dual-like signals that drive strong branching" but the strong branching heuristic and the information flow from the NLP to $\beta$-CROWN is not explained in the paper.
- It is unclear how $\alpha$-CROWN and the other competing methods are evaluated. Is only a single bound propagation pass performed with these methods? If so, it is not surprising that they would produce looser bounds than the proposed method. SNOV should be evaluated against a complete verifier which performs branching as well (such as $\alpha, \beta$-CROWN and also GCP-CROWN which is enhanced with cutting planes). By running those for the same time budget as SNOV, a proper comparison between the methods would be possible.
- The runtime of the algorithm is significantly higher than that of other methods such as $\alpha$-CROWN and seems to explode when evaluated on slightly larger datasets such as CIFAR10 (see Table 3). I am therefore unsure whether the proposed method would actually scale to networks and datasets of practical size.
- The empirical evaluation lacks important details: Details on the neural network architectures that are evaluated are missing. The authors say that "We implement SNOV for MLP, CNN, ResNet, and Transformer architectures across three benchmark suites.", does this mean that all results in the paper are averages across these architectures? Separate results should be reported for these architectures. Besides this, the methods should be evaluated for multiple inputs and not only for one input as is currently done.
- Section 3.1 is quite imprecise and dense, therefore difficult to understand. E.g. what do the authors mean by "induce loose big-M" or "binaries explode"? This should be extended to provide more context and explain the points that are being made here in more detail.
- The section on "Low-rank KKT updates" (line 234ff) is dense and difficult to understand. A lot of the concepts being referred to here are not introduced.
- The algorithm introduces a number of hyperparameters ($\vartheta, \phi$) but there are no ablation studies and little justification for how these are selected.
- The authors state that some experiments are run on a small Mac machine while others are run on a server with 64 GPUs. It is unclear which experiments are run on which machine which is important to be able to assess the runtimes that are provided. Besides this, the type of GPUs and CPUs should be provided, and it should be clarified whether the approach runs on all 64 GPUs in parallel.
- The paper is full of typos, grammatical errors and incomplete sentences. I tried listing some of them below but eventually stopped taking note of all of them while reading. I would encourage the authors to thoroughly revise the paper with a focus on grammar and language.


### Minor weaknesses and typos
- The notation in the paper is somewhat chaotic and never properly introduced. The authors use $l, u$ as well as $\underline{l}, \overline{u}$ and $\underline{L}, \overline{U}$ to denote lower and upper bounds. E.g. in line 65, both are used in exactly the same context. The notation should be clarified and, if applicable, unified.
- Line 130: Shouldn't the objective function being minimised here be $f(x)$ which is previously introduced as the function representing the specification? $s(x)$ is never introduced
- Figure 3: What are $\xi(0)$ and $\xi(1)$? $\xi$ is introduced as the "specification coefficients" in line 122, but it's unclear what the indexing refers to here.
- Figure 6: The overapproximation area in the left part of the figure extends outside the blue bounds which is incorrect. The figure should be corrected so that the overapproximation is entirely contained in the lower/upper bounds.

- Line 121: is a task-specific specification **is** represented by function --> is a task-specific specification represented by **a** function
- Line 122: One of the common specification is --> One of the common specification**s** is
- Line 123: see Section for particular examples. --> Which section to the authors refer to here?
- Line 303: "ReLU can be written as the solution of the projection quadratic programming" --> this is not a sentence and I don't understand what it means. The authors should fix the grammatical errors here.
- Line 352: We observe that bound propagation based methods efficiently produce the bounds but far loose --> We observe that bound propagation based methods efficiently produce **bounds, but that they are far too loose**
- Line 353: The relative relaxation gap $\overline{\Delta}_{0.1}$ are between --> The relative relaxation gap $ \overline{\Delta} _{0.1} $ **is** between 
- Line 353: "Table 1 and Table 2." is not a valid sentence, this needs to be rewritten.
- Line 355: Section5 --> Section 5
- Table 1 shows that NLP solver reaches up to 7 times faster than MIP --> Table 1 shows that **the** NLP solver reaches up to 7 times faster than MIP. Also what is reached here? The sentence makes no sense in its current form
- Line 361: making the MIP **is** faster than NLP --> making the MIP faster than NLP
- Line 364: algorithm taking advantages of --> algorithm taking **advantage** of
- Line 365: the efficiency of bound propagation method --> the efficiency of **the** bound propagation method
- Table 1 and all other tables: "when large perturbations" should be replaced with "**under** large perturbations". Also "Verifying One Image of MNIST Dataset" should be "Verifying One Image of **the** MNIST Dataset"
- Line 401: with the lower bounds of α−CROWN method --> with the lower bounds of **the** α−CROWN method
- Line 403: to take **a** hundreds of seconds --> to take hundreds of seconds
- Line 404: the proper initialization of NLP solver --> the proper initialization of **the** NLP solver
- Line 405: the efficiency of NLP has about 18 times improvement --> the efficiency of NLP **improves by a factor of 18$**
- Line 407: As shown in Section 5 **that** bound propagation method like α−CROWN is sensitive to large perturbations --> As shown in Section 5, bound propagation method**s** like α−CROWN **are** sensitive to large perturbations,
- Line 408: Grammar, producing a too loose lower bounds --> producing loose lower bounds

### References

[1] Zhang, H., Wang, S., Xu, K., Li, L., Li, B., Jana, S., Hsieh, C.-J. & Kolter, J.Z. (2022) General Cutting Planes for Bound-Propagation-Based Neural Network Verification. doi:10.48550/arXiv.2208.05740.

[2] Brix, C., Bak, S., Johnson, T.T. & Wu, H. (2024) The Fifth International Verification of Neural Networks Competition (VNN-COMP 2024): Summary and Results. doi:10.48550/arXiv.2412.19985.

### Questions
- How are the competing bound propagation methods run? Is branching conducted for these?
- How were the hyperparameters selected?
- Which experiments are run on which of the machines and are all of the GPUs used in parallel?
- The authors claim to solve the problem to $\epsilon$-optimality but then employ a heuristic stopping criterion which, as far as I see, does not guarantee $\epsilon$-optimality. Could the authors clarify what exactly they mean when describing the optimality here?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a novel neural network verifier, named SNOV, that combines state-of-the-art branch-and-bound based on linear relaxations with the use of of a non-linear-programming solver (NLP) to compute upper bounds, as opposed to the customary use of local grandient-based optimizers and primals from the lower bounding algorithms.
This is aimed at what the authors call the "large-perturbation regime".
Experimental results show that SNOV is significantly faster than the employed MILP solver on the considered benchmarks.

### Strengths
The idea to use NLP for BaB upper bounds is novel and potentially very interesting for the neural network verification community. 
It does seem to be significantly faster than the considered MILP solver on the provided benchmarks.

### Weaknesses
What I believe to be the main weaknesses of the paper are related to the presentation and the experimental section.

**Presentation**.

The presentation assumes familiarity with concepts related to interior-point methods. I do not think this is reasonable, and the authors should provide at least a basic introduction in the appendix. KKT conditions are mentioned (and are related to some of the technical improvements) without any technical explanation. Given that neural network verification is a mixed community, with people coming from ML, from formal methods, and from optimization, this appears to be particularly necessary.

Furthermore, important details are omitted: a new branching strategy is introduced, but no details are presented. It is unclear to me what networks are employed for the experiments. It is also not quite clear what is the precise purpose of the complementarity reformulation.

Additionally, I would encourage the authors to tone down the narrative a bit. For instance, it's hard to say that the proposed algorithm has "consistent improvements in scalability and reliability over six state-of-the-art baselines". Improvements are in either scalability or accuracy, and in a very limited experimental setup (see below), and most of these baselines are really far from the state-of-the-art for neural network verification (IBP has never been, for instance, it is mostly used for training purposes).

**Experiments**.

The experiments appear to be carried out over extremely small networks (judging from the MILP runtimes), and over relatively small perturbation radii (it is common to use up to $\epsilon=0.3$ and $\epsilon=8/255$ for MNIST and CIFAR-10, respectively).
Furthermore, no branch-and-bound baseline is provided. 
Speed improvements over MILP solvers are not surprising of their own, and provide no indication on whether the proposed approach would be beneficial to the state-of-the-art (for instance, whether it speeds up alpha-beta-CROWN).
The fact that no code is provided makes it even harder to assess the experimental results.


Given the inconclusive experimental section and the rushed presentation, I do not believe the paper is ready for publication at this stage. But I would be happy to increase my score if these concerns are addressed.

### Questions
- Are you running the NLP sequentially over each branch-and-bound subdomain? If so, is the point associated to the current subdomain  upper bound feasible for it (in other words, does it satisfy all split constraints)? How can this scale? 
- How are intermediate pre-activation bounds set for the MILP? MILP solvers appear to benefit from very tight bounds (see discussion in the GCP-CROWN paper).
- Can you include other scalable BaB methods (the standard alpha-beta-CROWN, at the very least) among the baselines?
- Could you please comment on "A Branch and Bound Framework for Stronger Adversarial Attacks of ReLU Networks", Zhang et al., ICML22? This work focuses on improving the upper bounding part within branch and bound, and it seems extremely relevant for the submission. It would be very important to also benchmark against it.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents SNOV, an exact neural network verifier that combines NLP-based primal heuristics with dual bounds obtained from $\alpha$-$\beta$-CROWN.

Overall, the paper presents several interesting ideas, but I find the presentation to be lacking several important mathematical details, especially regarding the parts that appear to be new in the paper. Furthermore, numerical results are incomplete / hard to follow, and I found some flaws in the experimental setting / solver comparison. My score is largely motivated by the fact that key information is missing from the paper, and the limitations of the numerical experiments. I believe the core ideas have some merit but the paper needs a significant re-write before being ready for publication.

### Strengths
* The paper considers large input-domain perturbations, which is less commonly tackled in existing verification literature
* The paper leverages primal and dual information to accelerate the efficiency of NLP primal search and dual $\alpha$-$\beta$-CROWN bound propagation
* The idea of low-rank KKT updates following branching on a neuron activation is new, however its presentation could have been more exhaustive

### Weaknesses
* I do not consider the "hybrid scheme" (Section 3.2) of SNOV to be a strong novelty. As noted in the introduction, existing verifiers combine a primal (to find adversarial examples if they exist) and a dual component (to obtain certified bounds). For instance, $\alpha-\beta$-CROWN implements primal heuristics like gradient-based attacks.
* Section 3 partitions existing works into "exact" MIP-based vs "approximate" branch-and-bound methods; I do not agree with the paper's classification. 
  * The standard terminology in the NN verification literature is to distinguish between
    * _complete_ verifiers: methods that are guaranteed to either certify robustness or find an adversarial example given sufficient time. Virtually all such methods are based on a mixed-integer representation of the trained neural network, input domain and verification property; methods and tools differ in how they evaluate primal/dual bounds and in their implementations. $\alpha$-$\beta$-CROWN, Marabou, nnenum, CORA, etc... are complete verifiers.
    * _incomplete_ verifiers: methods that may terminate without a definitive answer, i.e., which are heuristic in nature. CROWN, $\alpha$-CROWN or gradient-based attacks are incomplete. Note that complete verifiers often combine a incomplete verifier with a branch-and-bound scheme.
  * the paper's classification between Mixed-Integer Programming and Branch-and-Bound does not capture the fact that i) MIP-based methods rely on branch and bound for completeness and ii) the methods cited in Section 3.2 are based on MIP formulations.

* Building on the above comment, Section 3 would benefit from a re-organization, and deserves additional relevant references such as i) existing verifiers such as CORA, Marabou, nneum, etc.. and ii) existing works that propose primal algorithms, eg.:
  * [_Optimization Over Trained Neural Networks: Taking a Relaxing Walk_](https://arxiv.org/abs/2401.03451)
  * [_Nonlinear Optimization with GPU-Accelerated Neural Network Constraints_](https://arxiv.org/abs/2509.22462)

* Several methodological components are mentioned but not explained, e.g. the paper makes several mentions of "strong branching scores" but these do not appear to be described anywhere in the paper
* Tables 1-7 have inconsistent notations: some use $|\gamma|_{\infty}$, some use $\gamma$. 
* The use of $\gamma$ notation also conflicts with the notation $\epsilon_{B}$ used to define the input domain at the beginning of Section 6. * In Tables 4 & 5, subscript $u$, $u_{ini}$ and $u_{adj}$ are not defined.
* Table 3 and Table 6 are identical
* When solving verification tasks, one can terminate the solve as soon as an adversarial example is found or the instance is proven to be robust. Adding this termination criterion would likely affect the performance results reported in Section 6
* Several claims, e.g., lower number of branch-and-bound nodes, are not supported by any results
* Parts of the text are not proper English sentences, e.g. "Table 1 and Table 2." (l. 353)

### Questions
* Can the authors comment on the choice of representing ReLU activation using complementarity constraints as opposed to simply representing them as nonlinear functions in the NLP formulation?
* How would the complementarity constraint approach handle non-piecewise activation functions, e.g., sigmoid?
* Section 4 makes several mentions of $\beta$-CROWN as the method used to obtain node-level lower bounds. Should this have been $\alpha$-CROWN instead? Several parts of the text refer to algorithmic components of $\alpha$-CROWN.

### Soundness
2

### Presentation
1

### Contribution
2
