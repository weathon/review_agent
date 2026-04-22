# Doubly Robust Monte Carlo Tree Search

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
We present Doubly Robust Monte Carlo Tree Search (DR-MCTS), an algorithm that integrates doubly robust off-policy estimation into MCTS to improve sample efficiency. Our hybrid estimator combines Monte Carlo rollouts with DR estimation through a variance-minimizing weight computed online. Unlike biased bootstrapping methods that sacrifice asymptotic correctness, DR-MCTS achieves variance reduction while preserving unbiasedness. Unlike entropy-based approaches that exhibit domain-dependent performance, DR-MCTS demonstrates consistent improvements across diverse settings including game playing, mathematical reasoning, and embodied planning. The benefits are particularly pronounced in LLM-augmented environments where each simulation is computationally expensive, making DR-MCTS well-suited for the growing class of language-model-guided planning applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors modify the rollout step in MCTS with a value estimator that mixes the base returns with an advantage estimator, both using importance-sampling to estimate a boltzmann target policy. They derive an optimal mixing coefficient that minimizes the variance of the mixed estimator on top of a fallback option in the low-visit regime. The new method seems to show improved solve rates on a set of planning problems, however it is statistically inconclusive.

### Strengths
- The authors propose a simple modification to the rollout step for MCTS by using an IS-weighted advantage instead of the returns, which reduces variance.

- Quite an extensive related work section, which I found interesting for comparing recent approaches for value estimation/ rollouts in MCTS

- Proper choice of confidence intervals (Wilson intervals) for the win-rates in the experiment section, which I too rarely see other researchers do.

### Weaknesses
The LLM statement on page 10 includes "No LLMs were used for data generation, ...". The authors then contradict themselves in eg., the paragraph in lines 363-372 where GPT-4o was used to generate the policy prior, or lines 347-356 where GPT-4o-mini was used as a world-model.

*Major comments:*
The introduction claims that the new method achieves superior performance and decision quality, however, none of the results show statistical significance. Aside from the result that the new method is considerably faster, which is definitely valuable.
The authors must either adjust their claims in the introduction (especially contribution 3.) or add more statistical repetitions to match the current claims with sufficient power.  

In 1.3.2, I don't follow the claim that doubly robust estimation addresses the high variance of IS. The authors claim that using the baseline function reduces variance, which is correct, this is a result also referred to as the advantage function. However, this reduces the variance of the **returns**, not the IS-ratios. The importance weights will have just as high variance as before, although using an advantage will of course exacerbate this issue less. 
Why not look at more advanced estimators from the RL-literature? For example, TD-lambda, Retrace, V-trace? See also the work by Khandelwal P. (2016) On the Analysis of Complex Backup Strategies in Monte Carlo Tree Search.

In Eq.6 I don't follow why you would use $V_{MCTS}$ and $V_{DR}$ if they both estimate the same thing, and as you already write, $V_{MCTS}$ has guaranteed larger variance over $V_{DR}$ while remaining unbiasedness. This is not explained well, and so I don't trust this to be an actually better estimator.

The authors propose a fallback option in Eq. 8 in the low-visit count regime. 

Section 2.2 provides theory to backup the newly introduced method. However:
- Theorem 2.1: this is a trivial result, the linear combination of two unbiased estimators is unbiased so long that their mixing coefficient adds up to 1. That's more of a corollary or simply a consequence through deduction, which does not warrant a theorem.
- Theorem 2.2: I also doubt that this statement warrants a theorem as I expect the proof to use the decomposition rules of the variance and then reuse already known results. Aside from that, it is a useful statement.

*Minor comments:*
- The abstract is on the long side and overly detailed.
- In the introduction, the first and second paragraph end on the exact same point. 
- Citations are not properly formatted/ put as text-cite. Use \citep so that it reads (authorname, year) instead of authorname (year) unless explicitly referring to said prior work. 
- Double citations on line 79 to Painter et al., 84 to Grosse et al., 87 Borges and Oliveira, 101 Dudik et al., 103 Jiang Li. I stopped checking after this point, there were too many to enumerate. 
- The contribution section in the related work in lines 108-120 repeats the introduction. I also don't see how this is a fundamental change to the MCTS algorithm, you're just modifying the rollout step slightly.
- Line 179, the PUCT formula with learned $Q$ and $\pi$ is not called the polynomial UCT, its the predictor-UCT from Rosin C. D. (2009) Multi-armed Bandits with Episode Context.
- From the background it is not clear how we are going to approximate the value functions and policy prior, are we going to use neural networks? This should be more clear earlier on.

*Comment on Venue:*
After reading the whole document I think the authors do not use any form of neural networks (but non-parametric estimation instead I think) to approximate the policy or value networks, please correct me if I'm wrong. If there is no "learning" involved, then I don't think that ICLR is the ideal venue for this paper. Perhaps AAAI would be more fitting?

### Questions
- Could you think of a didactic small scale experiment where you can numerically validate the lower variance of your estimator compared to the baseline? This could be added as an informative figure to support your claims.

- Could you also provide a result that shows how often the rollout defers to the fallback option for the mixing coefficient $\beta^*$? Right now it is not clear how much the new method relies on your variance minimizing approach, or on a tuned fallback heuristic.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a variant of Monte Carlo Tree Search (MCTS) termed Doubly Robust MCTS (DR-MCTS). In this approach, the algorithm adaptively estimates the value function via a convex combination of the standard MCTS value estimate and a doubly robust value estimate derived from importance sampling. The central idea is that by defining the parameters of this convex combination as a function of the variance and covariance between the two estimates, one can achieve a hybrid MCTS design that reduces overall variance. The work provides theoretical guarantees for the conditions under which this variance reduction is attainable. Empirical benchmarks are demonstrated in both adversarial settings, showing a higher win rate against baseline MCTS, and in single-agent settings, where improved accuracy is achieved on step-wise reasoning tasks under a fixed computational budget.

### Strengths
The paper offers valuable insights and contributions towards constructing improved MCTS algorithms, specifically in achieving variance reduction and enhancing efficiency. It builds effectively upon previous technologies in the MCTS domain.

A notable strength is the authors' attempt to provide important theoretical guarantees regarding the applicability of their DR-MCTS algorithms, clarifying the specific circumstances under which variance reduction can be realized.

### Weaknesses
The depth of the theoretical contribution is somewhat open to question. Lemma 2.1 appears to be relatively straightforward to prove, as it primarily involves taking the expectation over the additive components of the value function. Furthermore, the condition for equality in Lemma 2.2 seems quite strict; it would be helpful to understand if this holds by definition for all DR-MCTS instances or requires stringent enforcement.

Regarding the algorithmic design, the contribution of the hybrid DR-MCTS feels relatively incremeneq.tal. The paper synthesizes a set of established technologies (MCTS, IS-MCTS, etc.), with its key innovation being the introduction of Equation 6, which balances the vanilla and DR-MCTS estimates via a convex combination. While this is a clever and practical idea, the core technical novelty may not be exceptionally high.

Concerning the empirical evaluation:

- For the 9x9 Go experiments, it would be interesting to see additional metrics, such as Elo ratings, for a more standardized comparison.

- The variance bands appear quite wide as expected for a sample size of only 30 games. Additionally, the result showing MENTS and IS-MCTS as weaker than vanilla MCTS is counterintuitive and warrants clarification from the authors.

With respect to the GSM8K results, it would strengthen the paper to include more recent state-of-the-art MCTS reasoning baselines, such as those in [1] and [2]. The appendix should also contain the explicit prompts used to elicit the chain-of-thought reasoning.

### Questions
- Should Thm. 2 be an inequality instead of strict equality? If it is an equality, then it seems very strict that the condition expressed by Eq. 12 would hold. Is a relaxation possible?

- What is the small "o" exactly, is this big O notation? I'm not too sure how Eq. 42 in the appendix became Eq. 43?

- We are now running two versions of MCTS, what are the trade-offs in terms of computational complexity in terms of memory and compute. Also can the computational result of one version of MCTS be employed for another?

- How exactly does a better Go engine imply reduced variance? Could one not have a biased MCTS that performs better at adversarial tasks compared to unbiased?

- Could the authors explain whether or not LLM reasoning technologies such as [1,2] be employed alongside their MCTS design, and/or why it was excluded in the study?

References

[1] Guan, Xinyu, et al. "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking." arXiv preprint arXiv:2501.04519 (2025).

[2] Zelikman, Eric, et al. "Star: Bootstrapping reasoning with reasoning." Advances in Neural Information Processing Systems 35 (2022): 15476-15488.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The number of simulations required for MCTS to converge is known to be highly dependent on the variance of its MC rollout value estimates and the quality of the underlying rollout policy. This paper proposes Doubly Robust MCTS (DR-MCTS), a method aimed at reducing this simulation requirement, especially for computationally expensive environments (e.g., those using LLMs).

The core contribution is the introduction of a doubly robust (DR) off-policy evaluation to estimate the rollout value. Further, the paper proposes an adaptive hybrid estimator ($V_{hybrid}$) that dynamically combines the standard MC rollout estimate ($V_{MCTS}$) with this new DR estimate ($V_{DR}$). 
This combination is optimally weighted based on variance estimations, with the stated goal of minimizing the overall estimator variance. The authors claim that this variance reduction leads to improved sample efficiency, requiring fewer simulations to converge to an optimal policy. The method is evaluated on Go tournament, the GSM8K math reasoning benchmark, and the VirtualHome planning environment.

### Strengths
- **Novel Integration of Doubly Robust Estimation in MCTS:** 

The paper presents a novel approach to improving the simulation efficiency of MCTS, a long-standing challenge. Instead of simply truncating rollouts or replacing them with a learned value function, the authors are the first to propose integrating doubly robust (DR) off-policy evaluation directly into the MCTS value backup. The quality of this contribution is further deepened by the proposal of an adaptive hybrid estimator, introducing a state-action dependent mixing coefficient, $\beta(s, a)$, to learn a data-dependent combination of the standard Monte Carlo rollout estimate and the DR estimate, with the explicit goal of adaptively minimizing the estimator's variance.

- **Demonstrated Effectiveness in Practical Domains:** 

The significance of this work is highlighted by its successful application to diverse and challenging domains beyond traditional board games. The paper clearly demonstrates that this approach is effective in environments like mathematical reasoning (GSM8K) and household planning (VirtualHome), where simulation costs are a primary bottleneck. This demonstrates the potential for this method to be a significant tool for the growing field of LLM-based planning and complex sequential decision-making. The core ideas are presented with sufficient mathematical clarity to spur further research in this direction.

### Weaknesses
**1. Lack of Empirical Comparison for the Core Contribution (**$V_{hybrid}$**)**:

- The paper's main contribution is the "adaptive hybrid estimator" ($V_{hybrid}$), which claims to minimize variance by dynamically combining $V_{MCTS}$ and $V_{DR}$. However, the empirical evidence to support this claim is critically lacking.
- There are no ablation studies that directly compare the performance of using only $V_{MCTS}$, only $V_{DR}$, and the proposed $V_{hybrid}$.
- Furthermore, to prove that $V_{hybrid}$ actually minimizes variance, the paper omits a quantitative analysis comparing the actual empirical variance of each estimator.

**2. Ambiguity and Potential Bias in Estimator Targets**:

- The paper does not clearly distinguish whether $\hat{V}, \hat{Q}$ (Eq. 9, 10) estimates the value of the *rollout policy* ($\pi_b$) or the *target policy* ($\pi_e$). It seems that $R_i$ is obtained from rollouts using $\pi_b$, whereas the paper's goal would be to estimate the value of $\pi_e$ ($V^{\pi_e}$). Improperly combining these (in the hybrid estimator) can lead to bias from policy mismatch, in the form of $\hat V^{\pi_e}(s)=\sum_a \pi_e(a|s)\hat Q^{\pi_b}(s,a)$. The paper does not analyze or address this potential bias.
- In addition, there is no justification of target policy $\pi_e$, defined as the softmax over $Q$. The paper lacks analysis of the target policy choice (gap between true current tree policy and the suggested target policy) and does not explore mitigations.

**3. Limited Experimental Scopes:**
- **Missing Critical Baseline:** The paper's primary motivation is "computationally expensive simulations," such as LLM calls. In this problem domain, the current standard approach is AlphaZero-style MCTS. This involves entirely omitting expensive rollouts and *replacing* them with a learned value function at the leaf nodes.
- **Missing General Benchmarks:** Furthermore, while the paper claims a general improvement for MCTS, its experiments are limited to specific domains (tournament in 9x9 Go, GSM8K, VirtualHome). The lack of evaluation on broader, standard RL benchmarks (e.g., Atari) makes it difficult to assess the general applicability of the methodology.

**Minor comments:**

- To strengthen the paper's motivation and make it a more self-contained presentation for a general audience, the introduction should briefly clarify (i) why a low-variance value estimate is preferable, (ii) why we should focus on the value of target policy rather than that an arbitrary rollout policy for MCTS algorithms, especially in terms of the simulation efficiency.
- The claim that “MCTS is proposed to tackle partially observable environments” in Introduction is misleading; POMDP consideration seems not the point of this work.
- In Equation (2), PUCT seems to indicate *Predictor-UCT* (often written as PUCB/PUCT), rather than *Polynomial UCT*.

### Questions
Q1. Could the authors provide an ablation study comparing the performance and the estimator variance of the proposed $V_{hybrid}$ estimator against two simpler baselines: (i) an estimator that only uses the DR estimate ($V_{DR}$, i.e., $\beta=0$) and (ii) the standard MCTS estimator ($V_{MCTS}$, i.e., $\beta=1$)? 

Q2. What is the justification for defining the target policy $\pi_e$ as the softmax over the current $Q$-values? How sensitive is the method's performance to this specific choice versus other potential definitions of $\pi_e$?

Q3. How do the authors justify the use of $\hat{V}$ , which uses an estimate of $Q^{\pi_b}$ (the value of the rollout policy, not $\pi_e$) in Equation 10? How is the bias introduced by this policy mismatch (estimating $V^{\pi_e}$ using components of $Q^{\pi_b}$) quantified and addressed?

### Soundness
3

### Presentation
2

### Contribution
2
