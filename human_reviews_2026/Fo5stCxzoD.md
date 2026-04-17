# Beyond Slater’s Condition in Online CMDPs with Stochastic and Adversarial Constraints

- Decision: Reject
- Scores: 4, 6, 2, 4, 6

## Abstract
We study online episodic Constrained Markov Decision Processes (CMDPs) under both stochastic and adversarial constraints. We provide a novel algorithm whose guarantees greatly improve those of the state-of-the-art best-of-both-worlds algorithm introduced by Stradi et al. (2025). In the stochastic regime, i.e., when the constraints are sampled from fixed but unknown distributions, our method achieves $\widetilde{\mathcal{O}}(\sqrt{T})$ regret and constraint violation without relying on Slater's condition, thereby handling settings where no strictly feasible solution exists. Moreover, we provide guarantees on the stronger notion of positive constraint violation, which does not allow to recover from large violation in the early episodes by playing strictly safe policies. In the adversarial regime, i.e., when the constraints may change arbitrarily between episodes, our algorithm ensures sublinear constraint violation without Slater's condition, and
achieves sublinear $\alpha$-regret with respect to the unconstrained optimum, where $\alpha$ is a suitably defined multiplicative approximation factor. We further validate our results through synthetic experiments, showing the practical effectiveness of our algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies online CMDPs where rewards may be adversarial, and constraints are either stochastic or adversarial. It proposes WC-OPS (Weighted Constrained Optimistic Policy Search), a constrained optimization-based method that performs online mirror descent over a moving, optimism-based feasible set. The authors prove, without requiring Slater’s condition, $O(\sqrt{T})$ regret and positive violation in the stochastic constraints case; in the adversarial constraints case they obtain sublinear violation and sublinear $\alpha$-regret against the unconstrained optimum.

### Strengths
This paper establishes best-of-both-worlds guarantees for online CMDPs with stochastic and adversarial constraints; in the stochastic setting, it further provides positive-violation guarantees and shows Slater’s condition is not necessary.

### Weaknesses
- Although the paper improves upon Stradi et al. (2025c) by both tightening the guarantees and eliminating reliance on Slater’s condition, these gains are obtained by computing a safe decision set and solving a constrained optimization each round, incurring higher computational overhead than the primal–dual method in Stradi et al. (2025c). Because the proofs largely defer to the correctness of the safe set estimation, the analysis is comparatively straightforward; in effect, the paper’s main contribution is the weighting strategy for constraint estimation.

- The ``Slater-free'' statement is potentially misleading. In the adversarial regime, the analysis requires a Slater-like margin to obtain $\alpha$-regret, arguably a stricter variant of Slater’s condition than the classical form.  

- The paper repeatedly contrasts its theory with Stradi et al. (2025c), but the experiments do not include that baseline. The submission also lacks a systematic treatment of computational complexity.

- The paper is largely built on the paper on constrained bandits, "Beyond primal-dual methods in bandits with stochastic and adversarial constraints," and extends it into the CMDP setting; most of the techniques are from this paper. It is unclear what the technical challenges and contributions are. 

- The paper focused on model-based design and optimization on a stationary measure space. It might be infeasible in a large state-action space, and it would be better to consider the policy optimization.

### Questions
Please see the weakness above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies online episodic constrained Markov decision processes (CMDPs) under both stochastic and adversarial settings and proposes a new algorithm, Weighted Constrained Optimistic Policy Search (WC-OPS). WC-OPS departs from standard primal–dual methods by performing no-regret updates via online mirror descent over a moving decision space that adapts to online constraint estimates, coupled with a UCB-like optimism mechanism. In the stochastic setting, the algorithm attains $\tilde{O}(\sqrt{T})$ regret and $\tilde{O}(\sqrt{T})$ cumulative constraint violation without requiring Slater’s condition. In the adversarial setting, it achieves sublinear cumulative constraint violation without Slater’s condition and sublinear $\alpha$-regret relative to the unconstrained optimum. Numerical experiments indicate that WC-OPS delivers competitive performance compared to state-of-the-art baselines.

### Strengths
1. This paper introduces a weighted constraint estimation scheme with a moving feasible set instead of the standard primal-dual framework, which is more adaptive and avoids the discussion about Lagrangian relaxations. The authors also introduce a positive violation metric that prevents compensation across episodes, which is more practical and stringent.

2. This paper achieves optimal $O(\sqrt{T})$ worst-case regret in the stochastic setting without Slater’s condition, improving the results of prior work which required it or achieved worse rates $O(T^{−3/4})$.

3. The paper provides high-probability regret and violation bounds with detailed proofs in the appendix.

4. The paper is well-structured with a clear statement of analysis and detailed explanation of algorithm, table and figures.

### Weaknesses
1. This algorithm only serves for the tabular setting, which weakens its practicality and versatility in real-world problems.

2. The algorithm needs to maximize the occupancy measure over a transition confidence set, which could be time-consuming and have computational cost. It would be better to discuss the computational cost.

3. The definition of the problem-specific parameter $\rho$ seems to be too complex. It may be meaningless or difficult to fetch when applying this algorithm to real problems.

4. The paper only conducts experiments in synthetic, layered MDPs. The performance
of the algorithm under the real-world environment remains unknown.

### Questions
1. How sensitive is the algorithm to the choice of parameters $\eta$, $\gamma$ and $\delta$? Can the authors improve the
results by tuning the parameters?

2. What are the pros and cons of WC-OPS compared to primal-dual method? Can the authors provide a more detailed clarification for choosing WC-OPS?

3. The authors define $\alpha$-regret with respect to the unconstrained optimum in adversarial setting. Why do the authors choose this baseline instead of the most common optimum baseline?

### Soundness
3

### Presentation
3

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
The paper studies online episodic CMDPs with bandit feedback, unknown transitions, adversarial rewards, and either stochastic or adversarial constraints. It proposes WC OPS, an optimistic occupancy measure algorithm with a moving feasible set built from transition confidence sets and optimism bonuses on constraints, updated by an online mirror descent step with implicit exploration. Claimed rates: in the stochastic constraints case, $O(\sqrt{T})$ regret, standard violation, and positive violation, all without Slater; in the adversarial constraints case, sublinear violation without Slater and sublinear $\alpha$ regret against the unconstrained optimum with $\alpha=\rho/(1+\rho)$. Synthetic experiments compare against OptCMDP and a primal dual baseline.

The results for the stochastic setting are standard and not novel, even without assuming Slater’s condition. The adversarial setting adds little new insight—its $\alpha$-regret formulation restates known impossibility results and introduces a weaker comparator defined by $\alpha=\rho/(1+\rho)$, which never recovers standard regret even for benign constraint sequences. Overall, I think the paper’s “best-of-both-worlds” claim is overstated: the moving-weights mechanism is reasonable but conventional, while the theoretical and empirical results offer limited genuine advancement.

### Strengths
First, removing Slater in the stochastic constraints case while keeping \sqrt{T} rates for regret, violation, and positive violation is an attractive message for safety-sensitive learning.

The moving feasible set with an optimism bonus on constraints is a clean alternative to primal-dual updates. It aligns with known optimistic design for losses and transitions, and it avoids tuning of dual steps tied to Slater.

### Weaknesses
Computing $u_t(s,a)$ over a confidence set of transitions can be very expensive, since it is an occupancy maximization over a polytope of MDPs. The paper does not discuss how this maximization is performed or approximated in practice per episode. Any complexity statement or an efficient reduction would help.


The experiments are limited to synthetic settings with stochastic constraints. There is no evaluation under adversarial constraints and no test on larger CMDPs or tasks with richer state spaces. Since the claims include both regimes, at least one adversarial constraints experiment would be helpful, and some wall clock results would show the cost of computing confidence sets, optimistic occupancies, and the OMD projection. 


In the adversarial constraints case, the move to cumulative alpha regret is standard, but the definition of alpha depends on a margin rho that is the worst slack over all constraints and all state action pairs in the support of a strictly feasible occupancy. In realistic CMDPs, rho can be very small, which makes alpha near zero and the regret target nearly vacuous. The paper should either quantify rho for natural benchmark families or provide a lower bound on rho under standard reachability or controllability assumptions. As it stands, the claim of sublinear alpha regret can hide a very weak comparator.

The adaptive term $\Gamma_{t,i}$ is presented as the mechanism that allows the same algorithm to handle both stochastic and adversarial settings, automatically adjusting the learning rate so that when violations are small ($\Gamma_{t,i}=0$) it behaves like the stochastic estimator, and when violations are large it adapts to adversarial feedback. However, the theoretical separation between these two modes is not rigorously justified.
1. The $\alpha$-regret analysis in the adversarial regime depends on a problem-specific margin $\rho$ and defines $\alpha = \rho / (1+\rho)$. This margin also governs the adaptive term indirectly, but the algorithm never estimates or uses $\rho$ explicitly. As a result, the paper implies that the same procedure works for both regimes ``without knowing the setting,'' yet it does not show that $\Gamma_{t,i}$ correctly tracks or bounds $\rho$ in mixed or intermediate regimes.
2. What if in the adversarial setting, the rewards are chosen as stochastic but constraints are adversarial, the analysis offers no guidance on which comparator applies. Since the impossibility result arises from adversarial constraints, even mild adversariality should still yield only $\alpha$-regret, not standard regret. The paper does not articulate this distinction, leaving unclear whether such mixed settings revert to standard regret or remain constrained to $\alpha$-regret bounds.
3. If the adversary chooses constraints that always satisfy the feasibility condition with large uniform slack, then theoretically $\rho$ could be large and $\alpha \rightarrow 1$.  The algorithm should then reduce to the unconstrained (or stochastic) case. Yet the results never tighten to full-regret bounds as $\alpha \to 1$; the analysis remains fixed at the weaker $\alpha$-regret level. This gap means the method cannot recognize when the problem effectively becomes unconstrained or easy.

Because of these points, I believe the claim of being ``best-of-both-worlds'' is **overstated.** The algorithm indeed shares one form across regimes, but its guarantees remain conservative (\emph{$\alpha$-regret}) whenever constraints are adversarial, even if they are benign.

### Questions
Could you explain which technical step specifically avoids the dependence on a strict feasibility margin, and how the circular reasoning around $\Gamma_{t,i}=0$ is resolved rigorously?

Can you formally show that $\Gamma_{t,i}$ remains small (or zero) in the stochastic case and stabilizes in the adversarial case without knowing the regime a priori?

Since $\rho$ can be arbitrarily small, doesn’t this make $\alpha$-regret potentially vacuous?
Can you show any setting where $\alpha$ is meaningfully bounded away from zero?

What is the per-episode computational complexity, and how is $u_t$ computed in your experiments

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies online episodic constrained MDPs with adversarial rewards and stochastic or adversarial constraints and proposes an algorithmic framework that achieves desired performance guarantees without Slater's condition being satisfied. In the stochastic constraints regime, the algorithm obtains rate-optimal $\tilde O(\sqrt{T})$ regret and constraint violation bounds without Slater's condition. In the adversarial constraints setting, they take the notion of $\alpha$-regret and prove that the algorithm achieves an $\alpha$-regret bound of $\tilde O(\sqrt{T})$. The algorithm avoids a primal-dual Lagrangian update rule, which often leads to a dependence on Slater's constant. Instead, they take feasible sets that encourage optimistically satisfying constraints. Based on this innovative technique, the paper ends up achieving Slater constant-free bounds.

### Strengths
- The theoretical contribution is that it obtains the first performance bounds for learning online CMDPs that do not require satisfying Slater's condition. 
- The algorithm design is novel. It is natural to take an OMD-style policy update for learning online MDPs. The novelty, however, is that instead of penalizing the constraint via Lagrangian multipliers, they impose an optimistic constraint when taking an approximate feasible set in each episode. This leads to the Slater constant-free regret and constraint violation bounds.

### Weaknesses
The approach of taking optimistic constraints when constructing feasible sets reminds me of LP-based approaches for learning CMDPs. In fact, Efroni et al. (https://arxiv.org/pdf/2003.02189) have already come up with an algorithm for the stochastic setting that achieves (strong) regret and (strong) constraint violation bounds without Slater's condition. It seems that for the stochastic setting, the bounds due to Efroni et al. are tighter in terms of the number of states. It would be more exciting if one could derive an algorithm that achieves a zero constraint violation bound and a regret bound that does not depend on Slater's constant (See Bura et al. https://proceedings.neurips.cc/paper_files/paper/2022/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html).

### Questions
- Can the algorithm be modified to achieve a zero constraint violation bound and a regret bound that does not depend on Slater's constant?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies online episodic CMDP and improve on the best-of-both worlds algorithm achieving better regret and violation guarantees $\tilde{\mathcal{O}}(\sqrt{T})$ for both stochastic and adversarial constraint and only adversarial reward which is a more general case. The paper also introduces the WC-OPS algorithm, which achieves sublinear regret theoretically. Finally, it demonstrates a few experiments.

### Strengths
1.  The paper is theoretically sound.
2. The perfect bounds are certainly helpful
3. The algorithm has merits and definitely the paper paves the direction of new research

### Weaknesses
1. Many typos. Listing a few
      a) Page 2 footnote 2 the simplex dimension is incorrect. Either it should be n dimensional or line 3 under section 2.1 the definition of probability transition function (|X|+1) and anywhere else simplex is used, it has accounted $\Delta_{n}$ to be n dimensional and not n-1 dimensional.
     b)  $X_{k}, X_{k+1}$ not defined
     c) Under the section 3.1 the definition of $M_{t}(x^{'}|x,a)$ uses $t$ as subscript which should be $\tau$. Later in the same paragraph the inidicator functions were mistyped again.
    d) Section 3.3 definition of $\epsilon_{t}(x,a)$, denominator is max{.,.} and not $max 1,N_{t}...$
These typos make it difficult to read

2. Under section 2.1 the conditions for occupancy is dependent on initial state distribution but not mentioned.
3. A little more clarity on the weights estimation would be good. This is good, but not very practical.
4. Abrupt end of the main paper. It seems the paper could not have a proper closure due to page restriction. I suggest moving some details to the appendix.

### Questions
1. The experimental setup seems very unrealistic. Just curious whether the algorithms can be projected to some benchmarks or not.

2. Curious about the effect of bonus terms on $\gamma$ and the results (empirical).

3. Decising the $\alpha$ term seem very difficult or solving an optimization problem might be difficult or time taking. Curious how thw authors decide on the same during their experiments.

### Soundness
4

### Presentation
3

### Contribution
3
