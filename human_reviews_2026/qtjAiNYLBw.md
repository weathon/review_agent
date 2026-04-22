# Provable Distributional Value Iteration under Partial Observability

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
In many real-world planning tasks, agents must tackle uncertainty about the environment’s state and variability in the outcomes induced by stochastic dynamics and rewards. Motivated by recent progress in world model approaches---where latent models approximate beliefs and support planning---we extend Distributional Reinforcement Learning (DistRL), which models the entire return distribution for fully observable domains, to Partially Observable Markov Decision Processes (POMDPs). Concretely, we introduce new distributional Bellman operators for partial observability and prove their convergence under the supremum $p$-Wasserstein metric. We also propose a finite representation of these return distributions via $\psi$-vectors, generalizing the classical $\alpha$-vectors in POMDP solvers. Building on this, we develop Distributional Point-Based Value Iteration (DPBVI), which integrates $\psi$-vectors into a standard point-based backup procedure—\emph{bridging DistRL and POMDP planning}. Our experiments demonstrate that DPBVI recovers classical Point-Based Value Iteration (PBVI) in the risk-neutral case, validating the distributional extension.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a novel framework that extends distributional RL to the partially observable MDP setting.

### Strengths
The problem setting is well-motivated; the combination of partial observability and distributional RL seems like a worthwhile topic to explore. The empirical results are encouraging.

### Weaknesses
My biggest concern with this paper is that a combination of incoherent notation and lack of explanation by the authors makes it difficult to fully understand the proposed framework. For instance, consider $Z(b)$, first introduced in line 187, as the return distribution with respect to some belief $b$. Later in the paper, $Z(b)$ seemingly takes on a new interpretation as the value function (i.e. the expectation of the return distribution) with respect to $b$, seemingly contradicting the previous definition (more concretely, consider equation 10, where the resulting $Z(b)$ would be a scalar and not the return distribution itself).

These kinds of notational inconsistencies are scattered throughout the paper, making it difficult to understand the framework. Moreover, even from a conceptual level, the paper lacks clarity. For instance, again consider $Z(b)$ which is never formally defined: are the authors implying that the framework learns a distribution $Z$ with respect to the entire belief distribution $b$ (i.e., a distribution with respect to another distribution)? Or is the intent that the framework learns the distribution $Z$ with respect to $b(s)$ for some state $s$ (i.e., a distribution with respect to a single belief probability)? A formal definition would clarify many of these ambiguities.

Similarly, the flow of the text feels disjointed at times. For instance, consider the claim made in lines 198-201. At first glance, this appears to be a claim made with no justification. Upon further reading, it appears that Theorems 1 and 2 are meant to support this claim, yet this is never explained explicitly to the reader, thereby causing confusion.

Another concern is the empirical analysis. While it is encouraging that DPBVI converges to the same solution as PBVI, it is underwhelming to not have any empirical evidence that shows whether the correct distribution is actually being learnt. For example, in the experiments performed, it is possible that the authors’ method may be learning the wrong distribution that happens to have the same mean as the correct distribution.

Finally, the constant references to risk-sensitive decision-making is unnecessary given that the authors are not proposing any novel risk-sensitive results. In particular, many of the references to the risk-sensitive case ultimately amount to unsupported claims and speculation, which is unhelpful and a distraction to the contributions that the authors are actually making. Distributional RL is a compelling enough field that can stand on its own without needing to involve risk-sensitivity to motivate the need for it. Perhaps even more importantly, the amount of space freed up by removing the risk-sensitive discussion can be used to better explain the authors’ contributions (see my comments above).

**Minor Comments:**
- Line 60: typo: gamma-contraction**s**
- Line 80: citation typo: Astrom shows up twice
- Line 82: typo: finite-
- Line 99: This section is missing early works on distributional RL (i.e., pre Bellemare 2017)
- Some key citations are missing in Section 3. For example, the claim that the belief state serves as sufficient information for the agent to behave optimally in lines 117-118.
- Line 196: $B’$ is never defined.
- Figure 3 in Appendix B is great, and should be in the main text.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Summary:

This paper proposes a new value iteration method for partially observable MDPs (POMDPs) under the p-Wasserstein metric, which is to find a policy that maximizes the reward in the worst-case distribution within this Wasserstein ball with p probability. Specifically, the authors consider uncertainty in the reward and transition function, and build on the existence of the convergence guarantees of the value iteration with the Wasstertain metric. The authors show that the value function can be represented as a piecewise linear functions, which also matches the result with standard POMDPs. The authors also demonstrate that their method matches the regular point-based value iteration with no uncertainty or in the risk-neutral case.

My main question is on Theorem 2, where the authors state that the action gap stabilizes after some iteration K, and any greedy action selection would be optimal as the policy will not change after this iteration. Is it possible to determine an order of magnitude of the value K as a function of the number of states, discount factor, and the Wasserstein metric? I think this is the critical proof of the paper, and the authors can further explain and develop this proof in the appendix.

What is the computational complexity of a single DPBVI iteration?  Specifically, what is the order of |M| in this case? Is it a function of the number of beliefs or POMDP states?

The authors state that they do not formally address risk-sensitive objectives and full return distributions, but I think they can still provide some comments or clarifications on what type of risk-sensitive measures both in theory and practice, as this is not mentioned either in the paper or in the numerical results.

Overall, I think this paper is a good theoretical contribution, but it needs clarifications on the proofs and practical examples or reformulations for using POMDPs with practical risk-sensitive measures.

### Strengths
The paper makes a solid theoretical contribution to extending standard value functions to risk-sensitive measures using POMDPs and Wasserstein metrics. The proofs and the resulting formulation match the classic POMDP value functions, which is a strength of the paper.

### Weaknesses
The paper can be improved by extending the proof of Theorem 2, and also giving comments or numerical examples of solving POMDPs with risk-sensitive measures that can be expressed by the Wasserstein metric.

### Questions
My main question is on Theorem 2, where the authors state that the action gap stabilizes after some iteration K, and any greedy action selection would be optimal as the policy will not change after this iteration. Is it possible to determine an order of magnitude of the value K as a function of the number of states, discount factor, and the Wasserstein metric? I think this is the critical proof of the paper, and the authors can further explain and develop this proof in the appendix.

What is the computational complexity of a single DPBVI iteration?  Specifically, what is the order of |M| in this case? Is it a function of the number of beliefs or POMDP states?

The authors state that they do not formally address risk-sensitive objectives and full return distributions, but I think they can still provide some comments or clarifications on what type of risk-sensitive measures both in theory and practice, as this is not mentioned either in the paper or in the numerical results.

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
3

### Summary
This paper presents a formal extension of distributional dynamic programming to Partially Observable Markov Decision Processes (POMDPs). The authors introduce distributional Bellman operators for partial observability and prove they are contractions under the supremum p-Wasserstein metric. They also propose $\psi$-vectors as a finite, distributional analog to classical $\alpha$-vectors , which represent the return distribution for each state and maintain the piecewise linear and convex (PWLC) property in the Wasserstein space. Building on this, the paper develops Distributional Point-Based Value Iteration (DPBVI), an algorithm that integrates the vectors into the point-based backup procedure. Experimental results show that DPBVI converges to the same risk-neutral solution as classical PBVI, albeit with an order of magnitude increase in computation.

### Strengths
- The paper explores an interesting direction: applying distributional dynamic programming to POMDPs. This is a novel perspective that could open up new research avenues.

- The theoretical results, while straightforward, are nice to have for completeness. They confirm that distributional Bellman operators for POMDPs align with those for belief MDPs.

### Weaknesses
- The premise is confusing. Distributional RL is fundamentally about learning, but this paper focuses on planning in a known POMDP. This feels more like distributional dynamic programming than RL.

- Theorem 1, 2, and Corollary 1 are not surprising. Since any POMDP can be rewritten as a belief MDP, the extension of distributional RL theory is almost trivial.

- Section 4.2 lacks rigor and clarity. The definition of $\psi$ vectors and $\Psi$ is inconsistent and confusing:

  1. $\psi^s$ is described as a distribution over returns, yet the paper refers to $\psi$-vectors and uses inner products as if these were well-defined vectors.

  2. Equation 9 defines $\Psi$ as a set, not a vector, which contradicts the terminology. Figure 3 suggests discretization via binning. This conflates the general case (the theoretical ground truth) where the return distributions are continuous, with the categorical approximations (like C51).

  3. Convergence guarantees for categorical approximations differ from those for continuous distributions, so Equation 10 and Theorem 3 are questionable. For example, for the categorical approximation, convergence is proven only for Cramer, not p-Wasserstein.

  4. The assumption in Section 5 (“mean-preserving when its support spans all possible returns”) is critical, yet the theory in 4.2 seems to ignore this until the end.

- DPBVI appears to be a straightforward extension of PBVI. Given that Section 4.2 is shaky and other results are trivial, the novelty is limited.

- Experimental results are insufficient and unconvincing: They only show that DPBVI converges to the same result as PBVI in risk-neutral settings, but much slower. No experiments demonstrate any advantage of DPBVI over PBVI, especially in risk-sensitive scenarios. The added computational cost of DPBVI is significant, yet no justification is provided through empirical benefits.

### Questions
1. What exactly is the mathematical structure of $\psi^s$ and $\Psi$? Are these vectors, sets, or distributions? The paper needs to make this precise.

2. How can Equation 10 and Theorem 3 hold if $\Psi$ is based on categorical approximations? Isn’t this only approximate, not exact?
What is $\Gamma$ formally? How is the inner product $\langle \Psi, b \rangle$ rigorously defined?

3. Why does the paper use categorical distributions for theoretical development? How does this affect convergence guarantees?

4. Why are there no experiments showing the benefits of DPBVI in risk-sensitive settings? Without this, what is the practical motivation for the approach?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors tackle both state and outcome uncertainty in Partially Observable Markov Decision Processes (POMDPs) by provably extending (for the first time to the best of my knowledge) the Distributional Reinforcement Learning (DistRL) framework and providing convergence guarantees, an important step towards risk-sensitive planning for real-world applications.

This is accomplished by introducing distributional Bellman operators to the POMDP setting and proving convergence to the optimal return distribution (if there is a unique optimal policy) under partial observability (i.e., distributional Bellman operators remain $\gamma$\-contractions under the suprenum p-Wasserstein metric).

The second key contribution is the introduction of $\psi$\-vectors which replace each scalar entry of the standard POMDP $\alpha$\-vector with a distribution over returns. Finite $\psi$\-vectors are then proven to be sufficient to represent the optimal distributional value function while preserving the PWLC property in the Wasserstein space.

The final major contribution is Distributional Point-Based Value Iteration (DPBVI) which adapts PBVI to the distributional setting using $\psi$\-vectors. Under risk-neutral conditions (i.e., reward expectation only), $\psi$\-vectors collapse to $\alpha$\-vectors and key convergence guarantees (Corollary 1) are preserved whilst operating in the space of return distributions. Experimentally, DPBVI is shown to converge ($\epsilon = 1e-3$) to the same value function as PBVI using the MiniGrid DoorKey and Two-state Noisy-Sensor POMDP environments.

### Strengths
This paper is well written and provides convincing evidence for each of the core contributions within the defined scope of establishing the theoretical foundations for distributional planning under partial observability. The core significant contributions are detailed in my summary.

I do not know the literature on DistRL well enough to be  certain that this is the first extension of distributional planning to the POMDP setting, but that being the case this appears to be significant and impactful work with good potential for future extensions and real-world applications.

I found the formalisation and proofs to be written in a way that provides a good intuitive understanding alongside the guarantees.

### Weaknesses
Whilst the purpose of this paper is to establish theoretical foundations, and appears to do so thoroughly, the experimental results are limited to validating the convergence of DPBVI w.r.t PBVI under risk-neutral / reward expectation only objectives. Given the wider goals of this work to enable risk-aware real-world planning under partial observability, the contribution would be improved by investigating experimentally:

- How DPBVI provides new information, do the $\psi$\-vectors show meaningful return distributions experimentally?
    
- “We hypothesize these spikes arise from categorical projection error“ - could this be investigated further e.g., by varying the number of bins M?
    
- DPBVI has a significantly higher runtime “due to the additional cost of distributional operations“. Could this be expanded upon to provide any insight or assurances regarding scalability beyond these small POMDP environments?

### Questions
- Do the $\psi$\-vectors show meaningful return distributions experimentally?
    
- “We hypothesize these spikes arise from categorical projection error“ - could this be investigated further e.g., by varying the number of bins M?
    
- DPBVI has a significantly higher runtime “due to the additional cost of distributional operations“. Could this be expanded upon to provide any insight or assurances regarding scalability beyond these small POMDP environments?

### Soundness
3

### Presentation
4

### Contribution
4
