# Frozen Policy Iteration: Computationally Efficient RL under Linear $Q^{\pi}$ Realizability for Deterministic Dynamics

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
We study computationally and statistically efficient reinforcement learning under the linear $Q^{\pi}$ realizability assumption, where any policy's $Q$-function is linear in a given state-action feature representation. Prior methods in this setting are either computationally intractable, or require (local) access to a simulator. In this paper, we propose a computationally efficient online RL algorithm, named *Frozen Policy Iteration*, under the linear $Q^{\pi}$ realizability setting that works for Markov Decision Processes (MDPs) with stochastic initial states, stochastic rewards and deterministic transitions. Our algorithm achieves a regret bound of $\widetilde{O}(\sqrt{d^2H^6T})$, where $d$ is the dimensionality of the feature space, $H$ is the horizon length, and $T$ is the total number of episodes. Our regret bound is optimal for linear (contextual) bandits which is a special case of our setting with $H = 1$. 


Existing policy iteration algorithms under the same setting heavily rely on repeatedly sampling the same state by access to the simulator, which is not implementable in the online setting with stochastic initial states studied in this paper.  In contrast, our new algorithm circumvents this limitation by strategically using only high-confidence part of the trajectory data and freezing the policy for well-explored states, which ensures that all data used by our algorithm remains effectively *on-policy* during the whole course of learning. 
We further demonstrate the versatility of our approach by extending it to the Uniform-PAC setting and to  function classes with bounded eluder dimension.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper makes convincing progress in developing computationally and statistically efficient algorithms to learn an optimal policy in linear $Q^\pi$ realizable MDPs, here with the additional assumptions that the dynamics are deterministic.
I checked the proofs and they look correct and in retrospect simple which is a further positive point in favour of the paper.
Moreover, the "frozen" technique seems to be useful in the experiments.

### Strengths
The analysis is simple and elegant.

Moreover, you need only the realizability of the policies that are played by the algorithm. This is a much milder assumption compared to the Linear $Q^\pi$ realizability assumption imposed in [1,2] which crucially relies on the realizability also of policies that are never selected by the algorithm (in order to prove that the expectation of any low range function is linear).

[1] Weisz et al. Online RL in Linearly -Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore

[2] Zakaria Mhammedi, Sample and Oracle Efficient Reinforcement Learning for MDPs with Linearly-Realizable Value Functions

### Weaknesses
1) If I understand correctly, the freezing technique requires keeping in memory the past linear weights of the Q function. Since the number of states to be frozen will eventually be of order $\epsilon^{-2}$, it seems to me that the suggested algorithm requires memory scaling as $\epsilon^{-2} d$. Is this correct? If yes, please mention it clearly in the paper.
Similarly, also deploying the learned policy does not seem very easy because it requires keeping in memory all the history of the $Q$-values weights, and a snapshot of the dataset at the different times $t$. At deployment time, when a new state action pair is visited, it does not seem very easy to check at which time step it started to be covered. Do you have any ideas how you could learn a policy parametrized by only $d$ parameters in this setting?

2) The regret analysis holds only for stochastic rewards and not adversarially chosen. It might be better to clarify this.

3) The work "Sample and Oracle Efficient Reinforcement Learning for MDPs with Linearly-Realizable Value Functions" by Zakaria Mhammedi is very relevant and should be discussed in the related works.

### Questions
If the memory issue I pointed out is correct, do you see any way to run the algorithm with memory that is independent of $\mathrm{poly}(\epsilon^{-1})$?

The author never says that after $D$ updates of the dataset than any state action pair is covered. I think this remark would help a lot with the proof sketch of Theorem 1. Could you present the proof stating that the statement of Lemma 6 implies Theorem 1 if every state action pair is covered, and then state that this happens necessarily after D updates of the dataset, invoking Lemma 1?

Do you have any idea about how this algorithm could be extended to the adversarial reward setting ?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a computationally and statistically efficient reinforcement learning algorithm, called Frozen Policy Iteration (FPI), under the linear $Q^\pi$ realizability assumption.
Unlike prior works, FPI requires no simulator access and operates with stochastic initial states and rewards, assuming only deterministic transitions.
The key idea of FPI is to freeze the policy for well-explored states—once all actions at a state are sufficiently covered, the algorithm stops updating that state’s policy. This mechanism ensures that all collected data remain effectively on-policy throughout learning, enabling efficient regret minimization without the need for resampling or simulator resets.
FPI achieves a regret bound of $\tilde{O}(\sqrt{d^2 H^7 T})$, which is remarkable since most prior works under this assumption attain only PAC-type guarantees.

### Strengths
Overall, the paper is well-written and easy to follow.
Achieving what appears to be the first regret bound under the linear $Q^\pi$ realizability setting is an interesting result.
However, the paper seems to require clearer articulation and positioning of its key contributions.

### Weaknesses
1. Weisz et al. (2023) study a similar setting, except that they allow for stochastic transitions. Their algorithm attains only a PAC guarantee, not a regret bound. If their analysis were restricted to deterministic transitions, would their algorithm also achieve a regret bound comparable to FPI? A discussion of this comparison, maybe following Theorem 1, should be included in the paper (including explicit contrasts between the PAC guarantees).

2. The computational cost of the proposed algorithm is not discussed. A comparison of the computational efficiency with prior works would strengthen the paper.

3. The proposed algorithm is limited to finite state and action spaces, which diminishes the benefits of using function approximation.

4. No experimental results. Even simple empirical validations or illustrative examples would help demonstrate the practicality of the proposed approach.

### Questions
1. At which point in the analysis is the deterministic transition assumption essential?

2. During exploration—when a state–action pair $(s,a)$ is insufficiently covered by the dataset—do you think it would be possible to achieve better performance by selecting actions based on *optimal design* methods rather than choosing them randomly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In online reinforcement learning with deterministic transitions, the authors propose frozen policy iteration under linear $Q^\pi$ realizability. The method admits only high-confidence part of each trajectory into the dataset and freezes the policy at sufficiently explored states and learn without restarting simulator or resampling. They prove high-probability regret bounds, uniform-PAC guarantees, and extensions based on the eluder dimension, and demonstrate the effectiveness of freezing on small-scale control tasks.

### Strengths
1. They present a computationally efficient algorithm that operates without simulator restart or resampling, and provide theoretical guarantees via high probability regret bounds, Uniform-PAC results, and extension to function classes based on the eluder dimension.
2. They conduct ablation studies on the effect of freezing across two RL environments and report detailed implementation choices to aid reproducibility.

### Weaknesses
1. The theory relies heavily on linear $Q^\pi$ realizability (Assumption 1) and deterministic transitions (Assumption 3).
2. The regret bound and Uniform-PAC bound exhibit significant dependence on the horizon $H$.
3. PAC guarantees may become loose over practically interesting $\varepsilon$ ranges when $\kappa$ is not small.
4. The experiments are limited to verifying Algorithm 1 and ablation studies on freezing, while Algorithm 2—one of the core components of the proposed theory—was neither implemented nor evaluated.

### Questions
1. I am curious how the strong dependency on $H$ arises in the regret or Uniform-PAC bounds, and whether there is scope for mitigation.
2. In the Algorithm 1 experiments, is it possible to compare with additional baseline algorithms beyond the freezing ablation? Could you also provide experimental validation for Algorithm 2?

### Soundness
2

### Presentation
2

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
The submission employs algorithmic peeling for the exploration uncertainty to achieve the first sublinear regret with tractable computational complexity for learning in MDPs with linear $Q^\pi$ realizability under deterministic transition and step-wise disjoint state spaces. Their analysis bypass the need of simulations of "rollout reset" in the online setting and naturally compatible with the Uniform-PAC analysis given the multi-level regression structure is similar to that in He et al. 2021 (Uniform-pac bounds for reinforcement learning with linear function approximation)

### Strengths
- The delicate exploitation of the loopless deterministic "tree" structure via peeling and the design of $k_t(s)$ is novel
- The multi-level uncertainty slicing in Algorithm 2 aligns well with the intuition and is easy to follow

### Weaknesses
- On the disjointness assumption of $\mathcal{S}$: It seems that the regret analysis heavily rely on the disjointness assumption of the state space, i.e., $\mathcal{S}\_{h_1} \cap \mathcal{S}\_{h_2} = \emptyset$ when $h_1 \neq h_2$, effectively enforcing the process to be a "tree", e.g., in **the proof of Lemma 18**, etc. This is an unusually strong assumption because such an assumption is often considered acceptable only in adversarial MDPs.

### Minor weaknesses

- According to the definition of $\bar{L}$, the time complexity of Algorithm 2 appears to be finite only if $\kappa > 0$, which is counter-intuitive. This might be only a matter of presentation but the reviewer encourages the authors to elaborate it in the final version.
- Line 052: To the best of the reviewer's knowledge, MDPs with linear bellman completeness are only known to be tractable under deterministic transitions [1] or constant many actions [2]
- On Lemma 12: The authors did not mention (in the main text) their assumption on reward noise in Preliminaries or Theorem 2.
- Reindexing $\mathcal{D}_{t,h}$ is acceptable but its notation should be detailed before Section 4.1

References

[1] Wu, Runzhe, et al. "Computationally efficient rl under linear bellman completeness for deterministic dynamics." arXiv preprint arXiv:2406.11810 (2024).

[2] Golowich, Noah, and Ankur Moitra. "Linear bellman completeness suffices for efficient online reinforcement learning with few actions." The Thirty Seventh Annual Conference on Learning Theory. PMLR, 2024.

### Questions
- Although it is acceptable for a theory paper to not implement Algorithm 2, the fact that Algorithm 2 requires $\kappa$ as an **input**, e.g., in the computations of $\bar{L}$ and $\Delta_l$'s, is indeed a concern since it is unclear how to estimate $\kappa$ a priori. Do the authors plan to justify or make some educated guess on the mitigation of this fact?
- Line 269: the term "near-optimal" might be misleading, I guess the authors mean low-$\mathcal{D}_{t, h}$-uncertainty for $s_h^{(t)}$?
- On the $\sqrt{d}H$ in Assumption 2: let us say $\kappa$ is very close to $0$, then what would be the upper bound of $Q^\pi$ in Assumption 1, i.e., the upper bound of $Q^\pi$ in Assumption 1 then becomes $\sqrt{d}H$?

### Soundness
3

### Presentation
2

### Contribution
3
