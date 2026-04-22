# Fine-tuning Behavioral Cloning Policies with Preference‑Based Reinforcement Learning

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
Deploying reinforcement learning (RL) in robotics, industry, and health care is blocked by two obstacles: the difficulty of specifying accurate rewards and the risk of unsafe, data-hungry exploration. We address this by proposing a two-stage framework that first learns a safe initial policy from a reward-free dataset of expert demonstrations, then fine-tunes it online using preference-based human feedback. We provide the first principled analysis of this offline-to-online approach and introduce BRIDGE, a unified algorithm that integrates both signals via an uncertainty-weighted objective. We derive regret bounds that shrink with the number of offline demonstrations, explicitly connecting the quantity of offline data to online sample efficiency. We validate BRIDGE in discrete and continuous control MuJoCo environments, showing it achieves lower regret than both standalone behavioral cloning and online preference-based RL. Our work establishes a theoretical foundation for designing more sample-efficient interactive agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a hybrid algorithm, BRIDGE, that combines offline behavioral cloning with online preference-based RL to address data scarcity and unsafe exploration challenges. The method first constructs a confidence set over policies using the Hellinger distance in trajectory distribution space, and then fine-tunes the policy online using preference feedback constrained to this confidence set. Theoretically, the paper establishes a regret bound that converges asymptotically to zero as the number of offline demonstrations increases, formally characterizing the relationship between offline data quantity and online learning efficiency. Empirical results on both discrete and continuous control environments show that BRIDGE outperforms offline BC and online PbRL baselines.

### Strengths
- The theoretical results clearly support the necessity of the hybrid algorithm. The provided regret bound is sharper and offers meaningful insight into how offline demonstration data improves online sample efficiency. While I did not verify the detailed proofs in the appendix, the proof sketch is mathematically sound and grounded in established theoretical tools.
- The writing and organization are clear, with a strong logical flow from motivation to theoretical results and experiments.

### Weaknesses
- Some experimental details are deferred to the appendix but are essential for understanding the results (see specific questions below).
- The computational complexity of the optimization step (line 7 in Algorithm 1) is unclear.

### Questions
- In Figure 2, the number of offline demonstrations used should be specified. Without this, the performance comparison with online PbRL (which starts without offline data) seems incomplete. Also, what policy does the online PbRL baseline start from at iteration 0?
- In Figure 2, why do the dotted lines (offline BC) lack CI? Is the offline dataset fixed across the 20 runs?
- In Figure 3, how is the number of policies quantified in continuous environments? Is the state-action space discretized?
- What motivates the use of the Hellinger distance over other divergence measures? Is it mainly for analytical convenience?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the challenge of applying reinforcement learning (RL) in real-world settings where specifying rewards is difficult and exploration is risky. The authors propose a two-stage framework: first, learn an initial policy using Behavioral Cloning (BC) from reward-free offline expert demonstrations; second, fine-tune this policy online using preference-based feedback. The paper introduces BRIDGE (Bounded Regret with Imitation Data and Guided Exploration), an algorithm designed for this "offline imitation + online preference fine-tuning" setting. The main contributions are:

1.  **The first theoretical framework** for analyzing this offline-to-online preference learning paradigm without ground-truth rewards during fine-tunin.
2.  **Regret bounds** (Theorem 4.1) that explicitly quantify how the amount of offline expert data ($n$) improves online sample efficiency, showing regret approaches zero as $n \rightarrow \infty$. The bounds rely on constructing confidence sets using the Hellinger distance between trajectory distributions, with radii shrinking as $O(1/\sqrt{n})$ (Theorem 4.2).
3.  **Empirical validation** of BRIDGE in discrete (StarMDP, Gridworld) and continuous control (MuJoCo Reacher, Ant) environments, showing lower cumulative regret compared to standalone BC and online preference-based RL (PbRL) baselines.

The core idea is to use offline data to construct a Hellinger ball confidence set around the BC policy, constraining the online preference learning phase to this smaller, safer region of the policy space.

### Strengths
**Novel Theoretical Contribution:** Provides the first rigorous theoretical framework and regret analysis for the widely used but previously theoretically unexplored paradigm of offline imitation learning followed by online preference-based fine-tuning.

**Clear Connection Between Offline Data and Online Efficiency:** The derived regret bounds (Theorem 4.1) explicitly and quantitatively show how the amount of offline data ($n$) reduces online regret, formally justifying the approach's sample efficiency benefits. The use of Hellinger distance confidence sets with radii shrinking as $O(1/\sqrt{n})$ provides a clear mechanism.

**Unified Algorithm (BRIDGE):** Proposes a concrete algorithm that integrates offline BC and online PbRL via uncertainty weighting and constrained exploration within the derived confidence set.

### Weaknesses
**Complexity of Bounds:** While insightful, the final regret bound in Theorem 4.1 and particularly the expanded version in Theorem E.1 are quite complex, making direct interpretation challenging. While the dependence on $n$ is clear, understanding the interplay of all MDP parameters ($S, A, H$), $\gamma_{min}$, $d$, $B$, $W$, etc., is less straightforward. The simplified analysis in Appendix B is helpful but doesn't cover the full algorithm.

**Computability of Hellinger Confidence Set:** Constructing the Hellinger confidence set $\Pi_{1-\delta}^{offline}$ involves estimating the Hellinger distance between the BC policy and candidate policies under the estimated dynamics $\hat{P}$. While Appendix A.4.3 provides an efficient calculation for the discrete case, the practical implementation for continuous environments resorts to an $L_2$ distance proxy in embedding space and constructive sampling around $\pi^{BC}$. This discrepancy between the theory (Hellinger distance over trajectory distributions) and continuous implementation requires further justification or analysis regarding its impact on the guarantees.

**Dependence on $\gamma_{min}$:** The bounds depend on $\gamma_{min}$, the minimum visitation probability of the expert policy. This value is unknown and hard to estimate. While Assumption 3 is milder than typical offline RL coverage assumptions, the practical implications of a potentially very small $\gamma_{min}$ (specialized expert) on the bounds could be discussed further.

**Choice of Embedding Function:** The experiments and Appendix A.1 highlight the critical impact of the trajectory embedding function $\phi$ on performance. The theory assumes $\phi$ is known, but provides little guidance on choosing or learning a good embedding, which remains a key practical challenge.

### Questions
**Questions**

**Continuous Implementation vs. Theory:** Could you elaborate on the theoretical implications of using the $L_2$ distance proxy in embedding space and the constructive sampling method for $\Pi^{offline}$ in the continuous experiments, compared to the Hellinger distance over trajectory distributions used in the theory? Does this approximation affect the validity of the regret bounds in the continuous setting?

**Sensitivity to $\gamma_{min}$:** How sensitive are the theoretical bounds and the practical performance of BRIDGE to the value of $\gamma_{min}$ (Assumption 3)? What happens if $\gamma_{min}$ is extremely small, corresponding to a highly specialized expert?

**Offline Data Quality:** The theory assumes offline data comes from an expert policy $\pi$. The ablation in Appendix A.1 briefly explores noisy data. Could you comment further on how the theoretical guarantees might degrade if the offline data is suboptimal or contains non-expert trajectories? Theorem 4.1 seems robust to some noise via $\gamma_{min}$ but assumes the *demonstrations* come from $\pi^*$.

**Computational Cost:** Calculating the Hellinger distance (even with the efficient method in Appendix A.4.3) and optimizing over the policy set $\Pi_t$ (Line 7, Algorithm 1) seems potentially computationally expensive, especially as the state/action space grows. Could you discuss the scalability of BRIDGE? The appendix mentions runtime but doesn't explicitly compare the overhead vs PbRL.

**Trajectory Embedding $\phi$:** Given the significant impact of the embedding function shown in the ablations, do you have insights into how one might systematically choose or even learn an effective $\phi$ in practical scenarios where the underlying reward/preference structure is unknown?

### Soundness
2

### Presentation
2

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
The paper "fine tuning behavioral cloning policies with preference-based RL" is a novel piece of work exploring the links between imitation learning and online pereference-based RL. The authors present a unified framework (BRIDGE) which combines these two - aiming to address the potential limitations of RL in domains where a) writing down appropriate reward functions is difficult and/or b) letting your agent perform open exploration would be risky or impractical.

Their key idea can be summarized as: firstly, one learns a "safe" warm-start policy from expert demonstrations only (no rewards), secondly you construct a "confidence set" in the space of policies based on this, and lastly the policy is updated using preference-based feedback which takes the form of a binary score between pairs of presented trajectories.

In Theorem 4.1 the authors provide the main theoretical result of the paper that offline data reduces online regret. They present empirical results on discrete environments (StarMDP and Gridworld) and continuous environments (Reacher and Ant) showing that their method achieves lower regret than two baselines.

### Strengths
The work is highly original and a very interesting idea. The paper's most significant contribution is as the first theoretical framework for studying the interplay from offline to online preference learning. The novel idea centres on using the Hellinger distance between distributions of trajectories to build the above mentioned confidence sets. It is the radii of these sets which the authors demonstrate is reduced with more offline data. They derive a non-trivial regret bound in terms of the amount of offline data and equivalent the radii of these sets.

The theoretical developments appear technically sound and well-motivated, although the proofs are extensive and largely relegated to the appendices. Due to time constraints I was not able to verify every step in detail.

The paper's presentation is very clear and well-written, with extensive detail. Each stage is logically introduced with clear figures and empirical data. There are extensive appendices.

### Weaknesses
An important point which is relegated to a footnote on page 6 is the applicability of the theoretical results. Section 3 introduces finite MDP settings before the theoretical results, and the footnote mentions that the results are presented for "tabular, stationary transitions here", but that the framework "readily adapts to other transition model classes". I think this could be made more clear, either by making this adaption more evident or presenting the scope of the theoretical results as distinct from the empirical ones (which include continuous control environments).

The embedding functions are clearly an important consideration and are shown to affect performance. The authors briefly touch on the limitations this could impose - a more detailed analysis of this could be insightful. For example, what are the consequences for more practical domains where the appropriate underlying structure may be unknown?

There are various assumptions made throughout the paper which are mostly introduced as standard to the field, but the implications or limitations of these are not drawn out in detail. Relatedly, a more general discussion of the limitations of the work is missing and would benefit the paper. 

The majority of the theory sections are in the appendix. For accessibility and interest it might be nice to include a high-level proof summary. It could also be interesting to see a discussion comparing BRIDGE to modern RLHF or preference-based fine-tuning algorithms.

### Questions
- Related to the comment about the footnote and the theoretical result applicability, the confidence set construction and theoretical results often rely on the finiteness of S and A (for example, theorem 4.2 explicitly contains |S| and |A| terms. How are these results adapted to the continuous setting?
- Since the performance heavily depends on the choice of embedding function, have you thought about proposing ways to learn such embeddings from data?
- Have you considered the sensitivity of your algorithm to a noisy feedback signal?
- In practise, how should someone choose the confidence set radius or the number of offline trajectories?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors provide the first fundamental analysis of the offline-to-online approach and introduce BRIDGE, a unified algorithm that integrates two signals through an uncertainty-weighted objective function. They derive a regret bound that narrows as the number of offline demonstrations increases, thus explicitly linking the amount of offline data to the efficiency of online samples.

### Strengths
The theoretical part is excellent.

### Weaknesses
The experimental section could be more comprehensive.

### Questions
Have you considered experiments in other cases, or multi-scale scenarios?

### Soundness
4

### Presentation
4

### Contribution
3
