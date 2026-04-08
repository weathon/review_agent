## Human Reviewer 1

### Summary
R2PS is a framework for computing worst-case robust, real-time pursuit strategies in graph-based pursuit–evasion games under partial observability. It extends dynamic programming to handle asynchronous movements and limited observations, providing theoretical guarantees of pursuit optimality. By incorporating a belief preservation mechanism to maintain uncertainty over the evader’s position and embedding this process into an Equilibrium Policy Generalization reinforcement learning framework with graph neural network policies, R2PS achieves efficient and transferable learning. Experiments across diverse graph environments show zero-shot generalization and consistent outperformance over existing game-theoretic RL baselines.

### Strengths
1. The paper extends the classical Markov PEG dynamic programming framework to asynchronous movement and partial observability settings, providing clear proofs of optimality.

2. It leverages belief-based DP guidance to inform an adversarial reinforcement learning process and introduces a cross-graph training mechanism that enables zero-shot structural generalization.

3. The empirical study spans both synthetic and real-world graph environments (e.g., Downtown, Eiffel Tower), incorporating analyses of observation range, runtime efficiency, and time complexity.

4. The proposed method achieves notable improvements in computational efficiency and real-time responsiveness compared to traditional DP-based replanning approaches.

### Weaknesses
1. The related work section is underdeveloped.

2.  The test set is not particularly large or diverse, and key sensitivity analyses (e.g., observation range effects on trained RL policies) are missing.

3. The paper’s writing quality is uneven. Some abbreviations are undefined, and the structure is occasionally confusing (e.g., placing time complexity analysis within the evaluation section rather than methodology).

4. Although the mathematical derivations for belief updates are technically sound, they would benefit from clearer illustrations or explanatory examples to improve accessibility for readers unfamiliar with partially observable multi-agent systems.

### Questions
1. How does R2PS scale with an increasing number of pursuers or higher graph complexity (e.g., connectivity, degree, presence of cycles)? Can the theoretical guarantees extend to multi-pursuer settings as $m$ increases?

2. Is there a quantifiable bound on the sub-optimality of belief-averaged or RL-derived policies relative to the ideal DP baseline, particularly under reduced observation windows?

3. Could the authors provide detailed wall-clock runtime and scaling plots comparing GNN inference and DP computation across a wider range of graph sizes and hardware setups?

4. Since the reward is binary (0/1, line 108), should the reward definition be refined or generalized for more nuanced pursuit outcomes?

5. How does Theorem 1 differ from the results in Lu et al., and is the proof in Appendix A.1 itself a novel contribution? Additionally, Equation (3) requires clearer explanation or motivation.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
4

### Confidence
2

---

## Human Reviewer 2

### Summary
In the paper the authors presents a method for learning worst-case robust real-time pursuit strategies (R2PS) in graph-based pursuit-evasion games under partial observability. The authors extend a dynamic programming (DP) approach to handle asynchronous moves and partial observability via belief preservation and integrate it into a reinforcement learning framework for cross-graph generalization. Experimental results demonstrate that the proposed method outperforms existing baselines in both synthetic and real-world graphs.

### Strengths
1. The paper provides a solid theoretical foundation, extending DP-based policies to asynchronous and partially observable settings with proofs of optimality under certain conditions. It looks that the analysis is thorough and well-supported.
2. The proposed R2PS framework is highly applicable to real-world security and robotics scenarios where graph structures may change dynamically and full observability is unrealistic.
3. The combination of belief preservation with adversarial RL and cross-graph training is novel and effectively addresses the challenge of zero-shot generalization to unseen graphs.
4. The method outperforms baselines like PSRO.

### Weaknesses
It has an assumption that the evader’s policy is known or can be approximated. In real world settings, the evader’s strategy may be non-stationary or adversarial in a more complex sense.

### Questions
Have you considered scenarios where the evader also operates under partial observability? This could better model symmetric real-world pursuit-evasion problems.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2

---

## Human Reviewer 3

### Summary
The paper investigates robust cross-graph strategies for pursuit-evasion games (PEGs) in the partial-observability setting. It is assumed that the evader can move asynchronously with perfect knowledge, while the pursuers are uncertain about the evaders location. The paper introduces two methods, one provably optimal dynamic programming formulation and an RL-based formulation based on the Equilibrium Policy Generalization (EPG) (Lu et al. 2025). It is shown that these methods are more robust and successful compared to baseline heuristics and general RL methods.

### Strengths
The paper is generally well-written. It appears to be sound (at least as far as I can judge as a non-expert). The one-sided partial observability setting is natural for pursuit-evasion games and treated both theoretically and practically in the paper. The extension of the EPG framework to the partial-observability setting is novel. The simulations show convincing improvements over various baselines.

### Weaknesses
The paper is an extension of Lu et al. (2025a) to the partially observed setting. Both the DP and the EPG approach seem to be slight extensions of this earlier work. The paper does not seem to develop any fundamentally novel techniques. 

The simulations show improvements in comparison with simple baselines and the competing PSRO approach. In particular, Lu et al. (2025a) already reported clear improvements of EPG against the PSRO policies in the cross-graph setting for PEGs and thus the performance improvements appear to mainly stem from the advantages of this framework in this setting.

### Questions
1. Can you emphasize the novel aspects of this work compared to Lu et al. (2025a)?

Typos:
- line 141 and 145: do you mean "worst-case" evader?
- line 262, 264, 269, 333: Pos and belief are set in math mode

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper introduces R2PS, a novel algorithm designed to generate worst-case robust strategies for real-time pursuit under partial observability. The authors formulate the pursuit-evasion interaction as a two-player zero-sum game where the pursuer must act with limited information about the evader’s state and policy. By leveraging a lookahead-based game tree, R2PS computes strategies that minimize the maximum possible regret regardless of the evader’s behavior. The method incorporates a pessimistic planning approach over belief states and ensures the pursuer’s strategy remains effective even against adaptive or adversarial evaders. Through empirical evaluation in grid-world environments, the results demonstrate that R2PS significantly outperforms baseline approaches such as QMDP and other belief-based planners in terms of robustness and capture rate, especially under strong observation uncertainty.

### Strengths
- Theory–practice bridge: Proves a clean minimax recursion for the DP table and upgrades it to asynchronous evader moves; then embed this belief-preservation mechanism into RL.

- Real-time, cross-graph generalization: The R2PS training scheme yields zero-shot gains on unseen real-world graphs and large runtime advantages over recomputing DP online.

### Weaknesses
- Modelling assumption (no exits): While this choice enables Algorithm 1’s near‑optimal time‑complexity guarantees for Markov PEGs, it departs from many real‑world reach‑avoid settings where evaders succeed by reaching designated exits. R2PS is also based on algorithm 1. The no‑exit assumption may therefore limit practical relevance.
- Need more baselines: Table 2 compares against PSRO, which is too narrow to convincingly demonstrate the advantage of the proposed method. Moreover, capture rates are limited on several graph, making it unclear whether the graphs are difficult or the R2PS is not good enough. More stronger baselines would clarify this.
- Ablations on teacher guidance: Given the sparse SAC reward, the DP‑guided teacher signal may be the dominant factor behind R2PS’s performance. Ablations that remove or vary the teacher signal are needed to identify the main source of gains.

### Questions
- R2PS leverages a DP‑guided teacher term, while PSRO does not. This extra signal can improve both sample efficiency and final performance. Do you think the absence of teacher guidance contributes to PSRO’s weaker results in Table 2?
- Could the authors elaborate on how the real-time lookahead tree is constructed under partial observability? Specifically, how are the belief nodes expanded and pruned, and how does the algorithm manage the exponential growth of possible observation histories?
- Given that the method involves repeated regret minimization over a belief tree at each step, how does R2PS scale with larger grids or more complex environments? Are there any runtime metrics or profiling results that show it can be deployed in real-time?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 5

### Summary
This paper addresses the challenging problem of pursuit-evasion games on graphs under partial observability, adversarial evaders, and real-time decision constraints. The authors propose R2PS, a framework that combines theoretically grounded dynamic programming for worst-case evader modeling with belief-state-based reinforcement learning using GNNs and pointer networks. The method is evaluated on a diverse set of synthetic and real-world graphs, demonstrating strong generalization to unseen environments and outperforming baselines.

### Strengths
1. Well-Motivated and Practical Problem:
   The focus on partially observable PEGs with worst-case adversaries is highly relevant to real-world security applications. The use of real-world map data (Google Maps, Scotland Yard board game) strengthens the practical grounding.

2. Strong Theoretical Foundation:
   The paper provides a formal justification for using DP-based evaders under asynchronous move assumptions (Theorem 1), which is non-trivial and crucial for modeling a truly adversarial opponent. This theoretical anchor elevates the work beyond purely empirical RL approaches.

3. Innovative Integration of Belief and RL:
   The belief preservation mechanism—which maintains a distribution over possible evader locations and integrates it into the policy network—is elegantly designed and effectively addresses partial observability without resorting to full POMDP solvers (which are intractable at scale).

4. Real-Time Feasibility:
   The use of GNNs + pointer networks enables fast inference, making the approach suitable for deployment—a notable advantage over planning-based POMDP methods.

### Weaknesses
1. Limited Ablation on Belief Representation:
   While the belief mechanism is central, the paper lacks ablation studies on how belief is encoded (e.g., histogram vs. learned embedding) or how belief update frequency affects performance. Is the gain primarily from belief averaging, or from the specific network architecture?

2. Scalability Beyond ~200 Nodes Unclear:
   All test graphs have ≤231 nodes. It’s unclear how R2PS scales to larger urban or infrastructure networks (e.g., city-scale road graphs with 10⁴–10⁵ nodes). GNNs may suffer from over-smoothing or memory bottlenecks.

3. Assumption on Evader Observability:
   The DP evader is assumed to have global observation, which is realistic for a worst-case adversary. However, the paper does not explore asymmetric information settings where the evader also has limited sensing—a more balanced and arguably more realistic scenario.

4. Reproducibility Concerns:
   Graph construction from real-world locations (e.g., “Downtown Map”) lacks methodological detail: How were nodes/edges extracted from Google Maps? What’s the granularity?

5. Comparison to Modern MARL Methods:
   While Grasper and EPG are cited, the paper does not compare against recent multi-agent RL baselines that handle partial observability (e.g., QMIX, MAVEN, or RODE). It’s unclear whether the gains are due to the belief mechanism or simply better architecture.

### Questions
1) While the belief mechanism is central, the paper lacks ablation studies on how belief is encoded (e.g., histogram vs. learned embedding) or how belief update frequency affects performance. Is the gain primarily from belief averaging, or from the specific network architecture?
2) Graph construction from real-world locations (e.g., “Downtown Map”) lacks methodological detail: How were nodes/edges extracted from Google Maps? What’s the granularity?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2