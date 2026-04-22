# Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6, 6

## Abstract
For a two-player imperfect-information extensive-form game (IIEFG) with $K$ time steps and a player action space of size $U$, the game tree complexity is $U^{2K}$, causing existing IIEFG solvers to struggle with large or infinite $(U,K)$, e.g., differential games with continuous action spaces. To partially address this scalability challenge, we focus on an important class of 2p0s games where the informed player (P1) knows the payoff while the uninformed player (P2) only has a belief over the set of $I$ possible payoffs. Such games encompass a wide range of scenarios in sports, defense, cybersecurity, and finance. 
We prove that under mild conditions, P1's (resp. P2's) equilibrium strategy at any infostate concentrates on at most $I$ (resp. $I+1$) action prototypes. When $I\ll U$, this equilibrium structure causes the game tree complexity to collapse to $I^K$ for P1 when P2 plays best responses, and $(I+1)^K$ for P2 in a dual game where P1 plays best responses. We then show that exploiting this structure in model-free multiagent reinforcement learning and model predictive control leads to significant improvements in learning accuracy and efficiency from SOTA IIEFG solvers. Our demonstration solves a 22-player football game with continuous action spaces and $K=10$ time steps, where the offense team needs to strategically conceal their play until a critical moment in order to exploit information advantage. Code is available [here](https://github.com/ghimiremukesh/cams/blob/iclr/).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the problem of solving two-player zero-sum differential games with continuous state/action spaces and one-sided incomplete information (where P1 knows the payoff type, and P2 only has a prior belief). The core theoretical contribution is proving that the Nash Equilibrium (NE) strategies for both players have an "atomic" structure: P1's behavioral strategy concentrates on at most I actions, and P2's on at most I+1 actions, where I is the number of possible payoff types.

### Strengths
The proposed CAMS framework effectively utilizes the theoretical structural guarantee to bypass the curse of dimensionality associated with continuous action spaces. 

The idea of solving a smaller, reformulated nonconvex-nonconcave problem independent of the original action space size is powerful and clever.

The work bridges differential game theory and computational game theory, offering a viable path to solve realistic problems.

### Weaknesses
1. The goal of the paper is to solve an IIEFG with a very large action space. I noticed that the authors claim that there is currently no insightful action-state-time abstraction method, but in my opinion there are lots of works on action abstraction and reducing the size of the game tree. The authors do not fully discuss the differences between this work and existing works [1-20] in the paper. The author should discuss why the previous method of dealing with continuous action space in IIEFGs fails in the 2p0s1 incomplete information game.

2. The introduction is not very well written. The author abruptly raises the 2p0s1 problem, but I don't think this problem is very important because the assumptions are too idealistic. The author's description in the contribution is also not convincing enough. There seems to be no theoretical breakthrough, and I don't see enough innovation in the algorithm.

3. In line 113, the author claims that “convergence guarantee only exists if the NE lies in the interior of the simplex ∆(U)”, which requires citing relevant literature. There are already some works [4, 21] that give bounds for infinite U.

4. The authors use a gridded value approximation algorithm, but do not discuss or use common value approximation methods in IIGs [22-27]. This is because the value of the information set in IIGs changes with strategy, and simple approximations may not fully reflect the true value. While the multigrid method is used to accelerate the value approximation, the paper lacks a compelling explanation or theoretical guarantee for why it is particularly effective or necessary for solving the underlying Hamilton-Jacobi-Isaacs PDEs in this context. The benefits remain empirically observed but not deeply justified.

5. The writing is often dense and confusing, with terminology and notation that are not well-aligned with the broader computational game theory literature. Key concepts, proofs, and the overall narrative are not self-contained enough for a general audience, making it difficult to assess and appreciate the contributions fully.

6. The paper claims to optimize the complexity from U^{2K} to I^K. In my opinion, optimizing the metric from 2K to K is similar to the approach used in CFR. The optimized CFR [28, 29] can enumerate information sets during iterations, rather than states. The complexity reported in this paper is not fundamentally different from CFR, and the claim that the CFR algorithm is U^{2K} requires further discussion.

7. There is already work that studies the Nash equilibrium strategies of both players when one player reveals her strategy [30, 31]. This work needs to further elaborate on its novelty.

8. The experimental benchmarks used in the paper, including the selection of games, metrics, and the fairness of method comparisons, need to be verified. Achieving outstanding performance on widely recognized games is urgently needed (e.g. StarCraft).

[1] David Schnizlein, Michael H. Bowling, and Duane Szafron. Probabilistic state translation in extensive games with large action sets. IJCAI 2009

[2] John Alexander Hawkin, Robert Holte, and Duane Szafron. Automated action abstraction of imperfect information extensive-form games. AAAI 2011

[3] John Alexander Hawkin, Robert Holte, and Duane Szafron. Using sliding windows to generate action abstractions in extensive-form games. AAAI 2012

[4] Sam Ganzfried and Tuomas Sandholm. Action translation in extensive-form games with large action spaces: Axioms, paradoxes, and the pseudo-harmonic mapping. IJCAI 2013

[5] Noam Brown and Tuomas Sandholm. Regret transfer and parameter optimization. AAAI 2014

[6] Noam Brown and Tuomas Sandholm. Simultaneous abstraction and equilibrium finding in games. IJCAI 2015

[7] Shuxin Li, Youzhi Zhang, Xinrun Wang, Wanqi Xue, and Bo An. CFR-MIX: solving imperfect information extensive-form games with combinatorial action space. IJCAI 2021

[8] Carlos Martin and Tuomas Sandholm. Finding mixed-strategy equilibria of continuous-action games without gradients using randomized policy networks. IJCAI 2023

[9] Boning Li, Zhixuan Fang, and Longbo Huang. RL-CFR: Improving action abstraction for imperfect information extensive-form games with reinforcement learning. ICML 2024

[10] Linjie Xu, Diego Perez Liebana, andAlexander Dockhorn. Strategy Game-Playing with Size-Constrained State Abstraction. CoG 2024

[11] Carlos Martin and Tuomas Sandholm. Joint-Perturbation Simultaneous Pseudo-Gradient. IJCAI 2025

[12] Boning Li and Longbo Huang. Efficient online pruning and abstraction for imperfect information extensive-form games. ICLR 2025

[13] Carlos Martin and Tuomas Sandholm. Solving Infinite-Player Games with Player-to-Strategy Networks. arxiv 2025

[14] Weijun Zeng, Yinghao Li, Xiaosi Chen, Zijie Chang, Fei Ge. GNN-ReBeL: Enhancing Neural Belief Representation for Imperfect-Information Games. IEEE SMC 2025

[15] Weijun Zeng, Yinghao Li, Xiaosi Chen, Zijie Chang, Fei Ge. An Investigation of Subgame Depth in ReBeL: Impact on Convergence and Performance in Imperfect-Information Games. IEEE SMC 2025

[16] David Abel, Nate Umbanhowar, Khimya Khetarpal, Dilip Arumugam, Doina Precup and Michael L. Littman. Value Preserving State-Action Abstractions. AISTATS 2020

[17] Noam Brown, Tuomas Sandholm and Brandon Amos. Depth-Limited Solving for Imperfect-Information Games. NeurIPS 2018

[18] Trevor Davis, Kevin Waugh and Michael Bowling. Solving Large Extensive-Form Games with Strategy Constraints. AAAI 2019

[19] Samuel Sokota, Gabriele Farina, David J. Wu, Hengyuan Hu, Kevin A. Wang, J. Zico Kolter and Noam Brown. The Update-Equivalence Framework for Decision-Time Planning. ICLR 2024

[20] Sam Ganzfried. Algorithm for Computing Approximate Nash Equilibrium in Continuous Games with Application to Continuous Blotto. Games 2021

[21] Christian Kroer and Tuomas Sandholm. Discretization of Continuous Action Spaces in Extensive-Form Games. AAMAS 2015

[22] Rosemary Emery-Montemerlo, Geoffrey J. Gordon, Jeff G. Schneider and Sebastian Thrun. Approximate Solutions for Partially Observable Stochastic Games with Common Payoffs. AAMAS 2004

[23] Anne Souquière. Approximation and representation of the value for some differential games with asymmetric information. Int. J. Game Theory 2010

[24] Auke J. Wiggers and Frans A. Oliehoek and Diederik M. Roijers. Structure in the Value Function of Two-Player Zero-Sum Games of Incomplete Information. ECAI 2016

[25] Gabriele Farina and Tuomas Sandholm. Fast Payoff Matrix Sparsification Techniques for Structured Extensive-Form Games. AAAI 2022

[26] Vojtech Kovarík, Dominik Seitz and Viliam Lisý, Jan Rudolf, Shuo Sun and Karel Ha. Value functions for depth-limited solving in zero-sum imperfect-information games. Artif. Intell. 2023

[27] Ratip Emin Berker, Emanuel Tewolde, Ioannis Anagnostides, Tuomas Sandholm and Vincent Conitzer. The Value of Recall in Extensive-Form Games. AAAI 2025

[28] Michael Johanson, Kevin Waugh, Michael H. Bowling and Martin Zinkevich. Accelerating Best Response Calculation in Large Extensive Games. IJCAI 2011

[29] Michael Johanson, Nolan Bard, Marc Lanctot, Richard Gibson, and Michael Bowling. Efficient nash equilibrium approximation through monte carlo counterfactual regret minimization. AAMAS 2012

[30] Samuel Sokota, Ryan D'Orazio, Chun Kai Ling, David J. Wu, J. Zico Kolter and Noam Brown. Abstracting Imperfect Information Away from Two-Player Zero-Sum Games. ICML 2023

[31] Weiming Liu, Haobo Fu, Qiang Fu and Wei Yang. Opponent-Limited Online Search for Imperfect Information Games. ICML 2023

### Questions
1. In the problem setting of this paper, does P2 know that P1 knows its belief distribution? If P2 knows that P1 knows its belief distribution, why does P2 let P1 know its belief distribution and do not modify the distribution? I think the problem setting is not very clear.

2. What are the specific steps in the LLM proof? How do you ensure the correctness of the proof? Since I am not good at theoretical analysis, I cannot judge whether the proof is valid.

3. Are the value estimates used only for the CAMS algorithm, or for all comparison methods (CFR, etc.)? If only the CAMS algorithm is used, it will be an unfair comparison.

4. The atomic structure is proven under the Isaacs' condition, which guarantees the existence of pure-strategy NEs in the "non-revealing" games. Could you discuss scenarios or game classes where this condition might fail and what the implications would be for the atomic structure of the overall NE?

5. Algorithms like CFR can be parallelized. Can CAMS be parallelized? Is your experiment parallelized?

6. What does Action Error mean in an experiment?

7. How is I generated from U? What is the size of I?

### Soundness
2

### Presentation
1

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
This paper formulates the football control problem as a two-player zero-sum differential game under asymmetric information, where the attacking agent has full state observability while the defending agent has only partial observations. The authors derive the equilibrium structure of such 2P0S games and exploit it to design robust, feedback-based strategies for both agents. By leveraging this game-theoretic formulation, the proposed method computes strategies that are provably robust against worst-case responses. Empirical results in the Google Research Football Environment demonstrate the effectiveness of this approach across multiple offensive-defensive scenarios, showcasing improved coordination, defensive anticipation, and sample efficiency compared to traditional reinforcement learning baselines.

### Strengths
- Theory–structure: Clear, well-motivated primal–dual formulation with rigorous atomic-support results (P1 $I$-atomic, P2 $(I{+}1)$-atomic) and a consistency theorem, explaining why equilibria lie on the simplex boundary for 2p0s1.
- Practical impact: Simple but powerful architectural change (action prototypes + logits) that plugs into value approximation, MARL, and MPC and delivers sizable accuracy/efficiency gains; compelling football demonstration otherwise out of reach for standard IIEFG solvers.

### Weaknesses
- Empirics emphasize action error and qualitative behavior. However, exploitability or best-response value gaps to certify closeness to NE are not systematically reported.
- The theoretical framework hinges on the 2P0S game structure, where only one player lacks full observability. However, in realistic football environments, both agents typically face partial observability.

### Questions
- How does the atomic structure and the primal–dual DP extend when dynamics or observations are stochastic so P1 cannot precisely steer beliefs?
- Can the authors elaborate on how feedback Nash equilibria are computed in practice for the 2P0S differential game setting? Is the computation exact, or does it involve discretization, function approximation, or numerical solvers? If approximations are used, how is the error controlled or bounded?
- The paper seems to focus on relatively isolated one-vs-one or two-agent scenarios. Can the framework be extended to team-based settings with multiple attackers and defenders? How would the equilibrium computation scale or need to change?

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
2

### Summary
This paper studies the specific case of zero-sum differential games where the only form of imperfect information is player 1's type (with a common prior). 

The key contribution is the proof that even though the action space (at every state) can be large or continuous, the support in the NE (behavioral strategy) is at most I (or I+1), where I is the number of types. This characterization of the equilibria can be utilized to solve problems in MARL and MPC.

### Strengths
- I generally enjoyed reading the paper. Most papers regarding game solving suffer from an overwhelming complexity in notation, and this paper is no exception; however, the authors partially alleviated this by having text that explains most of the detail well. 
- Experiments are generally well done, with a reasonably objective explanation of the results involved (e.g., Sec 6.2).
- The literature on solving continuous time, continuous action games with imperfect information is relatively limited; even though this is a very limited case, the empirical ability to scale looks like it will be a good addition to the literature.

Note: This is slightly outside my field of expertise, so I cannot confidently comment on novelty nor quality.

### Weaknesses
- There are several terms which I felt were not well defined, and a simple search did not did yield satisfactory definitions, e.g., I-"atomic".
- There are some technical questions I have/comparisons that I felt were unfair (see below)
- I found the introduction on the multigrid speedup difficult to understand and perhaps difficult to follow for the average reader. Given that it is an important component of getting good performance, perhaps a figure or summary would be more effective, at least in the main paper.

### Questions
- I am not convinced about the example given in Figure 1 and line 81. I agree that the branching factor is I instead of the (discretized) U. However, I do not think it is fair to compare the size of a IIEFG and the "primal and dual games" introduced by the author; specifically the way the depth of the search tree is K instead of 2K. 
- Is "action error" an appropriate measure of performance (Figure 5)? I would have thought the regular notion of exploitability is more appropriate in a 2p0s setting.
- How does the policy network of P2 look like (e.g., section 5.2). I think it is nice (and consistent with the argument that CAMS-DRL works better than alternatives when NE do not lie in the strict interior) that the architecture allows for this by having action prototypes. Could the authors comment on how the I+1 action prototypes were learnt? I would think there may be a brittleness problem here. For sxample, if due to say, local minima, not all action protypes were learnt (e.g., if two action prototypes turned out to be near identical), then the restriction of outputting only a logit of size I+1 can lead to very a distribution that is very suboptimal since one prototype was "wasted".
-Typos: Line 45 (uniformed vs uninformed)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies two-player zero-sum differential games with one-sided incomplete information. They identify an atomic equilibrium structure, which dramastically reduces the game tree complexity. Base on this, they propose a primal-dual DP reformulation and establishes convergence guarantees. They also leverage this atomic property to develop several scalable solvers. Experiments  shows large reductions in computational cost and improved solution accuracy compared to SOTA solvers.

### Strengths
- The paper is generally well organized.
- The paper identifies an atomic Nash structure that theoretically reduces the exponential complexity of imperfect-information differential games. The atomic-NE theorem is clean, leverages convexification geometry, and directly leads to algorithmic simplifications exploited throughout the paper. 
- It provides a unifying primal–dual reformulation, which bridges classical differential-game theory and RL/Control.
- The paper validates using realistic case study. The football experiment is an impressive  demo that this theoretical structure can yield large practical gains when assumptions hold.

### Weaknesses
- Assumption A5 is kind of strong. It requires full knowledge of dynamics and perfect recall, but many interesting real-world problems might violate these. The atomic results assumes the ability of P1 to precisely control the public belief, and the paper notes the stochastic case only yields lower bounds. These limitations should be emphasized when claiming broad applicability.  
- The CAMS value-approximation procedure requires solving many small minimax problems and training value networks. 
- Thm 5.1 gives a high-level complexity bound but constants and practical solver behaviour  are sensitive and not fully characterized.

### Questions
- Provide a short, explicit toy example, e.g. one or two time steps in the main text to illustrate how the convexification produces the I atoms.
- The paper mentions that P1’s inability to precisely control belief in stochastic settings and that the convexified Bellman operator becomes a lower bound. Is it possible to extend CAMS to stochastic settings?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies a class of two-player zero-sum differential games with one-sided information (2p0s1) and proves that equilibrium strategies are atomic in the number of payoff types $I$. P1 needs at most $I$ action prototypes and P2 at most $I+1$. This collapses game-tree complexity by reducing the effective branching factor from $U$ (all actions) to $I$ or $I+1$. Building on this, the authors implement three solvers to exploit this problem structure: a model-based pipeline (CAMS), a model-free RL version (CAMS-DRL using PPO/MMD), and a control variant (CAMS-MPC). The authors show learning speed and performance improvements vs methods that use large discrete action spaces. They also demonstrate behavior on an 11v11 American football domain.

### Strengths
- The paper is well written with concepts presented clearly.
- The proofs of $I$ and $(I+1)$-atomicity are useful and relevant to solving large 2p0s games with asymmetric information.
- The authors show empirical gains by leveraging this problem structure across a range of algorithmic approaches.
- OpenSpiel-compatible implementations are provided.

### Weaknesses
1) PPO/MMD hyperparameters and any relevant sweeps are not listed in the paper. It's possible that not enough hyperparameter tuning was done (for both the proposed method and baselines).
  

Rudolph et al. (2025) describe that (well-regularized) PPO and MMD are quite sensitive to the entropy coefficient. While they generally recommend quite high coefficients like 0.1 or 0.05, if PPO and MMD have issues converging to non-interior solutions, perhaps there are lower entropy coefficients (or different learning rates) that are sufficient for this domain, which is quite different from dark hex/PTTT?

It would be great to see a minor sweep on these baselines to make sure that PPO/MMD isn't flatlining just because, e.g. the entropy regularization is too high.

Rudolph et al. Reevaluating policy gradient methods for imperfect-information games. 2025.

--

2) Adding exploitability or approximate exploitability comparisons could improve this paper to help measure convergence to NE. This is less important as the authors provide other indicators of convergence.

### Questions
Your algorithms parameterize each player’s policy as a mixture over $I$ or $I+1$ action prototypes, relying on the paper’s existence result for atomic equilibria. In practice, how robust are the learned policies to an unconstrained best response that is not parameterized atomically (and instead operates over the raw action space)? How large is the risk that restricting your learning process to atomic mixtures can allow non-atomic opponents to induce out-of-distribution sequences and inputs in agents trained with the proposed method?

### Soundness
3

### Presentation
3

### Contribution
3
