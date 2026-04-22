# Potentially Optimal Joint Actions Recognition for Cooperative Multi-Agent Reinforcement Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
Value function factorization is widely used in cooperative multi-agent reinforcement learning (MARL).
Existing approaches often impose monotonicity constraints between the joint action value and individual action values to enable decentralized execution.
However, such constraints limit the expressiveness of value factorization, restricting the range of joint action values that can be represented and hindering the learning of optimal policies.
To address this, we propose Potentially Optimal Joint Actions Weighting (POW), a method that ensures optimal policy recovery where existing approximate weighting strategies may fail.
POW iteratively identifies potentially optimal joint actions and assigns them higher training weights through a theoretically grounded iterative weighted training process. We prove that this mechanism guarantees recovery of the true optimal policy, overcoming the limitations of prior heuristic weighting strategies.
POW is architecture-agnostic and can be seamlessly integrated into existing value factorization algorithms.
Extensive experiments on matrix games, difficulty-enhanced predator-prey tasks, SMAC, SMACv2, and a highway-env intersection scenario show that POW substantially improves stability and consistently surpasses state-of-the-art value-based MARL methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new way to identify the optimal joint actions to weight the learning of value-decomposition methods. An additional recognition $Q$-function is learned, which induces a set of actions that may potentially be optimal. While its learning proceeds toward approximating the optimal values, this set gets further refined, until only the optimal actions are present. Experimental results on a wide set of different problems show the better performances achieved by this method, as well as how it can be combined with general value-decomposition algorithms and improve over their vanilla versions.

### Strengths
The issue of overcoming the representation limitations of existing value-decomposition methods, for example through mean of weighting their learning updates as done by this paper, is a key direction to deliver better MARL algorithms. The idea of doing so by identifying the optimal joint actions in an effective way is a viable path for this, and already proved capable of achieving improvements in driving existing methods, as with WQMIX. The proposed set of experiments is very wide and varied, and the reported empirical results are strong. The paper is in general quite clear and easy to follow.

### Weaknesses
I struggle to understand some of the implementation choices made here: for example, the use of a separate $Q_r$ over the already existing $Q^\*$ is not entirely justified to me. Also, it is not completely clear how the proposed method should overcome the limitations identified in the existing CW-QMIX and OW-QMIX algorithms. Finally, the cost (in terms of computational time) of the proposed method seems a bit hidden and not sufficiently and clearly highlighted, making it difficult to actually assess the trade-off one has to make when choosing the proposed method over other ones. Please see the Questions below for a more in-details explanation.

### Questions
- You claim that, being trained against the optimal $Q^\*$, the maximizing actions of $Q_r$ are going to be the same as those of $Q^\*$ itself, and thus the set $A_r$ will include them. But the training of both $Q_r$ and $Q^\*$ is done simultaneously, and thus we are not guaranteed that their interplay is accurate before convergence occurred. This would probably lead to similar problems to those you highlighted for CW-QMIX and OW-QMIX (i.e., inaccuracies due to learning) no? Am I missing something here?

- I struggle to understand how the additional $Q_r$ is bringing any benefit over the use of $Q^\*$ directly to identify the optimal joint actions: at convergence, these will both represent the same action-value function, no? And $Q_r$ is indeed trained to chase the optimal unrestricted $Q^\*$. So why adding an additional structure rather than simply restructuring $Q^\*$ itself to be formulated as Equation (3)?

- When computing $w(s,\mathbf{a})$, how do we check if $\mathbf{a}\in A_r$? If we need to explicitly construct the set of recognized optimal actions $A_r$, then this may be a quite expensive operation. Such an aspect should be eventually stated clearly.

- It would be good to explicitly state the loss function for training $Q^\*$ in the paper, as currently it can only be found in Figure 1.

- In Figure 2, why are the values for $Q_1$ and $Q_2$ of ResQ different between sub-figure (h) and (i)?

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
This paper proposes POW (Potentially Optimal Joint Actions Weighting), a value-decomposition method for cooperative MARL. It introduces a recognition module $Q_r$ that identifies potentially optimal joint actions and up-weights them during training. The authors prove convergence guarantees and evaluate POW on matrix games, predator–prey, SMAC, and other benchmarks, demonstrating improved performance over QMIX, WQMIX variants, and other MARL algorithms.

### Strengths
1. The paper addresses the gap between WQMIX theory and its heuristic approximations.
2. The authors provide proof that the recognized action set converges to optimal actions.
3. The method was tested on extensive benchmarks, showing performance gain across both monotonic and non-monotonic tasks.

### Weaknesses
See the questions section.

### Questions
Questions about the current manuscript:
1. Will $Q_r$ and iterative loops increase implementation difficulty compared to QMIX or QPLEX? If so, how can this be mitigated?
2. May not be strictly required, but I wonder if any possible comparisons with CTDE actor-critic MARL methods like MAPPO?
3. Can we include the algorithm pseudo-code for clarity?
4. [Minor]: Line 423: the cited work REMIX seems not to match the given reference. Please check.

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
This paper proposes a new method named Potentially Optimal Joint Actions Weighting (POW) to address the problem of representativeness of a wide range of value factorization functions. This is achieved by assigning higher weights to actions that are found with potential to be optimal.

### Strengths
This work good theoretical groundings and a comprehensive range of experiments across different MARL environments. The matrix games provided are also useful to demonstrate the representational properties of the method and support the claims made. Theorem 1 and definition 1 are sound.

### Weaknesses
A number of different methods for value function factorization has been explored recently, and some of them can theoretically guarantee the factorization of any family of the environments (such as QTRAN). I feel the motivations for the proposed method are hence not sufficient, simply by saying that their object is to try to improve the expressiveness of the range of factorisation functions that can be represented.

In section 4.5.2, WQMIX seems to be better than POW in Figure 7(a) and 7(b); it is unclear to me how these results show that the performance of POW "stems from its recognition-weighting design, not from parameter count" (line 400); what it shows is that other methods such as WQMIX improve when the network size increases and QPLEX stays the same; it is unclear how it says something about POW's parameter count and in fact it shows that WQMIX performs better in 7(a) and 7(b) when the number of parameters is the same.

Some notations could also be made more clear; for instance, in figure 1, it is unclear to me the meaning of $A_{tot}$.

Please find below some more questions that reflect other concerns.

### Questions
1. in lines (168-169): " In all our experiments, we set $\alpha=0$, so only actions in $A_r$ contribute to updates, aligning theory with practice." - so what is the pointof proposing this $\alpha$ weight?
2. i wonder what happens if $\alpha$ is not 0? has that been tested?
3. could the authors elaborate on what is the loss "$L_{Q^*}$" in figure 1?
4. in eq 4, is the optimal value function $Q_*$ also learned? or is it known a priori? from the notation it seems almost like an on policy approach being described, which makes the link between notation and practice a bit unclear
5. in figure 3, the performances across the 3 different levels of penalties seem a bit inconsistent; for example, could the authors elaborate on why the proposed method performs better in p=-4 and p=-5 but not in p=-3?

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
The paper proposes POW, a value decomposition based MARL method under CTDE for optimal joint policy recovery using recognition weighting design. Empirically, the method demonstrates improved performance over baselines on multiple benchmarks

### Strengths
- This paper is generally well written and easy to follow; With a constrained Q_r to over include promising joint actions and focus learning signal on them and bridges the gap of WQMIX in ideal weighting and practical approximation.

- There are multiple benchmarks demonstrating the effectiveness empirically.

### Weaknesses
- Some of the claims can be too strong. For theorem 1, it assumes the condition of Q_r can converge to  ${Q^\star}$, however  ${Q^\star}$ is also unknown and to be learned. E.g in eq.4 it is not using \hat{Q^\star} but ${Q^\star}$, yet also claims Q_r is optimizized to $\hat{Q^\star}$. The gurantee in theorem 2 also requires A_r to converge to only optimal a; These are the actual challenging points in practice and cannot be just assumed; This setting is not beneficial to the actual experiment or pracitcal settings.

- Also as above, Q_r and Q_tot rely on $ \hat{Q^\star}$ for bootstrapping, however how the bias in  $\hat{Q^*}$ will effect A_r and Q_r is not discussed. Under extrem case they will not converge at all.

- Some of the SOTA works are missing. This paper only compares with some of the classic Cooperative MARL works.

### Questions
- I believe in multiple equations $ \hat{Q^\star}$ and  ${Q^\star}$ and mixed and used against the verbal description in paper, making it hard to understand if it's typo, approximation or error.

- Q_r(s,a) and Q_r (\tau, a) are mixed and used; According to verbal description it should be Q_r(s,a)?

- I wonder the C in eq5 can be discussed in terms of its sensitivity

- Is Q^* and Q_tot network the same as in WQMIX? Also I wonder the \alpha setting used for WQMIX in the experimental setting

### Soundness
2

### Presentation
3

### Contribution
3
