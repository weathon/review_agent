# Search Inspired Exploration for Reinforcement Learning

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Exploration in environments with sparse rewards remains a fundamental challenge for reinforcement learning (RL). Existing approaches such as curriculum learning and Go-Explore often rely on hand-crafted heuristics, while curiosity-driven methods risk converging to suboptimal policies. We propose Search-Inspired Exploration in Reinforcement Learning (SIERL), a novel method that actively guides exploration by setting sub-goals based on the agent's learning progress. At the beginning of each episode, SIERL chooses a sub-goal from the \textit{frontier} (the boundary of the agent’s known state space) before the agent continues exploring toward the main task objective. The key contribution of our method is the sub-goal selection mechanism, which provides state-action pairs that are neither overly familiar nor completely novel. It assures that the frontier is expanded systematically and that the agent is capable of reaching any state within it. Inspired by search, sub-goals are prioritized from the frontier based on estimates of cost-to-come and cost-to-go, effectively steering exploration towards the most informative regions. In experiments on challenging sparse-reward environments, SIERL outperforms dominant baselines in both achieving the main task goal and generalizing to reach arbitrary states in the environment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Search Inspired Exploration for RL (SIERL), a goal-conditioned method that expands a frontier of explored state-action pairs and selects sub-goals via a priority score that balances novelty with the learned estimates of $Q$-values. Training alternates between two steps of reaching a sampled sub-goal and pursuing the main task goal. Experiments on customized MiniGrid Hallway variants, BugTrap, and FourRooms environments show that SIERL performs competitively or better than HER, novelty bonuses, and other baselines.

### Strengths
* The framing of exploration as a search problem that progressively grows a frontier of reachable states is elegant and aligns with the intuition of structured exploration. 
* The two-phased training approach naturally induces a curriculum from easy-to-reach subgoals toward the main goal. The ablations also demonstrate that removing the frontier or switching the components harms the performance of the agent.
* On discrete navigation tasks, SIERL reliably reaches main goals and achieves notably higher success on arbitrary goals, suggesting improved general exploration and goal generalization compared to discussed baselines.

### Weaknesses
The paper has the following main issues:
* Ambiguity and potential bias in $Q$-based cost terms
* Over-reliance on the environment geometry
* Limited metrics and hyperparameter selection

Issue 1:
The paper introduces cost-to-come and cost-to-go terms derived from the learned $Q$-values, but never clearly defines how these are estimated or trained. It is unclear whether $Q$ is defined over $(s, a, g)$ or $(s, g)$ and how maximization over actions is done. Also, how do these values interact with the alternating exploration phases? Since the same $Q$-values guide both sub-goal selection and policy training, the method may reinforce optimistic or inaccurate estimates.

Issue 2:
All the environments considered in the paper have their main goal at the farthest reachable location, making subgoal generation naturally align with distance. If the main goal is not at the farthest position (e.g., the main goal lies in the top-right corner for FourRooms environment), how will things change? Would the frontier still expand outward, or would it focus on goal-directed behavior? Demonstrating the robustness to different main goal placements or task geometries would test whether SIERL truly adapts its exploration frontier. 

Issue 3:
In Figure 2, it’s not entirely clear whether the shaded regions represent confidence intervals or variance. Since the shaded areas overlap quite a bit, it’s difficult to draw strong conclusions about performance differences. Using only five random seeds may also limit the statistical confidence in the results, perhaps increasing the number of seeds or reporting statistical tests could help clarify the trends.
It also looks like several hyperparameters (such as phase horizons, familiarity thresholds, and percentile cutoffs) were manually tuned for each environment. It would be helpful if the paper could explain how these hyperparameters were selected. For example, whether they were tuned on validation environments or through a general heuristic, it will help to better understand the fairness and reproducibility of the comparisons.

Since the environments considered are discrete, including coverage metrics would offer a clearer, more quantitative view of each method’s exploration capability. Since the paper discusses other exploration strategies such as pseudo-count and frontier-based methods (RND, Go-Explore, LEAF, TLDR), comparing them using this metric would further strengthen the evaluation.

### Questions
* What was the reason for choosing deterministic environments? Could SIERL scale to stochastic settings, and would that affect frontier expansion? 
* How would the frontier behave if the main goal is not at the farthest location? 
* Could you provide a small experiment or visualization validating the curriculum aspect? 

Addressing these would greatly improve both the clarity and empirical credibility of the paper. 

---
*Minor comments*:
* Define the $\operatorname{softmin}$  in Section 4.3 and specify all score weights explicitly. Define $z(\cdot)$, $\sigma(\cdot)$ and vector weights once.
* The BugTrap environment looks similar to the hallway environment, so what’s the main reason for the significant difference in performance in this environment?
* Proofread for minor formatting issues (cost free,meaning Pg. 4)
* Since SIERL doesn’t reward raw novelty, it can plausibly avoid the noisy-TV problem. Highlighting this point will be nice to the paper. You can further test it in a stochastic or distractor-rich environment.

### Soundness
2

### Presentation
2

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
The paper studies exploration in sparse-reward reinforcement learning (RL).
The paper proposes a method that lets the agent attempt intermediate subgoals which are (1) at the frontier of its capabilities and (2) in the right "direction" as estimated by the agent's learned Q-values.
The paper then evaluates this method extensively in low-dimensional discrete environments

### Strengths
The paper studies an important and challenging problem: effective exploration without dense rewards.
The paper studies the proposed method systematically in reasonable lower-dimensional discrete environments.

### Weaknesses
The main weakness of the paper is that the proposed method is extremely closely related to [1]. [1] proposed a method, which also selects intermediate subgoals from a reachable frontier and balancing exploration-exploitation by balancing cost-to-come and cost-to-go. Moreover, [1] evaluates this method in high-dimensional navigation & manipulation tasks which closely resemble the environments studied in this paper.

I think this work has the potential to contribute to the field, but in my view this paper would need to systematically compare against [1].
To sufficiently contrast with [1] and systematically evaluate extensions to [1], the paper would most likely have to be rewritten substantially.
It might also be informative to evaluate against other methods to automatically select subgoals as referenced in [1], such as the common MEGA [2].
In the current state, I unfortunately cannot recommend the paper for acceptance.

Additional points:
* The proposed method seems a bit more complex than [1] and it would be interesting to analyze which extensions are necessary. The methods section is also hard to follow in some places. It might be useful to include a more detailed algorithm box within the main paper.
* It would be good to systematically ablate the hyperparameters such as $w_n, w_c, w_g$ and $F_{\pi}^{thr}$.

[1]: Diaz-Bone et al., DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning. https://arxiv.org/pdf/2505.19850
[2]: Pitis et al., Maximum entropy gain exploration for long horizon multi-goal reinforcement learning. https://arxiv.org/pdf/2007.02832

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SIERL (Search-Inspired Exploration for RL), a goal-conditioned exploration method that systematically expands a frontier of known state–action pairs and sets sub-goals from this frontier using a priority that blends (i) a familiarity/novelty filter and (ii) search-style cost-to-come and cost-to-go estimates derived from learned Q-values. Each episode alternates between two phases: (1) reach a selected frontier sub-goal to push the boundary outward, then (2) pursue the main task goal from that more informative starting point. Experiments on MiniGrid variants (Hallway, FourRooms, BugTrap) show higher main-goal success and better generalization to random goals than baselines; ablations confirm the importance of early switching, frontier filtering, and prioritization. Contributions include the frontier extraction + sub-goal selection mechanism, a Hallway benchmark controlling action-sequence difficulty, and an empirical study disentangling component effects.

### Strengths
- Introduces a frontier extraction + prioritization mechanism using a familiarity filter and softmin over cost-to-come/go estimates, giving a principled way to pick feasible, informative sub-goals rather than random waypoints.

- Provides clear algorithmic structure and pseudo-code (Algorithms 1–3), detailing phase switching (fixed horizons + probabilistic early switch), frontier maintenance, and sub-goal sampling, supporting faithful reimplementation.

- Method motivation and the exploration–exploitation dilemma in goal-MDPs with sparse rewards are clearly articulated before the two-phase strategy, making the design choices easy to follow.

### Weaknesses
- Overlap with classic goal-conditioned curricula (frontier expansion, sub-goals via reachability/cost heuristics) is high; the paper doesn’t sharply distinguish SIERL from HER-style relabeling or novelty/prioritized-goal sampling beyond the specific softmin priority and early-switch heuristic.

- The current implementation relies on visitation counts for novelty/familiarity, limiting applicability to continuous state–action spaces and making SIERL sensitive to discretization choices. The authors acknowledge this limitation and suggest pseudo-counts but do not evaluate them.

- Excluding both “too familiar” and “too novel” (s,a) pairs creates a narrow band on the frontier; while intuitive, its necessity vs. a simpler top-K novelty or Q-margin selector is not isolated.

### Questions
- Please write the exact priority function you use to select frontier sub-goals (including normalization/z-scoring, softmin temperature, and any novelty weights). How sensitive is performance to these constants? A small table of priors → success rates would help

- How exactly do you detect and maintain the frontier set (data structures, update frequency, de-duplication)? What are the tie-break rules when multiple candidates share the same priority, and how do you handle an empty or exploding frontier?

- You alternate between (A) reaching the frontier sub-goal and (B) pursuing the main goal, with a probabilistic “early switch.” What is the trigger, and how do you pick its probability/horizon? Ablate fixed vs. adaptive switching (e.g., based on estimated success probability or advantage).

- Are these derived directly from learned Q values or from a separate estimator? How do you mitigate bias when Q is poorly estimated early on? Please compare: (i) raw Q, (ii) value-ensemble uncertainty, and (iii) model-based short-rollout costs.

- Since counts don’t scale to continuous spaces, which pseudo-count or density proxy (e.g., kNN in feature space, RND density) works best for SIERL? Show results on a continuous maze (e.g., AntMaze) and analyze robustness to representation choice.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SIERL (Search-Inspired Exploration for RL), a goal-conditioned exploration scheme that maintains a frontier of candidate state–action sub-goals filtered by a familiarity/visit-count criterion. It also prioritizes sub-goals using a softmin over (novelty cost)×(weighted sum of cost-to-come & cost-to-go), and runs a two-phase episode schedule, which includes reach sub-goal then pursue the main goal with early-switching when encountering novel states. Experiments are in discrete MiniGrid-style worlds with metrics cover main-goal success and random-goal success.

### Strengths
1. The paper has clear mechanism and ablations. The frontier construction, softmin prioritization over (cost-to-come,cost-to-go), and early-switching are well specified and ablated.
2. This paper provides a framework that offers surriculum-like behavior without reward shaping, which extends its applicability. The method steers exploration by sub-goal scheduling rather than altering rewards (contrast with novelty bonuses/RND), which avoids reward hacking/noisy-TV pitfalls.
3. SIERL shows strong random-goal success, surpassing baselines.

### Weaknesses
1. The method relies on visit counts/familiarity and a discrete frontier. Despite the authors have noted this limitation and suggest pseudo-count extensions, the claims made in the paper should be narrowed to discrete, low-dimensional settings given the current status.
2. The pipeline relies on many environment-dependent hyperparameters—e.g., familiarity threshold , percentile cutoff, novelty exponent, weights components, horizons, timeout , and switch probability. Finding right hyperparameter itself becomes arguablely as difficult as define a curriculum.
​3. Evaluations stay on toy-like MiniGrid rooms. To strengthen impact, include standard hard suites (e.g., MultiRoom-N×, DoorKey-16×16) and at least a few ProcGen tasks.
4. Related work cites count-based families (hashing, pseudo-counts) and ICM, but the experiments don’t clearly include canonical count-based baselines (e.g., Tang et al. hashing counts; Bellemare/Ostrovski pseudo-counts) or ICM with matched tuning/compute. Please add one of them for a fair comparison. They are highly correlated to the paper's idea.

### Questions
1. How sensitive is SIERL to hyperparameter selctions? Provide sensitivity analyses and practical defaults/ranges.
2. Which “novelty bonus” implementations were used?
3. Can you demonstrate the scalability of SIERL? What's the task with highest dimensions that SIERL is able to solve?
4. For main-goal success, SIERL trails novelty bonuses in places, but for random-goal success it clearly outperforms on FourRooms and BugTrap. Why it is the case? If SIERL is doing well in random goal sucess, it should also finish the main goal perfectly. Is the random goal sampled here too simple compared to the actual goal?

### Soundness
2

### Presentation
3

### Contribution
2
