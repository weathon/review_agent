# DyRo-MCTS: A Robust Monte Carlo Tree Search Approach to Dynamic Job Shop Scheduling

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Dynamic job shop scheduling, a fundamental combinatorial optimisation problem in various industrial sectors, poses substantial challenges for effective scheduling due to frequent disruptions caused by the arrival of new jobs. State-of-the-art machine learning methods have been used to learn scheduling policies that can make prompt, robust decisions in response to dynamic disturbances. However, these offline-learned policies are often imperfect, necessitating the use of planning techniques such as Monte Carlo Tree Search (MCTS) to improve performance at online decision time. The unpredictability of new job arrivals complicates online planning, as decisions based on incomplete problem information are vulnerable to disturbances. To address this issue, we propose the Dynamic Robust MCTS (DyRo-MCTS) approach, which integrates action robustness estimation into MCTS. DyRo-MCTS guides the production environment toward states that not only yield good scheduling outcomes but are also easily adaptable to future job arrivals. Extensive experiments show that DyRo-MCTS significantly improves the performance of offline-learned robust scheduling policies with acceptable online planning time. Moreover, DyRo-MCTS consistently outperforms state-of-the-art MCTS algorithms across various dynamic scheduling scenarios. Further analysis reveals that its ability to make robust scheduling decisions leads to long-term, sustainable performance gains under disturbances.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes DyRo-MCTS, a robust Monte Carlo Tree Search framework for dynamic job shop scheduling (DJSS). It aims to improve online scheduling decisions under uncertain job arrivals by incorporating an action robustness estimation term into the MCTS tree policy. The method estimates robustness through machine idle-time distributions and introduces a DyRo-UCT formula balancing exploitation, exploration, and robustness. Experiments show that DyRo-MCTS improves upon both offline-learned scheduling policies (from DRL and GP) and vanilla MCTS with minimal added computation.

### Strengths
The paper addresses a practical and important problem—robust online planning in dynamic scheduling—by extending MCTS with a simple, interpretable robustness measure. The formulation is clear and methodologically consistent.

### Weaknesses
The novelty of DyRo-MCTS is limited. Integrating a robustness term into MCTS, while intuitively reasonable, constitutes an incremental extension of existing robust scheduling and MCTS formulations. Prior reinforcement learning and online planning studies have already aimed to handle incomplete or uncertain job information, yet related work comparisons are not sufficiently comprehensive. The experiments focus mainly on comparing against vanilla MCTS and lack evaluation against other modern online planning or robust RL approaches. Claims about negligible computation overhead are not quantitatively analyzed beyond MCTS iteration scaling, and no ablation or complexity analysis supports this.

### Questions
The study claims to handle online planning under incomplete information, but why are comparisons with other online or robust RL-based planners absent?

The method states minimal computational cost increase—where is the quantitative evidence or runtime breakdown to substantiate this?

How does DyRo-MCTS differ conceptually from prior robust MCTS or uncertainty-aware planning approaches?

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
4

### Summary
The paper augments MCTS for dynamic JSS (with new job arrivals) by adding a robustness score based on integrated idleness to better control weighted tardiness. An offline policy supplies action priors, and partial tree reuse is used to keep per-decision latency low.

### Strengths
The robustness term captures a practical scheduling intuition—discouraging early idle time—that is relevant when arrivals are uncertain and the objective is (weighted) tardiness.

The method needs minimal additional online planning time.

### Weaknesses
The robustness score is heuristic and domain-dependent; there is no theoretical link to optimality or selection consistency.

Coverage of scenario diversity is limited; the paper lacks analysis across varied WIP levels and machine counts.

The description of tree reuse in a dynamic environment (e.g., around line 248) is insufficient. It is unclear what statistics are retained/reset and whether reuse was actually enabled in experiments.

### Questions
1. Do you have a theoretical justification for reusing nodes when the state changes due to new arrivals? Is carrying over statistics from prior rollouts purely pragmatic? What exactly is retained (Q-values, visit counts, priors, normalization ranges), and what is reset?
2. For the first 1,000 arrivals, are the schedules identical across “w/o online planning”, “vanilla MCTS”, and “DyRo-MCTS”?
3. What is the typical number of jobs considered per decision during MCTS? Also, can you report results for settings with ≥100 concurrent jobs (WIP level) and ≥10 machines?
4. With 100 MCTS iterations and non-parallel rollouts, you report that MCTS takes around 0.021 seconds per decision. Are GP (Chen et al., 2025) and DRL (Liu et al., 2023) dispatchers in a similar decision time?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies a dynamic job shop scheduling problem (with dynamically arriving jobs), and proposes a monte carlo tree search (MCTS) method for online planning. The proposed method DyRo-MCTS integrates action robustness estimation in the tree policy of MCTS. In the experiments, the authors show the DyRo-MCTS outperforms a vanilla MCTS and improves the performance of several offline learned policies.

### Strengths
1.	Promising performance compared to offline policies and vanilla MCTS

2.	Good analysis for parameter sensitivity

### Weaknesses
1.	The authors review some MCTS applications in scheduling and note that mentioned ones are designed for static JSS. The proposed method differs from those for the targeted dynamic JSS. However, there are existing work on MCTS for dynamic JSSP that the authors may overlook such as [1-6]:

[1] Li, Kexin, et al. "An effective MCTS-based algorithm for minimizing makespan in dynamic flexible job shop scheduling problem." Computers & Industrial Engineering 155 (2021): 107211.

[2] Saqlain, M., S. Ali, and J. Y. Lee. "A Monte-Carlo tree search algorithm for the flexible job-shop scheduling in manufacturing systems." Flexible Services and Manufacturing Journal 35.2 (2023): 548-571.

[3] He, Zhou, et al. "Enhanced Monte‐Carlo tree search for dynamic flexible job shop scheduling with transportation time constraints." Expert Systems 42.2 (2025): e13727.

[4] Kim, Duyeon, and Hyun-Jung Kim. "Monte Carlo tree search-based algorithm for dynamic job shop scheduling with automated guided vehicles." 2022 Winter Simulation Conference (WSC). IEEE, 2022.

[5] Cheng, Yuxia, et al. "Smart DAG tasks scheduling between trusted and untrusted entities using the MCTS method." Sustainability 11.7 (2019): 1826.

[6] Li, Weiguan, Jialun Li, and Weigang Wu. "A dynamic scheduling algorithm with time varying resource constraints in colocation data centers." 2022 13th International Conference on Information and Communication Systems (ICICS). IEEE, 2022.

2. The key contribution of the work is to re-define the exploitation term with the weighted sum of action value and robustness, which brings a limited contribution, given that applying MCTS for dynamic scheduling is not new. 

3. In addition, the weighted term looks similar to RAVE [7] and other variants of UCT formula:

[7] Gelly, Sylvain, and David Silver. "Combining online and offline knowledge in UCT." Proceedings of the 24th international conference on Machine learning. 2007. 

4. Experiments only compare with offline policies.  Including classic online methods for dynamic scheduling are needed. It is not clear for the DRL baseline, whether the greedy or sampling strategy is used during inference.

### Questions
1.	Why does DyRo-MCTS allow jobs to be delayed, whereas MCTS does not?

2.	What is the setting (UCT terms?) of vanilla MCTS?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes to modify MCTS for the dynamic JSSP so as to improve robustness of the decisions. The robustness is defined as the early idleness of jobs and MCTS combines the usual Q value with robustness. Experiments show that the approach is beneficial and performs better than MCTS with different playout policies.

### Strengths
A novel MCTS algorithm for dynamic JSSP
The contribution is clear
Good experimental results

### Weaknesses
No comparison to other Monte Carlo Search algorithms than PUCT.

### Questions
When combining with GP do you use the GP policy as the prior in the PUCT formula?

### Soundness
4

### Presentation
4

### Contribution
4
