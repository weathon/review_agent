# ROSARL: Reward-Only Safe Reinforcement Learning

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
An important problem in reinforcement learning is designing agents that learn to solve tasks safely in an environment. A common solution is to define either a penalty in the reward function or a cost to be minimised when reaching unsafe states. However, designing reward or cost functions is non-trivial and can increase with the complexity of the problem. To address this, we investigate the concept of a *Minmax* penalty, the smallest penalty for unsafe states that leads to safe optimal policies, regardless of task rewards. We derive an upper and lower bound on this penalty by considering both environment *diameter* and *controllability*. Additionally, we propose a simple algorithm for agents to estimate this penalty while learning task policies. Our experiments demonstrate the effectiveness of this approach in enabling agents to learn safe policies in high-dimensional continuous control environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the problem of RL penalty reward design for safe reaching tasks. The setting is without discounted factors. The authors derive the calculation for the minimax penalty, the smallest penalty for unsafe states that leads to safe optimal policies. The authors first show the upper and lower bounds for this minimax penalty term based on task solvability and diameter, as well as the minimum and maximum rewards, then give a model-free practical estimate for the minimax penalty term using value estimates. The validity of their approach is first tested on the LAVA GRIDWORLD environment. Then they conduct experiments on the Safety Gym environment, and show that their proposed method can work better (less failure rate or cumulative cost with longer episode length) than baselines such as constrained RL, Lagrangian methods, and SauteTRPO.

### Strengths
1. The paper is well-written. The way the authors introduce these concepts is like a textbook, where abundant figures and examples provide a walkthrough for the key concepts.
2. The proposed method has strong theoretical guarantees.
3. Strong empirical results achieved compared to baselines. The experiments are conducted over 10-20 random seeds, showing statistical significance.

### Weaknesses
1. Missing baseline: the work does not compare with epigraph-based methods [1].
2. The simulation environment is just in 2D space (though observation state is 60D): not sure how the proposed method will behave in 3D space or for manipulation tasks.
3. Minor issue in writing: some citation issue at L193-194 and L316-317.


References:
1. So, Oswin, and Chuchu Fan. "Solving stabilize-avoid optimal control via epigraph form and deep reinforcement learning." arXiv preprint arXiv:2305.14154 (2023).

### Questions
The paper doesn't describe the limitations of the proposed method (beyond its applicability only to environments with unsafe terminal states). It will be great if the authors can comment on this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work presents a minmax penalty within TRPO algorithm, which applies the smallest penalty for unsafe states to generate safe policies.

### Strengths
The proposed security reinforcement learning algorithm has certain theoretical significance.

### Weaknesses
1. It has not been compared with existing reachability methods, which adopt the idea of minimax optimization. 
2. The proposed method cannot guarantee absolute security of the strategy in theory or practice, which is crucial for secure reinforcement learning. 
3. The comparison algorithm is relatively outdated.

### Questions
1. If Theorem 1 follows from the convergence guarantee of policy evaluation (Sutton & Barto, 1998), what is its significance? 
2. What does Theorem 3 actually prove? 
3. Can the proposed framework be combined with other reinforcement learning methods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new paradigm for Safe RL, whose core argument is that traditional CMDP methods, such as those using cost functions and Lagrangian multipliers, are unnecessary. Instead, the authors posit that the safety problem can be reformulated as a reward design problem.
The authors theoretically define a "Minmax penalty", which if assigned to all unsafe terminal states, ensures that the optimal policy of any standard, reward-maximizing RL algorithm will automatically be safe.
The authors derive theoretical upper and lower bounds for Minmax penalty, which depend on the "Diameter" (D) and "Solvability" (C). As this theoretical bound is difficult to compute in practice, the authors further propose a simple and model-free practical algorithm. This algorithm adaptively learns a sufficiently large penalty value by estimating the bounds of the value function online. Experiments demonstrate that combining this algorithm with standard RL algorithms achieves strong safety performance on benchmarks like Safety Gym, outperforming traditional constrained-optimization methods.

### Strengths
1. Reframing the episodic safety problem from a complex "constrained optimization" framework (CMDPs) back to a "reward design" problem is an insightful perspective. 
2. The proposed practical algorithm is simple and easy to implement. It does not require manual tuning of hyperparameters. It can be used as a "plug-in" with any off-the-shelf, value-based RL algorithm, offering strong generality.
3. The method shows excellent performance in the Safety Gym experiments, particularly under high-noise settings.

### Weaknesses
1. The entire theoretical framework  is explicitly built on "undiscounted stochastic shortest path" (SSP) MDPs. However, the core experiments used to validate the algorithm  are conducted in "discounted," continuous-control, non-SSP environments. This makes the connection between the theoretical derivations and the experimental results weak.
2. The practical algorithm completely omits the solvability factor C from the theoretical bound. The authors claim the adaptive nature of the algorithm "implicitly" compensates for this, but this claim is not supported by any theory or ablation.
3. It is not at all clear how this method would generalize to the non-terminating CMDP setting. Figure 21 in the appendix seems to suggest the method's performance degrades in such a non-terminating setting.

### Questions
1. Given that the theory is for undiscounted SSPs, while the experiments are in discounted, continuous environments, can the authors provide deeper insight or a theoretical argument as to why Algorithm 1, which is derived from SSP theory, remains effective and robust in a discounted setting?
2.  Could the authors provide an ablation study to justify the omission of C? For example, in a simple tabular gridworld where C can be computed, how does Algorithm 1 (omitting C) compare to a policy trained using the "oracle" theoretical penalty that includes C?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents an alternative to traditional constrained policy optimization methods by learning a penalty term for states that violate constraints, resulting in the learning of safe optimal policies. The paper derives its penalty term from the concepts of diameter and solvability, which are explained in the paper. The derived penalty requires knowledge of the environment's dynamics; thus, the paper provides a practical algorithm that sidesteps this issue and presents some experiments demonstrating the performance of their algorithm. The paper also provides an analysis of performance in cases where the assumptions hold and in other cases where they do not.

### Strengths
The paper's strengths lie in providing an alternative to unstable policy optimization methods by introducing a penalty term that eliminates the need for such approaches. The paper is well-presented and generally sound, albeit with some flaws. The analysis in a lower-dimensional environment, as well as the comparison between the performance of the practical algorithm and the environment where the method's assumption holds, is quite helpful.

### Weaknesses
The paper derives its penalty term using the concepts of diameter and solvability, which require knowledge of the dynamics. In the practical implementation of their method, which does not require knowledge of the dynamics. The empirical experiments are on the weaker side. The method underperforms Lagrangian TRPO in task performance. Also, the paper compares their method only with a single threshold; further, the method does not compare their approach with the PID Lagrangian method, which is SOTA in constrained policy optimization. Further, there's a serious flaw in how the paper motivates its approach; the authors provide reasoning that their approach offers an alternative to shaped constraint costs. However, in many constrained RL problems, the cost is considered to be sparse. In my view, this approach provides an alternative to the issue of constrained policy optimization, which can be unstable. This is a significant distinction.

Minor errors: 
- "minimized when reaching"  in the abstract
- references in lines 193 and 215, 316

Stooke, Adam, Joshua Achiam, and Pieter Abbeel. "Responsive safety in reinforcement learning by pid lagrangian methods." International Conference on Machine Learning. PMLR, 2020.

### Questions
- Can you compare your approach to Lagrangian TPO under different constraint thresholds?
- Can you compare your algorithm to PID Lagrangian approaches?
- Can you run experiments on other safety gym domains?

I would be willing to raise my score if the authors can provide answers to my questions.

Stooke, Adam, Joshua Achiam, and Pieter Abbeel. "Responsive safety in reinforcement learning by pid lagrangian methods." International Conference on Machine Learning. PMLR, 2020.

### Soundness
2

### Presentation
3

### Contribution
2
