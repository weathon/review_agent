# Bandit Learning for Online Scheduling with Immediate Decision

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Online scheduling has been extensively studied in computer science and economics owing to its broad applications. Motivated by streaming task processing in domains such as IoT data streaming and cloud resource allocation, we investigate an online scheduling setting where the scheduler must immediately decide whether to accept an incoming task. Consider a system with $M$ identical machines. At each time step, multiple tasks arrive, and each machine must immediately assign itself to a task or remain idle. Tasks that are not processed immediately are abandoned and cannot be revisited. Upon completion, a task yields a reward, which may be stochastic and initially unknown. Through repeated task completions, the scheduler can learn the reward distributions over time. In this work, we formalize this problem as online scheduling with immediate decision. We first analyze the setting with known rewards, for which we derive a worst-case competitive ratio and propose a near-optimal online algorithm. For the case of unknown and random rewards, we design an efficient bandit algorithm that balances exploration and exploitation, achieving an $O(\log T)$ regret over a time horizon $T$. Experimental results demonstrate the efficacy of the proposed algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studied the online scheduling with immediate decision problem where the tasks arrive streaming and machines need to decide to complete the task or not immediately without a second chance. For this problem, the authors consider two distinct settings, i.e., with or without knowing the reward. For the two cases, algorithms with theoretical guarantees are proposed.

### Strengths
1.	This paper proposes a new and interesting research problem.

2.	Strong theoretical results with approximation bounds are derived.

### Weaknesses
1.	The paper mentions several real-world applications as instances of the problem. However, the underlying relevance and motivation of this problem are not strong for me. Please further elaborate for clarity.

2.	In the setting in Section 3, the problem considers multiple machines. However, in the description, only one machine is considered. I wonder if a task is discarded by machine A, if it will be considered by another machine B? Do those machines follow the same strategies on decision by sharing information, or do they make independent decisions? If those task rewards follow the same distribution? Please clarify.

3.	Several minor issues: i) this paper is math-heavy, while some notations are not explained before use; ii) the paper is not well formatted, e.g., some formulas are out of scope.

### Questions
Please refer to my weakness comments.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies an online scheduling problem with immediate decision-making and uncertain rewards, motivated by applications like IoT and cloud computing. The authors first analyze the known-reward setting, establish a worst-case competitive ratio, and propose a near-optimal algorithm (MRDF). Then, for the unknown reward case, they introduce a bandit-based algorithm (S-UCB) that achieves $O(\log T)$ regret and asymptotically matches the competitive ratio of MRDF. Experiments are provided to validate the theoretical results.

### Strengths
1. The problem setup is clearly described. The paper is well-structured and easy to follow. 

2. The analysis of the deterministic MRDF algorithm is mathematically clean, and the idea of partitioning machines by task-length ranges is intuitively interesting. 

3. The analysis in Section 4 is the strongest part of the paper. Deriving the worst-case competitive ratio (Theorem 4.1) and then proposing an algorithm (MRDF) with a matching bound (Theorem 4.2) is a nice, complete theoretical result.

### Weaknesses
1. The main theoretical contributions feel incremental and can be viewed as extensions of prior work in bandit scheduling. The main technical novelty lies in the analysis of the preemptable property of each machine. The MRDF algorithm is essentially a length-partitioned greedy heuristic that heavily builds on classical scheduling principles such as maximum-density-first, while the S-UCB algorithm is just a direct plug-in of the classic UCB index into MRDF.  The preemption mechanism in the bandit scheduling is interesting but not fundamentally new.

2. I think the problem is also related to admission control with reusable resources (e.g., [1, 2]), and the authors should compare their work with this line of literature.

3. The regret analysis in Section 5 closely mirrors textbook UCB arguments, with a simple type-based reward model and no real coupling between machines. The dependence on $O(\log T /\Delta)$  is straightforward from a standard gap-based decomposition, so the result is predictable.

[1] The online knapsack problem with departures, ACM SIGMETRICS 2023.

[2] Dynamic Care Unit Placements Under Unknown Demand with Learning, Manufacturing & Service Operations Management 2025.

### Questions
1. Can the authors explain intuitively what drives the $O(1/M L^{1/M}_{\max})$ scaling? Is it tight for both preemptive and non-preemptive cases?

2. In Algorithm 2 (S-UCB), the updates only occur when a task is completed, meaning many preempted tasks give no feedback. Would this slow convergence? Is there any mechanism to exploit partial observations before completion?

3. The regret bound in Theorem 5.1 has an $O(N^2)$ dependency. This seems a bit high and isn't discussed much. Is this an artifact of the proof technique or do you believe it's a fundamental lower bound for this problem?

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
This paper investigates the online scheduling problem where tasks arrive sequentially and require immediate acceptance or rejection. The online scheduling runs on $M$ identical machines and allows preemption, where preempted tasks will never be revisited again. The authors introduce a Maximum Remaining Density First (MRDF) algorithm for known rewards and a Scheduling Upper Confidence Bound (S-UCB) algorithm for unknown rewards, leveraging bandit learning to balance exploration and exploitation. Theoretical analyses establish a competitive ratio of $\tilde{O} (1/ML_{max}^{1/M})$ for known rewards and an $O(\log T/\Delta)$ regret bound for unknown rewards. Experiments on synthetic data validate the algorithms' efficiency and convergence.

### Strengths
The paper introduces a well-defined and practically motivated model for online scheduling with immediate decision-making. The authors provide a rigorous theoretical analysis for both known and unknown reward settings. The derivation of the competitive ratio bound of $\tilde{O} (1/ML_{max}^{1/M})$ for the deterministic setting and the subsequent design of the near-optimal MRDF algorithm are substantial. Furthermore, the extension to the stochastic setting with the S-UCB algorithm achieves a sublinear regret of $O(\log T/\Delta)$.

### Weaknesses
- The model proposed in the paper requires immediate decision-making. If I understand correctly, this setting is a simplified version when the tasks are allowed to be queued and the carryover effect is considered. It is not easy to see the novelty and practicality of the proposed model.

-  The upper bound is gap-dependent and is less meaningful when $\Delta$ is small. It would be interesting to see if $\Delta$ can be eliminated in the theoretical result and a worst-case/gap-independent result can be established.

- The lower bound in Theorem 4.1 is a bit confusing. It is for deterministic algorithms and seems to be improved with a more sophisticated algorithm (e.g., randomized algorithms).   

- The experimental section lacks diversity in datasets, as it primarily uses synthetic data with uniform distributions, failing to demonstrate robustness under more complex, real-world scenarios like non-stationary task flows or heterogeneous environments. The experiment did not cover algorithms from other works, such as non-preemptive methods or other modified algorithms to adapt to the settings in the paper.

### Questions
Please see the weaknesses above.

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
This work deals with an online scheduling problem where the learner (scheduling) can learn the expected reward of each task type and use that information to make real time scheduling choices. They first analyse the known reward version and design a MRDF online scheduler that is nearly optimal wrt competitive ratio. They then extend this to unknown reward case by using a UCB style algorithm  so that the scheduler can balance exploration of task types with exploitation. They showcase the performance of the algorithm with regret bound theoretical guarantees. The work is complemented by synthetic experiments to illustrate that the proposed scheduler outperforms baseline in respective scenarios.

### Strengths
Online Scheduling and task allocation is an interesting area that has many real world application and this problem deals with online scheduling in a bandit setting style so the system can dynamically decide and learn at the same time.

The analysis wrt competitive ratio gives a view of difficulty of this problem setting and shows that the proposed algorithm performs closer to the lower bound. 

The experiments even though they are synthetic seems to align well with the theory to demonstrate that the learner can  outperform baselines in online scheduling complementing the sublinear regret in unknown reward scenario.

### Weaknesses
The problem setting doesn't have the option of buffer, which is important in many real systems so having at least a tiny pending queue will help facilitate many real world scenarios. 

The problem setting has a strong assumption with respect to preemption meaning the task suffers permanent rejection and it can also never be resumed. Also, the problem relies heavily on correct type labels which is hard to estimate or determine in many scenarios. 

Also, the regret measure doesn't involve learner competing against best offline policy rather against an online style algorithm with known rewards. This doesn't quantify the algorithm in classical regret perspective. 

*also refer questions

### Questions
The performance measure is competitive ratio against an online benchmark that knows rewards. Why was regret against the best offline policy with known rewards not considered ?

The problem setting handles the concept of discarding or preempt the task on the arriving task $k$. Does this problem setting facilitate a generalization of providing the concept of buffer to handle the stream of tasks i.e. discarding the task can be treated as relocation to the buffer machines. In this case, $|M|$ grows by a constant factor to account for buffer machines. This could help with generalizing to numerous real world application. Does that break underlying assumption of the problem framework or any theoretical guarantee ?   

The experiments are shown on synthetic arrival processes. However in many real world it tasks arrive either in a bursty fashion (bursty arrival of long jobs followed by bursty arrival of short jobs). How does the algorithm perform in this scenario, does the algorithm over explore or over commit in those phases ?

The regret seem to grow with $N$ which seems to stem from the exploration term, Can having a elimination style algorithm or Thompson sampling algorithm reduce the dependence of $N$ in regret bound ?

### Soundness
3

### Presentation
2

### Contribution
3
