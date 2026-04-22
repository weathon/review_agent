# RTG: Reverse Trajectory Generation for Reinforcement Learning Under Sparse Reward

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Deep Reinforcement Learning (DRL) under sparse reward conditions remains a long-standing challenge in robotic learning. In such settings, extensive exploration is often required before meaningful reward signals can guide the propagation of state-value functions. Prior approaches typically rely on offline demonstration data or carefully crafted curriculum learning strategies to improve exploration efficiency. In contrast, we propose a novel method tailored to rigid body manipulation tasks that addresses sparse reward without the need for guidance data or curriculum design. Leveraging recent advances in differentiable rigid body dynamics and trajectory optimization, we introduce the Reverse Rigid-Body Simulator (RRBS), a system capable of generating simulation trajectories that terminate at a user-specified goal configuration. This reverse simulation is formulated as a trajectory optimization problem constrained by differentiable physical dynamics. RRBS enables the generation of physically plausible trajectories with known goal states, providing informative guidance for conventional RL in sparse reward environments. Leveraging this, we present Reverse Trajectory Generation (RTG), a method that integrates RRBS with a beam search algorithm to produce reverse trajectories, which augment the replay buffer of off-policy RL algorithms like DDQN to solve the sparse reward problem. We evaluate RTG across various rigid body manipulation tasks, including sorting, gathering, and articulated object manipulation. Experiments show that RTG significantly outperforms vanilla DRL and improved sampling strategies like Hindsight Experience Replay (HER) and Reverse Curriculum Generation (RCG). Specifically, RTG is the only method that can solve each task with success rates of over 70% within given compute budget.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose Reverse Trajectory Generation (RTG) to address the sparse reward problem in the field of reinforcement learning (RL). Low sampling efficiency is indeed prevalent in complex RL tasks; to mitigate this, the authors introduce a reverse transition function to generate more beneficial training data, thereby accelerating the convergence of the RL policy. The authors’ idea is quite insightful, and their method achieves significant performance improvements compared to Hindsight Experience Replay (HER) and Reverse Curriculum Generation (RCG).

### Strengths
1.The proposed Reverse Trajectory Generation (RTG) method integrates the advantages of HER and RCG. It not only supports multi-objective goal setting but also avoids the complexity of curriculum design.

2.The authors elaborate on the specific implementation pathway of RTG: a Reverse Rigid-Body Simulator (RRBS) is used to construct an optimization problem for inverse solution, and methods such as forward replay are proposed to optimize the generated reverse data.

3.The proposed method demonstrates significantly better performance than HER and RCG across three tasks.

### Weaknesses
1.The authors' title is limited to RL, yet their proposed method relies on the Reverse Rigid-Body Simulator (RRBS) and manipulation tasks. The authors should clearly define the scope of the paper: if the method is only targeted at rigid-body manipulation tasks, this specificity should be reflected in the title; conversely, if a broader applicability is claimed, the authors must provide a more generalizable method along with corresponding validation.

2.The baselines selected by the authors are insufficient, as comparisons are only made with the classical HER and RCG. In fact, numerous variants of HER and RCG have been proposed in existing literature, and the authors should consider incorporating more recent and representative baselines to strengthen the comparative analysis.

3.The current simulation setup is overly idealized. More complex task configurations should be considered, such as validation with a physical robotic arm. Even a simple scenario tested on real hardware would significantly enhance the persuasiveness of the method’s evaluation.

### Questions
Similar to Weakness 3, Sim2Real transfer is a classic challenge in RL. I am curious whether the reversely synthesized data can enhance the performance of the method in real-world deployment. If the authors can provide validation for this aspect, I would consider increasing my review score.

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
The paper proposes a reverse-trajectory pipeline for sparse-reward manipulation: optimize short reverse-time trajectories from the goal, use a small beam search to expand candidates, then “forward replay” them to ensure feasibility and add them into the replay buffer alongside online data. Evaluated on several 2-D pushing/arrangement tasks with value-based agents, the method reports faster success than standard off-policy RL and simplified HER/RCG variants.

### Strengths
• Clear idea: generate from the goal backward, then verify forward and learn from those transitions.  
• Nice engineering: reverse trajectory optimization plus a lightweight search is practical.  
• Positioning: bridges goal-based data augmentation (HER-style) and reset/backplay ideas without heavy curriculum design, which are easy to follow.

### Weaknesses
• Scope is narrow. The method is shown only in 2-D with discrete pushing primitives. The proposed approach aims to be general, but the paper does not show 3-D or continuous control; the authors should add at least one 3-D or continuous-torque benchmark. 
• Baselines feel light. The comparison emphasizes simplified HER/RCG. The proposed method looks strong, but the paper does not show standard UVFA-HER or backplay/reset-to-state; the authors should include those or justify why they are inapplicable.  
• Compute transparency. Reverse optimization and beam search add overhead. The method claims efficiency, but the paper does not show wall-clock, solver iterations, or nodes-expanded; the authors should report runtime vs. success and sensitivity to beam width and horizon.  
• Distribution shift. Mixing reverse and online data can bias Q estimates. The approach is plausible, but the paper does not show stability or calibration; the authors should sweep mixing ratios and add simple conservatism checks (e.g., CQL-style diagnostics).  
• Generality beyond DQN/DDQN. The method is claimed to be broad, but the paper does not show actor-critic or model-based planners, such as MuZero.

### Questions
• How does success vary with beam width and horizon? Please show success vs. compute (nodes, solve time, iterations).  
• How much do forward replays alter reverse-optimized states? Quantify feasibility gaps before/after replay and their effect on learning.  

Another fundamental issue is: for some tasks, the entropy will increase along the traj but shrink by reversing the trajs. How it is possible to generate the reverse traj? Taking an example of unplug the socket, the reverse traj are much much complicated and hard than original one. And others like articulations. I am doubting the generalization of such a method on real problems.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces two methods: 1) Reverse Rigid-Body Simulator (RRBS), and approach which leverage differentiability to generate short-range inverse models and b) Reverse Trajectory Generation (RTG), which uses RRBS to provide synthetic trajectories which reach desired goals which are then used for off-policy RL. Conceptually, the approach blends hindsight experience replay with reverse curriculum generation on the policy optimization side, while using differentiable inverse models to avoid reasoning over long sequences of forwards actions. The approach is demonstrated on several 2-D table top manipulation tasks.

### Strengths
$\textbf{Elegant approach}$: The approach is elegant, simple, and well motivated. The approach for exploring in trajectory space rather than forward action space is rather intriguing, and potentially a powerful general insight. 

$\textbf{Clarity}$: The paper is well written and easy to understand. The paper effectively uses concrete examples to illustrate the main ideas behind the paper. 

$\textbf{Performance on benchmarks}$: On the benchmarks presented, the paper demonstrates a clear performance boost over baselines.

### Weaknesses
$\textbf{Scalability}$: The simple 2-D test domains presented in the paper do not effectively demonstrate whether the approach is competitive with baselines in more realistic, high dimensional settings. Despite the elegance of the approach, the results are substantially below the bar for acceptance at ICLR. I can’t argue for acceptance without seeing improvements over baselines on more standard robotics benchmarks. 

$\textbf{Specificity of solver}$: Even though the high-level idea is quite general, many of the details of the approach are tied to the specific physics solver used in the paper. Thus, it seems difficult for the community easily pick up the algorithm and use it off the shelf in different settings.  I strongly suggest the authors rework the algorithm in a more general, widely used differentiable physics engine.

### Questions
- Can the methods scale to standard high-dimensional robotics baselines? 

- Is the differentiable inverse dynamics approach easily adaptable to other simulators? 

- Why beam search verse other search heuristics? How would beam search scale to more high dimensional spaces?

### Soundness
3

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
3

### Summary
Exploration remains a key challenge when using reinforcement learning (RL) for sparse reward tasks, and many research papers have sought to address this problem from different angles: off-policy learning, goal relabeling (hindsight experience replay; HER), demonstrations, skill priors, as well as techniques that create training curricula by directly manipulating the initial state distribution of e.g. a simulation environment (reverse curriculum generation; RCG). This paper identifies key limitations of HER and RCG (limited use cases and reliance on human-designed curricula, respectively), and proposes reverse trajectory generation (RTG), a method that addresses these limitations by instead generating goal-reaching trajectories via optimization over a reversed transition function that models reverse-time environment dynamics (this work relies on a reverse rigid-body simulator in practice). Experiments are conducted on three simple 2D manipulation tasks with discrete actions, and empirical results indicate that the proposed method is significantly more data-efficiency than RCG and a slightly improved version of HER that the authors denote as implicit HER.

### Strengths
My initial assessment of this paper leans positive overall; it is very well written and technical contributions appear original and sound Specifically:
- I believe that this paper studies a relevant and timely problem (solving sparse reward RL problems in simulation), and is likely to be of interest to the community. The problem is clearly motivated, and shortcomings of existing work (HER and RCG in particular) are described in the introduction + related work sections. The paper is very well written, self-contained, and easy to follow; I believe that readers will be able to appreciate the technical contributions without much background knowledge. The illustration in Figure 1 is helpful for understanding the limitations of prior work.
- The proposed method seems technically sound and appears to rely fairly little on engineering or manual labor (compared to RCG) which I definitely consider a plus. It does rely on the assumption of a reliable (to the extent that the forward replay trick will work at least some of the time) reverse transition function, which I believe is not currently available in most robotics simulators, but it is interesting to see what is possible when you do have such privilege.
- The experiments seem well thought out and successfully demonstrate the advantage of the proposed method vs. HER and RCG on simple 2D manipulation tasks. The baselines seem reasonable.

### Weaknesses
While my overall assessment leans positive, I do have some reservations that I would like the authors to address:
- It is not clear to me how this method would be extended to continuous control tasks and/or discrete control tasks with a larger number of actions. Since the paper chooses to explicitly focus on robotics as application area (for which simulation is often used) which typically has a continuous action space and no reverse transition function readily available, I do believe that at the very least an attempt should be made at bridging this gap. For example, maybe an approximate reverse transition function could be obtained via sampling, and/or additional experiments on a simple 2D physics problem with continuous actions could be added.
- I am a bit skeptical about the BC and CQL results; can the authors please explain why there is such a big performance gap between these methods and the proposed method? If these methods have access to offline data that sufficiently cover the initial state distribution used during evaluation then I would expect it to be rather successful.
- This is relatively minor, but I would appreciate it if the authors can make the font size larger in their experimental result figures on page 9; I had to zoom in quite a bit in order to read the legend and axis labels.

### Questions
I would really appreciate it if the authors can address my comments in the "weaknesses" section above using written arguments and potentially additional experimental results. My main concerns / questions pertain to limitations, experimental setup, and baseline results.

### Soundness
2

### Presentation
3

### Contribution
3
