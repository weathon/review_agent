# Generalizable Multilevel Graph Optimization via Reinforcement-Guided Diffusion from Simple Subproblems

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Solving multilevel graph combinatorial problems (GCPs) is challenging due to their structural complexity and the limited generalization capabilities of existing learning-based optimization algorithms. Models trained on GCPs often fail to generalize to more complex or larger multilevel problems. We propose a Reinforcement Learning-Guided Diffusion Model (RLG-DM) that addresses this challenge by combining structural priors learned from simple problems. In the forward diffusion process, noise is progressively injected into the structure of a graph neural network, each representing and having been trained on a simple combinatorial optimization problem, such as the facility location or vehicle routing problem. In the reverse diffusion phase, a reinforcement learning controller guides the stepwise generation of subgraphs from a randomly initialized graph by dynamically selecting and combining the pretrained diffusion models based on task-specific hierarchies. Trained only on representative subproblems, the controller generalizes to unseen multilevel GCP tasks without retraining. We evaluate the proposed RLG-DM on representative multilevel GCPs, such as the location routing problem, the nurse rostering problem and \textcolor{red}{flexible job shop scheduling}. Experimental results show that RLG-DM consistently outperforms state-of-the-art baselines and generalizes effectively to structurally diverse, unseen tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a graph diffusion method, coupled with RL that is designed to create a generalizable solution to graph combinatorial problems. The main innovations are a forward diffusion process, in which noise is injected to maintain consistency with sub-problems, such as graph coloring, TSP, VRP, etc. In the reverse diffusion step, RL is employed to guide denoising, with rewards being aggregated across problem layers. The proposed method is demonstrated on location routing and nurse scheduling problems.

### Strengths
This method aims at providing a principled approach to injecting structural knowledge into the diffusion process, in the form of simpler sub-problems in forward diffusion. The idea makes intuitive sense, and it seems to bear out nice results. A diffusion process can leverage structural information from other problems to make it more useful.

The results appear promising—RLG-DM performs as well or better than many comparators in both problem sets.

### Weaknesses
The organization and writing could be improved. For example, without the appendix it would be very difficult to follow the RL process used in reverse diffusion. Without 7.1-7.4 in the appendix, I don’t think I would have understood it. I recommend moving some of that material into the main body of the text.

The results look good for RLG-DM, but it seems like a somewhat minor difference compared to some comparators. It’s hard to make sense of the performance gaps overall. For example, in instance 10, all of the methods seem to perform comparably. Is there some way to summarize these tables to account for differences in problem instances? Or perhaps to present it visually to make it easier to grasp.

Some minor errors:
- Line 161 GCP is used for graph coloring, but has already been defined for graph combinatorial problem
- Line 232 – “an creased”
- Line 272 seems to be a run-on sentence

### Questions
How sensitive is the problem to the choice of sub-GCPs? It seems likely that it is very sensitive, as the authors note in the limitations. What happens if there is a mismatch between sub-GCPs and the ML-GCP that you wish to solve?

Are sub-GCPs used sequentially during forward diffusion, like a curriculum? Or are they used simultaneously? It seems that (per line 269) one encoder is trained per sub-GCP. Does that mean one needs to know all of the relevant sub-GCPs for their eventual problem at training time?

How is the relative importance in (16) trained?

How are the problem instances selected? In Table 2, the instances are from 10-21, and I’m curious why that is the case.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduce a novel method to combine graph diffusion and reinforcement learning to solve multilevel graph optimization problem. The algorithm first trains GNN that predict the optimal solution along a guided forward diffusion process. Then, it trains a RL model to guide denoising process, where the state representation is computed via the GNN trained in the forward diffusion process. The forward process is trained with simpler sub-problem and the reverse RL process is trained on the actualy complex problem, where the Q-value is a composition of the subproblem. The deconstruction and concur enhances the model's ability in solve the original combinatorial problem. Experiment results show consistent improvement.

### Strengths
- The idea of sub-tasking and composition is very interesting, and it is well tuned for the specific graph optimization problem.

- The adaptive scheduling is innovative. It stabilize the training and creates more robust GNN.

- The experimental results show consistent improvement.

### Weaknesses
- This is a good and interesting paper, but is difficult to read. To understand the algorithm, one must go back and forth between the main text and the algorithm in the appendix.

- Many things are underspecified
    - How is the importance factor actually learned?
    - How the action is done, and how the action impact the state and graph.
    - I am assuming the Q-learning is based on DQN/Q-network, but this is never specified in the paper.

- Some notification does not make sense to me. In algorithm 1, it seems like $X$ and $\textbf{h}$ are different things, but in equation 6 $h_{t-1}$ seem to be passed in as $X_{t-1}$, with an additional noise? I don't think this is a minor notation problem, rather it impact audience's understanding of the algorithm.

- There is no code accompanying this submission, and it difficult to check the missed details in the paper.

- Following the points above, some description are also misleading to me. In line 232, what do you mean by decreasing? Do you mean a more negaive IG or decreasing IG over time? 

- The model has many components, and yet the author did not provide a principle in understanding how these components synergize. For example, (I think this is **critical**), the GNN is trained on a limited set of simpler problems, without structure change. However, the action in MDP involves modifying graph structure. There is a large discrepency between graph trained on fixed set of structure and a dynamically explored state space. The author did not provide justification of how this work. Some co-evolve system seem necessary for this work properly.

- The inference is conducted on few instances, leaving room for selection bias (in terms of samples, hyperparameter, and design selection).

### Questions
See above.

### Soundness
3

### Presentation
2

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
The paper proposes RLG-DM, a framework that embeds a reinforcement learning controller into reverse diffusion to guide solution generation for graph combinatorial problems. It models complex tasks as multilevel problems, defines level specific states and Q values for each level, and aggregates them into a single joint Q to pick actions during denoising. The forward stage learns transferable structural priors by pretraining frozen GNN encoders on simpler subproblems, and it introduces an information gain based noise scheduler with linear, cosine, and exponential variants.

### Strengths
1. Embeds a reinforcement learning controller into reverse diffusion to actively steer sampling toward feasibility, cost, and constraint satisfaction, rather than passively denoising. 

2. Multilevel formulation with coordinated decisions: Defines a multilevel state, learns per-level Q-values, and aggregates them into a joint Q for a single action rule. This directly tackles cross-level dependencies that standard flat GNN or diffusion baselines do not handle. 


3. Information-gain noise scheduling: Introduces an information-gain based scheduler for the forward diffusion with concrete linear, cosine, and exponential schedules to balance structural preservation and exploration.

### Weaknesses
1. No hard feasibility guarantee: The method uses an RL controller during reverse diffusion to guide generation toward feasibility and constraint satisfaction, but it does not provide a projection, masking, or proof that every sampled solution satisfies combinatorial constraints. This is a major problem with the method. GCPs many times have combinatorial constraints. What is the utility of the solution if it might be infeasible? (e.g., a coloring solution that violates that two neighbors cannot have the same color)
See, e.g., "Graph-based Deterministic Policy Gradient for Repetitive Combinatorial Optimization Problems" among prior ICLR papers that can guarantee feasibility of the learned solution. 

2. Admitted scalability and generalization limits: The Limitations section states that the current innovation lies in generalizing from simple to more complex GCPs rather than scaling to larger or more complex instances, and that performance depends on selecting appropriate simpler GCPs for training. This constrains flexibility and highlights remaining challenges in broader settings. 

3. Triple nested control loop may be costly: Inference iterates over diffusion time, over levels, and over subproblems at each level, with per-step Q computations and action selection. The paper does not analyze the computational overhead of this nested control relative to baselines. 

4. Limited evidence across problem families: Results are reported for two multilevel tasks, location routing and nurse rostering. The paper claims state-of-the-art improvements there, but broader validation on additional multilevel GCPs would strengthen the generality claim.

### Questions
See weaknesses, especially points 1, 3, and 4.

### Soundness
3

### Presentation
3

### Contribution
2
