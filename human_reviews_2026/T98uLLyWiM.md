# Adapting Reinforcement Learning for Path Planning in Constrained Parking Scenarios

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Real-time path planning in constrained environments remains a fundamental challenge for autonomous systems. Traditional classical planners, while effective under perfect perception assumptions, are often sensitive to real-world perception constraints and rely on online search procedures that incur high computational costs. In complex surroundings, this renders real-time deployment prohibitive. To overcome these limitations, we introduce a Deep Reinforcement Learning (DRL) framework for real-time path planning in parking scenarios. In particular, we focus on challenging scenes with tight spaces that require a high number of reversal maneuvers and adjustments. Unlike classical planners, our solution does not require ideal and structured perception, and in principle, could avoid the need for additional modules such as localization and tracking, potentially resulting in a simpler and more practical implementation. Also, at test time, the policy generates actions through a single forward pass at each step, which is lightweight enough for real-time deployment. The task is formulated as a sequential decision-making problem grounded in a bicycle model dynamics, enabling the agent to directly learn navigation policies that respect vehicle kinematics and environmental constraints in the closed-loop setting. A new benchmark is developed to support both training and evaluation, capturing diverse and challenging scenarios. Our approach achieves state-of-the-art success rates and efficiency, surpassing classical planner baselines by +96\% in success rate and +52\% in efficiency. Furthermore, we release our benchmark as an open-source resource for the community to foster future research in autonomous systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a reinforcement learning-based method for parking path planning using a bicycle kinematic model, demonstrating excellent performance in narrow and complex environments. 
To address the challenges of long horizons, high-precision control, and sparse rewards, the approach introduces two key techniques: curriculum learning, which gradually progresses from simple to complex scenarios to improve training stability; and action chunking, which groups multiple primitive actions into macro-actions to balance exploration efficiency and maneuver precision. For fair comparison, the method uses the same inputs as Hybrid A: ego and target poses, and obstacle contours. A "rollout-back" mechanism is designed to generate feasible initial states, resolving infeasibility caused by sparse representation. Inputs are transformed into the ego frame, normalized, and constrained by a limited perception range to simulate real sensor limitations; a cross-attention mechanism enables the agent to focus on critical obstacles. 
Experiments are conducted on the authors' self-constructed ParkBench benchmark, showing that the proposed method significantly outperforms Hybrid A in success rate, planning time, and travel distance. Ablation studies validate the effectiveness of curriculum learning and action chunking, while attention visualizations demonstrate strong environmental understanding. Despite degraded performance in open spaces and reliance on manually designed curricula, this work highlights the great potential of RL for generating human-like trajectories in complex parking tasks.

### Strengths
1.	The ParkBench is comprehensively designed and highly representative of real-world parking scenarios, making a significant contribution to the evaluation of parking trajectory planning methods. We look forward to its future expansion to cover even more diverse maneuvers.
2.	The paper effectively addresses a practical engineering challenge—initial pose infeasibility caused by sparse obstacle representation—through a well-designed roll-out mechanism that generates valid starting states, thereby improving training stability and realism.
3.	The integration of curriculum learning and action chunking significantly enhances both success rate and computational efficiency, demonstrating clear advantages over classical planning approaches in complex, constrained environments.

### Weaknesses
1.	The current curriculum is manually designed specifically for rear-in parking, which limits its transferability to other maneuvers such as parallel or angle parking. To enhance generalization and scalability, the authors are encouraged to explore automated curriculum learning methods that can adaptively generate training tasks across diverse scenarios.
2.	While the method performs exceptionally well in narrow and complex environments, its performance degrades in open or sparsely constrained spaces. Given that the current dataset primarily focuses on challenging narrow scenarios, it would be valuable to include more diverse environments—especially open layouts—in both training and evaluation, along with a detailed analysis of the model’s behavior across different scene types.
3.	Although the limited perception range simulates partial observability under real-world sensing constraints, all experiments are conducted in static environments. For a more realistic assessment, future work should evaluate the method’s robustness in dynamic settings involving moving obstacles such as pedestrians and vehicles, which are common in practical parking scenarios.

### Questions
1.	In Table 1, the line of"PPO (Ours) CL✓ Chunking✗" shows a significantly higher number of pivot points (53.4) compared to Hybrid A (3.2). Does this indicate that, in the absence of action chunking, the policy tends to generate excessive and unnecessary forward-backward transitions? It would be helpful to provide visualizations of such trajectories and analyze the underlying causes of this oscillatory behavior—whether it stems from poor temporal credit assignment, suboptimal exploration, or instability in policy learning.
2.	The 51 scenarios in ParkBench are extracted from real-world datasets. Could the authors provide more details on the distribution of these scenarios? Specifically, how many correspond to perpendicular parking, parallel parking, or other configurations? A breakdown of scene types would help assess the benchmark's diversity and representativeness and clarify the scope of the method’s evaluation.

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
5

### Summary
This paper employs a RL–based approach to replace the classical planner in path planning tasks. It empirically demonstrates the effectiveness of the proposed method in parking scenarios.

### Strengths
1. The paper shows the effectiveness of using the PPO algorithm as a planner in parking scenarios.

2. It introduces a new benchmark, ParkBench, to facilitate research on path planning in parking environments.

### Weaknesses
1. In the introduction, the authors claim that RL-based methods, as representatives of closed-loop approaches, remain underexplored in path planning. However, RL-based planners have been extensively studied in various path planning tasks that consider practical constraints across diverse real-world applications, including transportation, warehousing, and surgical robotics.

2. Numerous studies have focused on developing RL-based planners in related domains. Although this paper centers on parking scenarios, this domain is closely connected to transportation and autonomous driving. The related work section does not sufficiently review these prior studies.

3. In the empirical evaluation, the proposed method is compared only with one classical heuristic baseline. The authors do not include adequate baseline methods, especially those that also utilize RL techniques. As a result, the experimental results do not convincingly demonstrate the contribution of this work.

4. The classical heuristic method used as the baseline has shown strong performance and generalization across other scenarios. To further validate the proposed method’s effectiveness and generalization, it would be beneficial to evaluate it on additional tasks.

5. In the experimental section, the authors integrate only the PPO algorithm into the parking planner. A comparison with other standard RL algorithms would provide a more comprehensive understanding of the framework’s robustness and performance.

6. The contribution of this paper is limited. It primarily applies an existing RL algorithm to a parking task, which is not a novel direction for the community, although the development of the ParkBench benchmark and consideration of practical constraints are appreciated.

### Questions
1. Since this work demonstrates the practicality of an RL-based planner trained on ParkBench, is it possible to deploy this method in a real vehicle, similar to how the A* algorithm has been applied?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a new reinforcement learning based path planning system for parking problem in cluttered environments. The paper also introduces ParkBench, a new parking benchmark using bicycle model dynamic for simulating the environment with a mobile robot. The method demonstrates more than 96% success and 52% efficiency improvement compared to classical path planner.

### Strengths
* Strong empirical results: Achieves 92.2% success on ParkBench vs. 47.1% for Hybrid A*, and 2× improvement in time efficiency.
* Benchmark contribution: ParkBench fills a benchmark gap in parking evaluation, providing 51 realistic layouts for reproducibility and comparison.

### Weaknesses
* The RL system presented in this paper is fairly straight forward. The component includes a handcrafted curriculum for initial configuration and  motion primitive (action chunking). These components are well-established in the literature and the author does not demonstrate sufficient effort in integrating these components as a whole system.

* There exists a lot of RL-based motion planning for mobile robot, many of them are trained in high-fidelity simulator, such as Gazebo and IsaacLab. The author does not fully address these prior work to demonstrate the novelty of the system.
Xu, Zifan, et al. "Benchmarking reinforcement learning techniques for autonomous navigation." arXiv preprint arXiv:2210.04839 (2022).
Akmandor, Neşet Ünver, et al. "Deep reinforcement learning based robot navigation in dynamic environments using occupancy values of motion primitives." 2022 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2022.

* Writing tone: Some claims (e.g., “potentially eliminates need for localization and tracking”) are speculative and should be framed more cautiously. The use of language is not precise and formal for a research paper.

### Questions
Please see the weakness part

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present an approach for real-time path planning in tight spaces with Deep Reinforcement Learning. The paper shows how the approach outperforms an A* based baseline by a large margin. In addition to the paper they provide a benchmark for parking in tight spaces based on 2d coordinates, like lidar points showing the surroundings.

### Strengths
The performance compared to the baseline is quite impressive. 

The paper also contributes a benchmark.

Including a bicycle model introduces an easily tunable component, if more complex vehicle models are needed, and that should ensure feasible trajectories.

The benchmark provides very simple lidar scans which may seem like a disadvantage but is an easily transferable representation that should be easy to provide in many contexts. It is also very easy to simulate as the paper shows.

The approach, and architecture, seems surprisingly simple.

### Weaknesses
While having impressive results, they are on a novel self-created benchmark which did not give other authors opportunity to optimize on it. Therefore, results have to be taken with a grain of salt and if the benchmark is accepted in the domain time must tell how these results hold up.

It seems more direction changes are needed, compared to a simple A* algorithm. While other metrics indicate better performance it would be interesting how this scales, i.e. how many more pivot points, which should come with a more complex trajectory, are needed for what performance boost?

The approach only compares to Hybrid A* and no other approach. It would have been possible to compare with other non-deep learning methods like a Reeds-Shepp Curve planner or Dijkstra. 

Finally, this is a very good solution for this task but I am not completely convinced this is the right venue for it. Maybe IROS, IV or ITSC would have been more suitable. I leave that decision to the AC.

### Questions
Given the many penalties which are not primary goals, such as the goal achievement and collision, what are local minimas that were observed during training. In other words, were there "cheating" behaviors where the model optimized not getting penalized for being idle e.g. by moving very slow. 

Please note that concurrent work on the horizon should be compared with this work in the future. E.g. "RAFT: Regularized Adversarial Fine-Tuning to Enhance Deep Reinforcement Learning for Self-Parking
, Pighetti et al.". This was published in August 2025 so it is irrelevant for judging this work.

### Soundness
3

### Presentation
3

### Contribution
3
