# Deformable Linear Object Manipulations with Differentiable Physics

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
We address the challenge of enabling robots to manipulate deformable linear objects (DLOs), such as wires, ropes, and rubber bands. Prior work in this domain has primarily focused on narrow, task-specific problems, often relying on real-world demonstrations or handcrafted heuristics. Such approaches, however, do not scale to the diverse range of materials and tasks encountered in practice, where collecting sufficiently varied real-world data is impractical. Moreover, existing simulation environments provide limited support for the broad spectrum of material behaviors required for generalizable DLO manipulation. To overcome these limitations, we introduce a differentiable physics simulator specifically designed for versatile DLO manipulation. Our simulator models a wide range of material properties—including extensibility, inextensibility, elasticity, bending plasticity, and interactions with both rigid and deformable objects—thereby establishing a robust foundation for learning and evaluating manipulation skills. Building on this simulator, we propose a benchmark suite of representative DLO manipulation tasks that highlight their unique challenges. We further evaluate multiple policy learning algorithms on these tasks. The results show that reinforcement learning can learn closed-loop policies but requires prohibitively large amounts of data. In contrast, trajectory optimization is more efficient: gradient-based methods achieve the best sample efficiency when gradients are available, while sampling-based approaches are broadly applicable but less efficient.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents DLO-Lab, a differentiable simulator, and nine benchmark tasks. In the evaluation, the authors tried out PPO/SAC, CMA-ES, and gradient-based trajectory optimization on their benchmarks. A single slingshot sim-to-real example is also shown. In addition, the authors adopt an LLM “agent” to decompose tasks into subtasks and outputs grasp points as re-planning after each subtask.

### Strengths
- **Task suite.** The tasks cover a wide range of linear deformable manipulation (routing, wrapping, separating, slingshot). Short descriptions of rewards for each task are provided.
- **Method coverage.** RL (PPO/SAC) and planning-based methods (CMA-ES, gradient-based trajectory optimization) are evaluated.

### Weaknesses
- **Task difficulty/coverage is unclear.**
For RL, generalizability is important, but the paper does not show it in the task design. It is unclear how results change under large or medium randomized initial states and targets (e.g., different ball/box placements and angles in Slingshot). The main text points to an appendix for setups, but the results include no robustness evaluation across diverse configurations.

- **Sim-to-real details are insufficient.** 
The paper claims the sim-to-real gap is “manageable” based on one Slingshot demo, but there is no description of the system-ID procedure and no statistics on hardware success. Table 3 lists simulation parameters (e.g., slingshot stretching 8e5, bending 1e5), but the paper does not explain how these values were obtained or validated against the real setup. This limits sientific value and reusability.

- **Experiment setup details are missing.**
Nowadays, simulator speed matters, especially for RL. With a massively parallel setup, one can expect PPO to solve these tasks. The problem is that, for deformables, it is not trivial to run as many parallel environments as rigid-body cases. The paper should report #parallel envs, step time/FPS, and wall-clock. These numbers are also essential for the planning baselines to judge how fast (or real-time) they can produce trajectories.

- **Baselines are outdated**
Deformable manipulation has progressed recently in both model-free RL (HEPi, Hoang et al., ICLR 2025) and model-based methods with simulation gradients (SAPO, Xing et al., ICLR 2025). The authors evaluate only outdated baselines.

- **LLM agent usage is under-specified.**
Section 4.4 says the agent decomposes the task and outputs grasp points, re-planning after each subtask. However, the paper does not state whether the authors train or use one controller per task or different controllers per subtask, whether subtask-specific rewards are used, or whether the LLM agent is called during RL training rollouts or only at evaluation.

- **Videos for both success and failure cases**.
For robotic manipulation, videos are necessary to understand what “success” looks like and to judge jerkiness and/or stability. The paper unfortunately does not provide them.

-------------------------------

**Minor points:**

**Reporting**: Table 2 shows “maximum reward within a fixed number of episodes” with mean ± std, but the number of seeds and exact episode budgets per method are not stated. This weakens interpretability and reproducibility.

**Metrics**: heavy emphasis on reward; success rates would provide better intuition.

-------------------------------

(Hoang et al. 2025). Geometry-aware RL for manipulation of varying shapes and deformable objects. ICLR 2025

(Xing et. al. 2025). Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation. ICLR 2025

### Questions
See weaknesses.

### Soundness
2

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
5

### Summary
The paper presents a suite of deformable linear object manipulations asks, which are built with differentiable simulation Genesis. The main difference comparing to peer works lies in the support of plastic deformation and loop topologies. The proposed tasks cover a range of skills of wiring, coiling, wrapping a linear deformable object against some ambient rigid bodies. Reward design for each task is presented. The paper also includes a decomposition of the multi-stage manipulation task by querying an LLM. The performance with RL and evolution strategy methods are reported. A sim-to-real case is demonstrated on the slingshot task.

### Strengths
1. Simulating and benchmarking linear deformable object manipulation is an important topic and has much relevance to machine learning to robotics.
2. The technical details are easy to grasp. An implementation as an integral part of open-sourced genesis simulation is benefiting to open and reproducible research.

### Weaknesses
1. Limited scientific novelty. All technical methods are well known and the work seems like an aggregation of them for the application to linear deformable objects.
2. Vague contributions. The paper seems trying to include many points that are distant from the core theme of simulating/benchmarking linear deformable manipulation. What is the purpose of the agentic decomposition from LLM? How does it benefit the task scope that the paper is working towards?
3. Relevance of benchmarking tasks. From a robotics perspective, it is unclear how the proposed task suite is capturing the main challenges/key points in manipulating linear deformable objects. And the tasks appear a bit artificial given it is not hard to find industrial or household cases that have a better resemblance of reality, such as cable plugging or tying shoelaces. 
4. Unclear message from the skill learning results. As a benchmark, I suppose the tasks should challenge the existing standard methods while the results in Figure 4 seem that they are already solvable? Are the increasing return curves indicating task successes here? What would the authors expect the users to do with the benchmark if standard methods are already reaching "competitive performance"?

### Questions
1. Can the paper make a concise and clear statement about its scientific contribution?
2. What is the relevance of the proposed task suite to real-world scenarios?
3. Can the LLM-based task decomposition and proposed grasping points be shown to be necessary or useful for a benchmark paper?
4. What are the benefits of the proposed task suites comparing to the peers besides the support of "plastic bending" and "loop topology". In the end, neither of the two features look like something unimaginable if the exiting "labs" got some extensions. Why a new "lab" is necessary here? Are there any other arguments from runtime performance and robustness perspectives?

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
3

### Summary
This paper introduces a novel differentiable physics simulator for Deformable Linear Objects (DLOs) aimed at addressing the limitations of existing simulation environments for robotic manipulation. The primary contribution is a unified framework, "DLO-Lab," that models a wide and comprehensive range of DLO physical properties, including elasticity, inextensibility, bending plasticity, and loop topologies. They propose a benchmark suite of nine diverse DLO manipulation tasks designed to test these varied physical properties and interactions. The paper evaluates the performance of different policy learning paradigms on this benchmark: reinforcement learning (PPO, SAC), sampling-based trajectory optimization (CMA-ES), and gradient-based trajectory optimization. The results show that gradient-based methods are the most sample-efficient when gradients are available, while CMA-ES is a more robust general-purpose optimizer, particularly for tasks with sparse contact.

### Strengths
The framework incorporates several features that were not realized at the same time with other simulators, for example, full differentiability with a rich set of physical properties, bending plasticity, and loop topology, as well as coupling with both rigid and other soft-body materials (MPM), as shown in Table 1.


For the paper clarity, the motivation is strong and well-articulated. Visual aids like Table 1 (comparison to prior work) and Figure 2 (task illustrations) are highly effective at communicating the paper's contributions and scope.

### Weaknesses
1) Sim-to-Real evaluation is limited

The real-world experiment (Sec 5.3) can be a sound proof of concept, but the results shown in the paper are very limited. It demonstrates the transfer of a single open-loop trajectory for one task, which validates the simulator's kinematics and some dynamics but does not validate its suitability for closed-loop control, where a policy would need to react to perception feedback. Inaccuracies in simulated contact forces or friction models, for example, would only be exposed in a closed-loop setting. Additionally, we would like to see the comparison between the simulator and the real in the video.

2) Disconnection of agent task decomposition

Section 4.4 introduces the LLM-based agent for task decomposition, which seems disconnected from the paper's core contribution (the simulator and benchmark).. It's an application of the benchmark, but it's presented as a key feature of the simulator. The implementation details are missing from the text, for example, the details of models, prompt engineering, and robustness.


3) Low reproducibility of the results

The paper does not provide the code or much information about the implementation details, which makes it hard for readers to utilize the proposed framework or reproduce results.

### Questions
1) How robust is the framework the paper proposes? I would like to see the robustness of changing physical parameters, for example, transferring policies for more contact-heavy tasks, or tasks where friction dynamics are more critical.

2) Regarding the Agent Task Decomposition (Sec 4.4), what is the precise role of this agent? Is it used to generate the trajectories for the benchmarked algorithms (PPO, CMA-ES, GD)? Or is it a separate, high-level planner that would use policies trained by these methods as sub-skills?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on manipulation of deformable linear objects (DLOs), such as ropes and cables. The authors introduce a differentiable physics simulator tailored for DLOs, enabling gradient-based optimization for planning and learning. The simulator models elastic and frictional behaviors through a differentiable mass-spring representation, making it suitable for both analytical gradient computation and policy training.

The authors also present a benchmark suite of DLO manipulation tasks, including rope straightening, knot tying, threading, and shape matching. The benchmark systematically evaluates several representative policy learning paradigms, including reinforcement learning (RL), trajectory optimization, and sampling-based motion planning. The evaluation highlights the limitations of existing approaches and motivates future algorithmic innovations for deformable object manipulation.

### Strengths
(+) The introduction of a differentiable simulator for DLOs is highly valuable for the community. It provides a reproducible and extensible software for studying deformable object control, a domain that has historically lacked standardized tools.

(+) The authors evaluate a wide spectrum of methods (RL, trajectory optimization, and sampling-based approaches) under a unified environment, offering an insightful comparison of their respective strengths and weaknesses.

(+) By leveraging a differentiable mass-spring formulation, the simulator enables gradient-based learning and efficient optimization, a clear improvement over prior non-differentiable simulators like PyBullet or SOFA.

(+) Well-structured benchmark tasks: The benchmark tasks span a range of difficulty levels and manipulation types, providing a clear gradient for research progression.

### Weaknesses
(-) Lack of methodological novelty. The paper’s main contribution lies in the simulator and benchmark design; no new algorithm or modeling technique is proposed beyond existing differentiable physics formulations.

(-) Limited insight into algorithmic takeaways. While multiple learning paradigms are compared, the discussion lacks deeper analysis or clear conclusions about why certain methods succeed or fail, or what principles should guide future work.

(-) Weak real-world validation. Although the simulator is claimed to be physically accurate, real-world experiments are minimal and qualitative. A stronger demonstration of sim-to-real consistency would substantially improve the paper’s credibility.

(-) Scalability and efficiency concerns. The computational cost of differentiable simulation for long DLOs or contact-rich scenarios is not clearly discussed. Real-time feasibility remains uncertain.

(-) Limited generalization across object materials: The experiments are primarily conducted with homogeneous ropes; the simulator’s ability to handle variable stiffness or heterogeneous materials is not tested.

### Questions
Is the simulator capable of handling self-collisions or topological changes (e.g., forming or untying knots)?

How well does the differentiable simulator match real-world dynamics? Are there quantitative comparisons between simulated and physical trajectories?

Which policy learning algorithms benefited most from differentiable gradients, and which still relied heavily on sampling?

Could the simulator be extended to sheet-like deformable objects (e.g., cloth) or more complex soft structures?

What is the expected runtime per simulation step, and how does it scale with DLO length and resolution?

### Soundness
2

### Presentation
3

### Contribution
2
