# GapONet: Nonlinear Operator Learning for Bridging the Humanoid Sim-to-Real Gap

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
The sim-to-real gap, arising from imperfect actuator modeling, contact dynamics, and environmental uncertainty, poses fundamental challenges for deploying simulated policies on physical robots.
In humanoids, object manipulation further amplifies this gap: end-effector payloads alter joint inertia, gravity torques, and transmission efficiency, introducing state- and payload-dependent nonlinearities. Yet existing approaches lack both systematic analysis and a generalizable representation of this payload-induced degradation.
To address this limitation, we propose GapONet, a payload-conditioned nonlinear operator that maps simulation context functions to residual actions for hardware. We then introduce a payload-aware <collect–analyze–solve> framework to learn this operator GapONet. First, we curate a sim-real paired dataset TWINS spanning multiple payloads, robots, motions, actuation rates, and simulators, comprising more than 11,298 motion sequences. Second, we perform payload-aware system identification to isolate payload-related effects and quantify their contributions, and analyze sim-to-real gaps across different simulators. Third, we train the operator GapONet to predict delta action for real-time, generalized, payload-conditioned compensation. We further introduce actuation functions and sensor predictors, which enable parallel RL training of GapONet with substantially reduced energy consumption.
While tracking unseen motions, GapONet keeps the incidence of large sim-to-real gaps below 0.09%, whereas competing methods remain near 10%. By correcting upper-body gaps, GapONet also stabilizes lower-body locomotion tracking, laying the foundation for improved performance in humanoid loco-manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a well-motivated study on improving sim-to-real transfer for humanoid robots. It contributes a sim–real data collection pipeline and introduces TWINS, the first dataset focusing on payload-induced domain gaps across multiple robots, motions, and simulators. The authors further provide 30+ hours of synchronized sim–real data and quantitative analyses of simulator discrepancies. Finally, they propose GapONet, a payload-conditioned nonlinear operator that maps simulation actuation functions to residual hardware actions, demonstrating its feasibility through reinforcement learning.

### Strengths
1. The paper addresses a highly practical challenge in humanoid sim-to-real transfer by explicitly modeling the payload effect not as random noise but as a structured conditioning variable during motion tracking. This perspective is realistic and directly relevant for real-world humanoid deployment.
2. Building on DeepONet’s operator-learning framework, the paper presents a novel combination between nonlinear operator learning and the sim-to-real transfer process in robotics. It highlights the separable nature of the payload by treating it as the query variable in the operator formulation. Moreover, it introduces an insightful perspective on sim-to-real alignment—modeling the domain gap as a mapping from simulation actuation functions to residual actions, rather than as pointwise corrections.
3. The paper provides detailed technical descriptions of its data collection pipeline, including the construction of a cross-robot, cross-simulator dataset covering diverse humanoid motions under varying payload conditions. The dataset curation and labeling process are thoroughly documented.
4. The paper also includes a joint-level comparison across three mainstream simulators. Although not comprehensive and is not the focus of this paper, this analysis provides useful insights and contributes valuable reference for the community.

### Weaknesses
1. The paper does not compare the proposed method with other nonlinear system identification approaches, such as neural network-based or kernel-based methods. The related work section on nonlinear system identification is insufficient.
2. The generalization capability of GapONet beyond payload variation remains unclear. As a method for bridging the sim-to-real gap, its applicability under other changing factors is not demonstrated.
3. The experimental settings lack clarity, e.g. the training details of baseline methods are missing, including the choice of training and test sets, which makes reproducibility difficult.
4. The training pipeline and design choices are not well explained. It is unclear why reinforcement learning was chosen.
5. Some mathematical symbols appear in the paper without explicit definitions, making the derivations harder to follow.

### Questions
1. In Figure 1, why does the dataset illustration show the Unitree G1 performing Kung Fu, while the data used for experiments only include three types of lower-body gaits without whole-body tracking motions?
2. Why did you choose a reinforcement learning (PPO) algorithm to train GapONet, given that the original DeepONet paper uses supervised learning? The paper does not explain the motivation or provide the formal RL formulation.
3. In Section 4.3, you claim that computing all sensor values is computationally prohibitive. Could you clarify whether this limitation is due to the RL setup? If so, why is RL used instead of supervised regression?
4. Why is the default large-gap ratio set to 0.5 rad, especially when Figure 3(b) shows that the typical error is below 0.3 rad?
5. In Section 3.3.2, you state that the deviation is nonlinear and that phase lag is related to payload. However, the phase difference between sim and real does not necessarily imply delay, and there is no quantitative result showing nonlinearity. Could you clarify this claim?
6. How did you select the data and motions used in Section 3.3? In Figure 3(b), the deviation at 0 kg payload reaches 0.1 rad, but in Figure 3(c), the inter-simulator joint angles appear much smaller. Why?
7. In the motion tracking and trajectory tracking experiments, what role does GapONet play in real-world tests? 
8. In Section 5.2, you mention an online residual compensation method. Could you describe this method in detail? It is currently unexplained.
9. In Section 5.1(ii), what is the "Transformer-learned dynamics model"? Please specify its structure and training configuration.

Additional Feedback:

1. Writing and formatting:
   - Redundant sentence in Section 2.2.
   - Uneven layout and spacing issues in Figure 1.
   - Multiple typos in the *Method* section.
2. Technical presentation:
   - Please define all mathematical symbols when they first appear.
   - Consider reorganizing Section 4 to make the training pipeline easier to follow.
   - Provide explicit details for experimental setup.

### Soundness
3

### Presentation
2

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
This paper studies sim-to-real gaps introduced due to the end-effector payload during object interaction for humanoid loco-manipulation tasks. The authors proposed TWINS, the first dataset focused on payload-induced sim-to-real gaps across multiple robots, and found a consistent nonlinear increase in dynamics error. They address this by using GapONet, a nonlinear operator that maps simulation context features to real with a function-to-function learning objective, to propose delta actions that compensate for such gaps. Through empirical studies, they demonstrate the effectiveness of their method in motion tracking tasks.

### Strengths
This paper is well motivated, it provides solid problem formulation and theoretically analysis.

### Weaknesses
There exist multiple typos in this manuscript (for example: “Modelin” in figure 1) and inconsistent font. These errors in the main figure may reduce reader confidence in the presentation.

### Questions
1. "The discrepancy arises from coupled channels—gravity, friction, Coriolis and inertial coupling, actuator limits and efficiency drift, sensing noise, and delays—that a pointwise function mapping cannot capture or generalize." I am still not convinced why learning operators of actuator functions has been necessary or superior to point-wise mappings. Can authors provide more empirical evidence that the insufficiency of the point-wise mapping method fails to generalize?

2. In Table 1, while all methods demonstrate relatively close IQR and Range, baseline methods have significantly larger LGR. What is the explanation of this difference? I am also curious about the effectiveness of these metrics. Which one of the does the author consider the most faithful in measuring the sim-to-real gap?

3. How does the proposed method work for motion tracking that includes agility or actual object interaction? Motion tracking with payload might not be

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
This paper, "GapONet: Nonlinear Operator Learning for Bridging the Humanoid Sim-to-Real Gap," addresses the critical challenge of transferring policies learned in simulation to real-world humanoid robots. The authors correctly identify that the sim-to-real gap is exacerbated by complex, payload-induced nonlinearities and unmodeled dynamics in high-DoF systems. To tackle this, they propose GapONet, a novel payload-conditioned nonlinear operator network. GapONet is designed to learn a mapping from the simulation context function (i.e., the state and action in sim) to the residual action required on the real hardware, effectively acting as a nonlinear correction layer. The authors also introduce TWINS, a large-scale, synchronized sim-to-real dataset collected across multiple simulators and a real humanoid platform, which is a significant contribution in itself. Experimental results demonstrate that GapONet achieves superior performance in reducing the sim-to-real gap compared to competitive baselines, showing a reduction in tracking error and improved stability, particularly under varying payload conditions.

### Strengths
The core idea of framing the sim-to-real gap correction as a nonlinear operator learning problem is highly original and compelling. While prior work has used residual learning, the application of a payload-conditioned operator network (inspired by Neural Operators) to model the complex, functional relationship of the sim-to-real discrepancy is a novel approach in the context of humanoid robotics. The use of a branch-trunk decomposition to separate the payload-independent dynamics from the payload-dependent non-linearities is a clever architectural choice that enhances generalization. The work is highly significant for the field of sim-to-real transfer, especially for complex, high-DoF systems like humanoids. The introduction of the TWINS dataset is a valuable resource for future research. The GapONet model offers a powerful, generalizable framework for modeling complex unmodeled dynamics, which could be broadly applicable beyond payload variation to other sources of discrepancy (e.g., friction, compliance).

### Weaknesses
1. While the operator network formulation is the central claim of the paper, the experiments do not sufficiently justify its necessity over a standard, high-capacity Multi-Layer Perceptron (MLP) with the same payload conditioning. The authors should provide an ablation comparing GapONet to a simpler, non-operator network that takes the same inputs (sim context and payload) and outputs the residual action. Without this, it is difficult to ascertain if the performance gain is due to the operator learning formulation or simply the nonlinear, payload-conditioned residual structure.

2. The paper focuses heavily on payload variation. While this is a critical source of non-linearity, the true test of an operator network is its ability to generalize across different functional inputs. The current evaluation only tests generalization across a continuous parameter (payload mass).

3. he paper mentions the curation of the TWINS dataset, which is a major contribution. However, the paper does not explicitly state whether the dataset and the trained GapONet models will be made publicly available. Given the scale and complexity of the data collection, the lack of public release significantly hinders the reproducibility of the results and limits the impact of the dataset contribution.

4. The paper's primary focus is on a model-based correction approach. A key alternative for sim-to-real is robust policy learning via Domain Randomization (DR). The paper should include a more direct and quantitative comparison to a strong DR baseline, where the policy is trained with randomization over the payload range, to demonstrate the superiority of the GapONet correction approach in terms of sample efficiency or final performance.

### Questions
1. Ablation on Operator vs. MLP: Could the authors provide an ablation study comparing the proposed GapONet architecture against a standard, high-capacity MLP that is also conditioned on the payload and the simulation context? This is crucial to isolate the performance benefit derived specifically from the operator learning framework.

2. Generalization to New Tasks: The current experiments focus on generalization across payload mass for a fixed set of motions. Can the authors comment on or provide results for the generalization of a trained GapONet to a completely new motion or task that was not part of the TWINS training set?


3. Computational Overhead: What is the inference time overhead introduced by GapONet on the real hardware? Given that the correction is applied at the control frequency, the latency is critical. Please provide a quantitative measure of the inference time compared to the control loop frequency.

4. Role of the Branch-Trunk Decomposition: The paper mentions the branch-trunk decomposition. Could the authors elaborate on the specific functional form learned by the trunk network? Is the trunk network primarily learning the payload-independent dynamics, and the branch network the payload-dependent non-linearities, as hypothesized? A visualization or analysis of the learned functions would be highly informative.

### Soundness
2

### Presentation
3

### Contribution
2
