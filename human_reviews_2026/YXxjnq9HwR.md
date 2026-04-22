# Policy Transfer for Improved Sample Efficiency in Goal-Conditioned Reinforcement Learning

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Goal-Conditioned Reinforcement Learning (GCRL) tackles the challenging problem of long-horizon, sparse-reward goal-reaching tasks with continuous actions. Recent methods, relying on a two-level hierarchical policy along with a graph of sub-goal landmarks, have demonstrated reasonable asymptotic performance. However, existing algorithms suffer from poor sample efficiency due to the need to train the low-level policy from scratch, concurrently with the high-level policy, for each given task. We instead claim that transferring a pre-trained low-level policy between environments can dramatically improve sample efficiency and even success rates. We introduce an algorithm PROMO, consisting of a transferable low-level GCRL policy, and a high-level graph-based planner. Our self-terminating landmark generation procedure progressively covers the entire goal space with landmarks based on novelty and reachability. We demonstrate 3-4x improvements in sample efficiency over existing state-of-the-art methods on the challenging robotics tasks of AntMaze and Reacher3D, with the mild overhead of one-time policy pre-training. In addition, our method achieves near 100% success rate in almost all environments, as well as better training stability and much fewer, more informative landmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes PROMO (Progressive Reachability Optimisation MOdelling), a hierarchical framework for Goal-Conditioned Reinforcement Learning (GCRL) that aims to improve sample efficiency by reusing a pre-trained, transferable low-level policy across similar environments. PROMO further integrates a trajectory model to predict the motion of the low-level policy and an accessibility model to estimate valid goal regions, which are jointly used for landmark generation and graph expansion. While the method largely relies on engineering-driven design choices (e.g., TD3+HER, LSTM trajectory prediction, one-class SVMs), the empirical results demonstrate that PROMO can enhance sample efficiency and stability in GCRL benchmarks.

### Strengths
1. Clear and modular problem formulation:
PROMO offers a clean separation between low-level skill learning and high-level planning in the GCRL setting.
Unlike prior methods (e.g., DHRL, BEAG, NGTE) that jointly train both levels, PROMO formalizes the transfer of a pre-trained low-level controller as a fixed reusable module. This modularization improves conceptual clarity and provides a practical recipe for reusing skills across related environments.

2. Model-based reachability formulation:
The paper introduces an explicit reachability model R(g1, g2), defined as the conjunction of a learned trajectory predictor and an accessibility estimator. While simple, this definition represents a step toward formalizing how graph-based hierarchical policies can rely on learned dynamics priors instead of raw distance or value metrics. This shift from distance-based heuristics to a model-based reachability function adds conceptual novelty and a foundation for future model-based planning extensions.

3. Improved sample efficiency through skill transfer:
The empirical results convincingly show that reusing a pre-trained goal-reaching skill leads to faster convergence and greater stability (3~4× sample efficiency gains) across AntMaze and Reacher variants. This provides the first systematic demonstration that transferable skills can meaningfully accelerate hierarchical GCRL without joint retraining, supporting the broader argument that policy reuse can scale beyond single-task settings.

### Weaknesses
1. Limited conceptual originality  :
PROMO relies on well-established components: a low-level policy based on TD3+HER, an LSTM-based trajectory predictor, a one-class SVM for accessibility estimation, and a graph-based planner. While this combination is practically reasonable, it does not present a fundamentally new academic contribution. PROMO can be viewed more as a system-level integration and structural organization of existing GCRL techniques rather than a conceptually novel framework.

2. Limited transferability:
Although the paper emphasizes a transferable low-level policy, all experiments are conducted within the same domain, using the same robot morphology and dynamics (AntMaze variants). For instance, if the agent encounters a new bottleneck structure such as AntMazeBottleneck, the pre-trained skill could not generalize effectively. Similarly, for robot manipulation tasks involving contact-rich interactions rather than simple goal-reaching, the policy transferability would likely break down.

3. Heuristic dependence in landmark generation:
The landmark scoring function is a simple multiplicative combination of Novelty × Reachability × Boundary Separation.
There is no clear justification for why this multiplicative relationship is optimal, nor is there discussion about the potential need for weighting between the terms. Moreover, the overall exploration behavior appears quite similar to NGTE and BEAG, suggesting that the underlying novelty-driven expansion mechanism has not been substantially advanced.

4. Limited experimental diversity and scalability:
All PROMO experiments are conducted in low-dimensional goal spaces (2D AntMaze, 3D Reacher).
The goal space is also handcrafted, defined directly in terms of coordinates such as (x, y).
This design does not scale to high-dimensional or visual goal spaces, nor to contact-rich manipulation settings.
Recent studies have already begun to use latent goal spaces for more complex, visually grounded tasks in environments such as Robosuite, LIBERO, CALVIN, ManySkill, Kitchen (D4RL), and OGBench (Locomotion & Manipulation tasks)

### Questions
Please address the points mentioned in the Weaknesses section above.

In addition, I have a specific concern regarding the claimed sample efficiency and transferability.
You mention that the low-level policy in the AntMaze domain was trained for 2.5M steps separately.
From a single-environment perspective, this makes PROMO appear less sample-efficient than NGTE or BEAG.

The main argument is that PROMO benefits from reusing a fixed pre-trained low-level policy across environments.
However, if we assume that the low-level policy is based on relative subgoals, then NGTE and BEAG could equally reuse the same low-level policy trained in U-maze and keep it fixed when training on Complex-maze.
In that case, the transferability claim of PROMO may not provide a unique advantage.

Would it be possible to evaluate PROMO, NGTE, and BEAG under the same setting where all methods reuse the same pre-trained low-level policy (frozen during high-level training)?
This would allow for a fair comparison focusing solely on high-level graph construction and expansion efficiency, which seems to be the true distinguishing factor of PROMO.

### Soundness
3

### Presentation
3

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
The paper introduces a novel algorithm for hierarchical goal-conditioned reinforcement learning, PROMO, where they show that a low-level policy can transfer across many environments in a hierarchical RL setting. PROMO first trains a low-level goal-reaching policy in a simple, obstacle-free setting using TD3 + HER, which is then frozen and reused. A trajectory model (LSTM) predicts how this low-level policy behaves given a goal, and an accessibility model estimates which goals are reachable in the main environment. A high-level planner builds a graph of landmarks (subgoals) that progressively covers the goal space using a reachability scoring function. Planning between landmarks is performed with Dijkstra’s algorithm.

Their method improves sample efficiency compared to stare-of-the-art methods.

### Strengths
- their algorithm has good performance on the environments they studied
- their algorithm learned what seem like sensible landmarks

### Weaknesses
The paper’s claims and methodology need clearer quantification and exposition. It’s not evident how the authors measure that their landmarks are “fewer” or “more informative,” nor how they determine that the goal space is “covered using only as many landmarks as needed,” since no optimal baseline or comparison is shown. The “near 100% success rate” is also overstated given that the low-level policy is pretrained and transferred rather than zero-shot. The criterion for marking a landmark as “finished” appears arbitrary, lacking clear time or iteration constraints. Figure 2 and Section 4 are confusing, with unclear boundaries between training and inference, and the procedural flow of PROMO is hard to follow without a single unifying formal algorithm.

### Questions
- "much fewer, more informative landmarks": how do you quantify that they're "more informative"? Looking at Figure 3, (c-d), it not at all obvious that the landmarks displayed there is the fewest "most informative landmarks." That is assuming that c-d are showing landmarks. It's not clear what "model predictions" is.
- "our method achieves near 100% success rate in almost all environments" - this is expected because you're training/finetuning/transferring. Unless you are saying that you get 100% zero-shot transfer success rate, which I don't believe you are. You pretrain your lower-level policy to reach small goals in a small, obstacle-free environment. It's unsurprising that this is a good transfer policy for complex AntMaze environments.
- you say the landmark generation method "smartly covers the goal space using only as many landmarks as are needed". How do you quantify that only as many as needed are used. Do you know the true "optimal" amount of landmarks and that your method gets no more than that. Where do you show and compare against that?
- "A landmark is marked finished when exploring from it no longer gives novel achieved goals" - what time constraints do you use for this? This seems like an arbitrary criterion.
- Figure 2 is not easy to read. How do I read (2) of the left side. How are the boundaries selected? Are you describing training/action-selection? This is very confusing.
- Section 4 was not easy to read at all. The delineation between training/action-selection was not clear throughout. This section would benefit from a formal algorithm to describe everything together.
- you say your method reaches near 100% success rate in almost all environments, but figure 4b shows that your method is far from 100% if you account for the quite large variance it has. this is seems like an(other) example of overclaiming.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates policy transfer in goal-conditioned reinforcement learning (GCRL) through a hierarchical framework that separates a transferable low-level controller from a high-level planner. The proposed method, named PROMO, aims to improve sample efficiency by reusing a pre-trained low-level policy across environments. Experiments on AntMaze and Reacher tasks demonstrate significant sample-efficiency gains and higher success rates compared to existing hierarchical GCRL baselines.

### Strengths
1. The paper addresses an important and still open problem in reinforcement learning: improving sample efficiency in long-horizon sparse-reward goal-conditioned tasks. The research question is interesting and relevant to the field.

2. The empirical section is clear, and the performance improvements over baselines such as DHRL, BEAG, and NGTE are well presented.

3. The paper provides detailed descriptions of the algorithmic components and training pipeline, which helps reproducibility.

### Weaknesses
***1. Limitation of novelty:*** The main idea of separating high- and low-level policies is not new in hierarchical reinforcement learning. The structure closely follows prior work such as HIRO, HAC, and other two-level GCRL frameworks. Although the authors introduce transfer of the low-level policy, the proposed design does not clearly resolve the fundamental bottlenecks known in HRL systems. In particular, the high-level policy still relies on an abstract transition model without sufficient domain knowledge or grounded dynamics understanding. It is unclear how the high-level planner can reason about long-term transitions or choose sub-goals efficiently when the reachability model and trajectory predictor are both learned in limited or simplified environments. The paper would benefit from a deeper analysis or theoretical argument explaining why this decomposition can generalize beyond the pre-training domain.

***2. The design of LSTM.*** Based on Weakness1, it is more questionable about the effectiveness of the LSTM trajectory model. Using an LSTM to predict the trajectory of the low-level policy is an acceptable but limited choice. It works reasonably well in simple settings like AntMaze, where dynamics are low-dimensional and repetitive. However, such models struggle to represent multi-modal trajectories or long-range dependencies in complex domains. The experiments indeed demonstrate that while the approach performs well in the maze tasks, its advantage diminishes in the Reacher environment. This observation suggests that the trajectory model is not robust to more complex or highly nonlinear transitions. The authors should discuss how this component scales to more realistic or high-dimensional environments.

***3. Weak baselines and missing analysis.*** The selection of baselines is not sufficiently comprehensive. While the comparisons with DHRL, BEAG, and NGTE are standard, other relevant methods such as SoRB should have been included. In particular, SoRB directly addresses goal-space planning using learned value functions. CO-PILOT directly use the agent's exploration knowledge for planning. Moreover, the paper does not report the training cost or computational overhead of the additional LSTM trajectory model. Given that the method involves multiple training stages (pre-training, trajectory modelling, and high-level planning), reporting wall-clock time or FLOP estimates would strengthen the claim of improved sample efficiency.

### Questions
The methodological contribution appears close to that of existing HRL frameworks and even overlaps conceptually with recent transfer-based GCRL approaches. For instance, the ideas in “Policy Transfer for Improved Sample Efficiency in Goal-Conditioned Reinforcement Learning” resemble those discussed in https://arxiv.org/abs/2508.06108v1. Although the details differ, the general direction and motivation are highly similar. The authors should clarify the distinctive technical contribution and explain how their algorithm fundamentally advances beyond these existing efforts. Additionally, as noted earlier, the maze environment is simple enough that Hindsight Experience Replay alone can achieve strong performance. It is unclear whether a full hierarchical setup is necessary for such a domain.

### Soundness
2

### Presentation
2

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
This paper introduces PROMO (Progressive Reachability Optimisation MOdelling), a novel approach to improve sample efficiency in Goal-Conditioned Reinforcement Learning (GCRL). The key innovation is the application of policy transfer to GCRL by pre-training a low-level policy in a simple, obstacle-free environment and then transferring it to complex target environments. The method consists of three main components: (1) a transferable low-level GCRL policy trained using TD3+HER, (2) a trajectory model that predicts the policy's behavior, and (3) a high-level graph-based planner that uses a smart landmark generation procedure. The landmark generation optimizes over novelty, reachability, and boundary separation scores to progressively cover the goal space. Experiments on AntMaze and Reacher environments demonstrate 3-4x improvements in sample efficiency and near 100% success rates compared to state-of-the-art methods.

### Strengths
1. This paper introduces a new landmark generation procedure, incorporating three meaningful components (novelty access, reachability, and boundary separation) that lead to fewer but more informative landmarks.

### Weaknesses
1. The novelty and motivation of this paper is not clear. The authors claim their contribution as pretrained transfer skills in GCRL. However, a general game AI model should be tested beyond the navigation tasks (e.g., chess, shogi, Atari). 
2. The novelty of algorithms is very limited. There are many previous GCRL measured the reachability of the low-level policy.   
3. Moreover, the method is constrained by the choice of SVM ensemble for accessibility modeling, which limits applicability to (i) Low-dimensional goal spaces only (2D for AntMaze, 3D for Reacher) and (ii) Environments where obstacles can be modeled by the SVM ensemble.

### Questions
1. How sensitive is the method to the choice of a simple environment? What guidelines exist for constructing appropriate, simple environments for different domains?
2. How robust is the method when there are significant differences in dynamics between the simple and complex environments beyond just obstacle placement?
3. How could the method be extended to higher-dimensional goal spaces? Would replacing the SVM ensemble with neural density models (as suggested) maintain the same performance benefits?
4. Could the pre-trained policy transfer across different types of environments (e.g., from navigation to manipulation) or is it limited to variations within the same domain?
5. How would the method perform in real robotic systems where there might be additional sources of uncertainty and dynamics mismatch?

### Soundness
2

### Presentation
2

### Contribution
2
