# Emergent Dexterity Via Diverse Resets and Large-Scale Reinforcement Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Reinforcement learning in massively parallel physics simulations has driven major progress in sim-to-real robot learning.  However, current approaches remain brittle and task-specific, relying on extensive per-task engineering to design rewards, curricula, and demonstrations. Even with this engineering, they often fail on long-horizon, contact-rich manipulation tasks and do not meaningfully scale with compute, as performance quickly saturates when training revisits the same narrow regions of state space. We introduce \Method, a simple and scalable framework that enables on-policy reinforcement learning to robustly solve a broad class of dexterous manipulation tasks using a single reward function, fixed algorithm hyperparameters, no curricula, and no human demonstrations. Our key insight is that long-horizon exploration can be dramatically simplified by using simulator resets to systematically expose the RL algorithm to the diverse set of robot-object interactions which underlie dexterous manipulation. \Method\ programmatically generates such resets with minimal human input, converting additional compute directly into broader behavioral coverage and continued performance gains. We show that \Method\ gracefully scales to long-horizon dexterous manipulation tasks beyond the capabilities of existing approaches and is able to learn robust policies over significantly wider ranges of initial conditions than baselines. Finally, we distill \Method \ into visuomotor policies which display robust retrying behavior and substantially higher success rates than baselines when transferred to the real world zero-shot. Project webpage: https://omnireset.github.io

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on general manipulation skill learning through reinforcement learning in simulation. The authors propose a method using large-scale generation of reset states based on user-specified points of interest, eliminating the need for complex reward or curriculum design while improving training efficiency and robustness. Experiments conducted in both simulation and the real world validate the effectiveness of the proposed approach.

### Strengths
1. This paper clearly identifies a critical problem in current reinforcement learning methods for learning general manipulation skills within simulation and proposes using large-scale, valuable reset states to constrain the exploration space.
2. The paper is clearly written, concise, and includes a comprehensive evaluation.

### Weaknesses
1. The current tasks are not long-horizon enough. For example, real-world assembly often involves multiple-part assembly. How should the points of interest be designed in such cases? For a chair with four legs, the set of points could be very large, making sampling potentially time-consuming.
2. The current experiments use only a robotic arm. How would the approach scale to a high-DoF dexterous hand? How can valid grasps be sampled? 
3. For manipulation tasks that involve interaction with the environment, such as using a wall to adjust an object’s pose, current method seems can not effectively restrict the exploration space.
4. Since the points of interest may vary across objects, scaling to diverse objects might still require object-specific selection and significant user trial-and-error.

### Questions
1. How exactly are the goal configurations defined? What happens if defining them is difficult, for example, in the case of a dexterous hand?

### Soundness
3

### Presentation
3

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
This paper proposes a method to generate diverse initial states to enable more efficient reinforcement learning of long-horizon manipulation tasks. This is demonstrated on three contact-rich manipulation tasks, compared with baseline methods that utilize demonstration data. Sim-to-real transfer is achieved via distillation and cotraining with real data.

### Strengths
The proposed idea is simple and well executed on the proposed tasks.

### Weaknesses
I think the paper lacks enough evaluation to turn this into a really strong paper, for example:

1. The number of tasks used is quite limited. Given the rich set of tasks introduced in furniture-bench, where two of the tasks are coming from, it would be nice to evaluate the method on more tasks and report how many tasks are successful. This can give an idea of the limitations of the proposed method and what follow-up work can do to improve the results. 

2. Four different sources of reset distributions are proposed. However, this paper lacks ablations to verify how important those distributions are.

### Questions
1. The details of the baseline are missing, e.g., how many demonstrations are used? 

2. Given the success of imitation learning using a diffusion policy with demonstration, it is not clear to me why a pure diffusion policy wouldn't work with demonstration. Can the authors provide some explanation?

### Soundness
2

### Presentation
3

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
This paper introduces OmniReset, a simple and scalable framework for training reinforcement learning policies for complex, long-horizon, and contact-rich manipulation tasks.

The framework's key insight is to cleverly bypass the difficult exploration problem in RL. It automatically generates a large-scale, diverse distribution of reset states that covers all "reasonable" points along the path to the goal. This allows complex, dexterous behaviors to 
emerge naturally from standard large-scale RL optimization. OmniReset is designed to operate without human demonstrations, auto-curricula, or extensive task-specific reward/hyper-parameter engineering.

The method constructs its diverse initial state distribution(beta ρ) by composing four sub-datasets: Reaching Resets (D^R), Near-Object Resets (D^NO), StableGrasp Resets (D^G), and Near-Goal Resets (D^NG). This only requires minimal high-level user specifications: a set of goal configurations (G), a workspace bounding box (W), and a bounding box for near-goal/contact-rich states (NG).

The Experimental results show that OmniReset significantly outperforms demonstration-based baselines (BC-PPO, DeepMimic, Demo Curriculum) on Easy task variants and successfully scales to the Hard variants of Screw, Drawer, and Peg tasks, where baselines struggled to make meaningful progress. Furthermore, the resulting policies exhibit superior robustness to perturbations. Finally, the paper shows that policies trained in simulation can be successfully transferred to a physical robot through vision-based distillation and co-training with a small amount of real-world data.

### Strengths
1. Novelty in Solving Exploration: The main contribution is the proposal and validation of using a diverse, minimally structured set of resets as an alternative to complex auto-curricula or reliance on expert demonstrations to solve long-horizon RL exploration challenges.

2. Scalability and Performance: OmniReset successfully enables RL (PPO) to master complex, contact-rich tasks (e.g., Screw Hard) that were previously beyond the reach of existing techniques. Achieving success rates of over 97% on these benchmarks is a significant 
performance breakthrough

3. Robustness: Policies trained with OmniReset show exceptional robustness to perturbations and succeed over a much broader range of initial conditions compared to baseline methods like Demo Curriculum.

4. Effective Sim-to-Real Pipeline: The work demonstrates a practical and high-value use case by distilling the learned expert policy into a visuomotor policy that achieves a 30% real-world success rate after co-training with only 100 real demonstrations, vastly outperforming a policy trained only on the real demonstrations (0% success).

### Weaknesses
1. Shifts Complexity from Algorithm to Task Configuration: While the paper minimizes algorithmic complexity, it shifts the burden to non-trivial task-specific configuration requirements, which are the basis of the diverse resets. This challenges the central claim of "minimal human input”

Near-Goal State Bounding Box (NG): Defining the bounding box for contact-rich states requires expert knowledge about the precise geometry and interaction points necessary to solve the task (e.g., the threads and corresponding hole for the Screw task). This step essentially encodes a critical part of the task solution as a manual configuration, introducing a significant, task-specific engineering step.

Pre-computed Grasps: The method relies on a pre-computed dataset of feasible grasps from a sophisticated grasp sampler. This is a powerful but unstated dependency. For novel objects or robot hands where such a tool is unavailable, generating this grasp dataset would be a substantial engineering effort in itself, acting as a form of implicit prior knowledge that is not universally available.

2. Uncertain Generalizability Beyond Assembly Tasks: The effectiveness of the proposed four-part reset strategy (R,NO,G,NG) is demonstrated on three tasks that, while complex, all fall within the category of rigid object assembly and insertion. The framework's generalizability to manipulation paradigms with different structures is unproven

### Questions
While the authors achieve strong results on grasping and insertion, can this design generalize to other tasks, and can you quantify the labeling effort (annotation cost) required to design the task-specific annotations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a simple method to enable RL to learn long-horizon manipulation tasks. The key idea is to initialize the scene in a diverse set of initial states relevant to the task. The authors argue that "attempting to cover this space of behaviors may initially
appear intractable," but "the space of ‘reasonable’ manipulation behaviors is actually surprisingly small". To this end, the authors propose resetting to reaching, near object, stable grasp, and near goal. A user aids the process by providing near goal states. The user needs to provide a set of goal configurations, a workspace for the robot, and a set of near goal states, including contact-rich and goal states.

The authors show results in 3 challenging versions of manipulation tasks - drawer, screw, and peg in simulation. Additionally, the authors show the transfer of the screw task to the real world. To achieve real robot results, a state-based policy is distilled into a vision policy. Additionally, the authors found that it's necessary to co-train this distilled policy with a small amount of real robot data.

### Strengths
1. Impressive long-horizon manipulation task
2. Empirically, significant improvement over baseline

### Weaknesses
1. Only one task is shown on the real robot. What was the reason for not showing real-world results on all 3 tasks?
2. The authors claim "OmniReset automatically generates these resets with minimal human input". In that regard, I would expect to see results on more tasks.

**Minor**
1. Line 186: "We will let $s ∈ S$ denote the state space" -> "We will let $s ∈ S$ denote the state".
2. "transition dynamics of the simulator by $s′ ∼ P(·|s)$" -> "next state sampled from the transition dynamics of the simulator by $s′ ∼ c. P(·|s)$"
3. "$a ∼ π(·|s)$ to denote control policies" -> "$a ∼ π(·|s)$ to denote action sampled fromm control policy"
4. Line 211: Suggestion $J(π) = E_{s_0∼ρ,a∼\pi}[Σ_{t=0}^∞ γ^t r(s_t, a_t)]$ and remove "expection is alos taken w.r.t. the actions..."
5. In Figure 2 caption & on line 342: space after OmniReset
6. Line 359: Grammar "Both setting the orientation"

### Questions
1. Line 372: "standard reset distributions $ρ^S$ described above" - what exactly is it, and which line is it defined in?
2. What happens if we initialize around demo states with added perturbations? With enough perturbations, do you expect the state distribution to be sufficient and work similarly to the proposed method?
3. Since the core idea is having sufficient reset coverage in relevant states, is it possible to gather them without human assistance? E.g., via disassembly and perturbations around it?
4. How were the 100 real-world demos collected?
5. Several related complementary works are cited in the Related Work section. What happens when Omni Reset is combined with some of them?

### Soundness
3

### Presentation
2

### Contribution
3
