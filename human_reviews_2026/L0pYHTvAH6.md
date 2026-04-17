# Collision- and Reachability-Aware Multi-Robot Control with Grounded LLM Planners

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Large language models (LLMs) have demonstrated strong performance in various robot control tasks. However, their deployment in real-world applications remains constrained. Even state-of-the-art LLMs, such as GPT-5, frequently produce invalid action plans that violate physical constraints, such as directing a robot to an unreachable location or causing collisions between robots. This issue primarily arises from a lack of awareness of these physical constraints during the reasoning process. To address this issue, we propose a novel framework that integrates reinforcement learning with verifiable rewards (RLVR) to incentivize knowledge of physical constraints into LLMs to induce constraints-aware reasoning during plan generation. In this approach, only valid action plans that successfully complete a control task receive positive rewards. We applied our method to two small-scale LLMs: a non-reasoning Qwen2.5-3B-Instruct and a reasoning Qwen3-4B. The experiment results demonstrate that constraint-aware small LLMs largely outperform large-scale models without constraint knowledge training, grounded on both the BoxNet task and a newly developed BoxNet3D environment built using MuJoCo, which involves LLM planning for up to 25 robots. This work highlights the effectiveness of grounding even small LLMs with physical constraints to enable scalable and efficient multi-robot control in complex, physically constrained environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces two BoxNet environments to evaluate LLM-based multi-robot control. An A* based algorithm is used to generate golden plans. Reasoning traces are collected from the A* plans and annotated by LLMs. An efficiency reward component is proposed for GRPO method. Overall the novelties of this work are not sufficient for the conference.

### Strengths
1. The writing is clear and the paper is easy to follow.
2. Two new environments are proposed to evaluate LLM-based multi-robot control.

### Weaknesses
Overall the novelties and contributions of this work are limited for the following reasons:
1. The training protocol that utilizes SFT warmup followed by GRPO is conventional. The sole modification to this standard approach is the inclusion of a simple efficiency reward term r_{efficiency} within the RL objective.

2. The proposed BoxNet2D and BoxNet3D are quite simple and straightforward. The main different from previous BoxNet is use of discrete spatial coordinates for actions instead of choosing from predefined actions.

3. The paper evaluates FULLPLAN planner and REPLAN planner but the comparison between those two settings is not fully discussed. The results of these two settings are mixed in Table 2. The claim regarding this experiment setting is unclear.

4. Errors:
Line 343: Broken reference "Figure ??"
Line 359: left "(" is missing

### Questions
1. Based on the formula of efficiency reward calculation, r_{efficiency} is zero when len(s) < len(s*). It penalizes the excessive length compared to the A*-based golden plan, but it won't encourage a plan with shorter length than the "golden plan". How to explain the reduced StepDiff for RL-trained model?

2. The success rate and efficiency are normally considered contradictory to each other. I would expect adding the efficient reward for RL will hurt the success rate while reducing the plan lengths. However, the results of ablation study in Table 5 shows that this efficient reward can improve the success rate while reducing the total steps. Is there any explanation or hypothesis for this phenomenon?

3. The environments have different map sizes ranging from 2x2 to 6x6 (BoxNet). How does map size affect the results of RL training?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a collision and reachability-aware framework for multi-robot control, using grounded large language model (LLM) planners. To address the inherent limitations of LLMs regarding physical and geometric reasoning, the authors introduce Reinforcement Learning with Verifiable Rewards (RLVR). RLVR fine-tunes smaller LLMs (like Qwen2.5-3B-Instruct and Qwen3-4B) with environment feedback, rewarding only physically valid plans. This method reportedly enables smaller, grounded models to outperform larger state-of-the-art models like GPT-5 in collision-free planning for up to 25 robots in the new BoxNet2D and BoxNet3D environments. The work suggests that incorporating physical constraints through RL enhances the capability and reliability of more compact LLMs for scalable robot control, though the practical scalability and computational efficiency of the RLVR process itself would benefit from further discussion.

### Strengths
- The main idea of the paper, grounding an LLM with physical motion constraints,  is reasonable and easy to understand. The paper is clearly written and well-structured, making the overall framework and experiments easy to follow. The claimed contribution is also clear.

- At the conceptual level, the approach to the problem seems like it could work effectively, but under structured and somewhat simpler setups.

- The code and especially the video examples clearly demonstrate the proposed framework.

### Weaknesses
- The core idea of grounding an LLM with physical constraints is a significant and important topic. However, the proposed approach does not seem to offer a substantial conceptual contribution, as the authors retrained a slightly different formulation of a well-known RL objective function (GRPO).

- Regarding the claimed contribution, describing the physical constraints as "realistic" seems to be an overstatement, given the simplicity of the collision and reachability checking mechanisms employed. Furthermore, the environments appear to be designed in a way that minimizes potential robot-robot collisions, which subsequently undermines the claims about reachability.

- The claimed generalization ability of this method appears to be limited. Section 4.2 describes generalization to unseen environments, but these unseen environments share the same underlying grid structure as the training environments. Consequently, the claim of generalisation slightly overstates the scope of what the proposed approach can truly handle.

- The details concerning the robot’s action space, size, and reachability conditions, as well as the collision checking method, should ideally be explained in the main body of the paper rather than being relegated to prompts in the appendix.

- The solution generation time should be considered and added as an important evaluation metric.

- The study would be significantly improved by including a comparison of the proposed RL method with other relevant works in the literature.

- Table 2 in the results section could be more self-contained and clearer. Not all the evaluation metrics are adequately defined or clear from the table caption alone.

- In the related work section, the authors argue that SoTA methods struggle due to simplified physical constraints. However, the proposed work also appears to utilize similarly simplified constraints, which creates a point of inconsistency.

- An analysis or discussion of why the A* algorithm generates a less efficient plan compared to the proposed method should be included.

- The paper should include an example output within the document itself, rather than only providing it on an external website.

- The actual use or role of RRT motion planning in the proposed framework is not clearly stated or elaborated upon.

- Figure numbers are missing in some places (e.g., Line 343).

- Missing opening parenthesis “(“ (Line 359).

- Typo: “fanalyze” should be corrected to “analyze” (Line 372).

- Sentence incomplete: The sentence starting with “due to…” (Line 862) needs completion.

### Questions
In addition to the issues raised above (Weaknesses), here are some further questions:

- ${A}^{\*}$ is supposed to be an optimal algorithm. I would expect any algorithm to generate a plan with the same number of steps as ${A}^{\*}$ or more. Are the heuristics used in ${A}^{\*}$ admissible?

- Why are 2D objects in BoxNet modeled as point objects?

- Why do the robots in the videos exhibit unnecessary waiting, despite the availability of a handover mechanism?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an RL‑with‑verifiable‑rewards (RLVR) pipeline to “ground” small LLMs in multi‑robot planning constraints (reachability and collision avoidance). FULLPLAN (open‑loop) and REPLAN (closed‑loop) planners are trained and evaluated. “Physical constraints” are injected via the executable reward: a plan receives credit only if it both completes the task and passes programmatic checks for reachability/feasibility and collisions; an additional efficiency penalty encourages shorter/parallelized plans. Two BoxNet environments are used: a modified 2D grid world (up to 25 robots) with analytic checks for arm reach and collisions, and a 3D MuJoCo‑based UR5e setup (up to 9 robots) with RRT‑based motion and physics‑based reachability/collision checks.

### Strengths
1. Clear problem focus and executable grounding. The reward integrates verifiable checks for reachability/feasibility and robot/object collisions; only physically valid, task‑completing plans are rewarded. This is a reproducible recipe for constraint‑aware planning behavior. 

2. Strong empirical gains with small models. Grounded 3B/4B models outperform much larger baselines across both 2D and 3D setups. 

3. Thoughtful analysis of reasoning. The paper probes for emergent feasibility checks in the chain‑of‑thought, and shows RL increases explicit reachability/collision checks in the reasoning.

4. Open‑ vs closed‑loop comparison. Evaluating FULLPLAN and REPLAN is useful for understanding where feedback helps. 

5. Implementation clarity and cost accounting. The paper details datasets, constraints, prompts, GRPO settings, and compute to support reproducibility.

### Weaknesses
1. Prompt fairness on reachability (BoxNet2D). For BoxNet2D inference prompts, the textual context emphasizes collision rules but does not clearly encode numeric reachability limits; reachability is enforced by the simulator/reward. In contrast, BoxNet3D prompts do include explicit reachability bands/geometry. This asymmetry muddies the “prompt fairness” story across settings and may partially credit RL for implicitly learning a rule that was not textually available to zero‑shot baselines in 2D. 

2. Sim‑only, single family of tasks. Results are limited to BoxNet variants; there are no real‑robot evaluations or other manipulation benchmarks, though the authors acknowledge this limitation in the paper. 

3. Limited robustness stress tests. While the paper tests layout/coordinate/map variations, it does not evaluate controller noise/disturbances or perception errors, nor the latency of REPLAN vs FULLPLAN under tight timing.

4. Efficiency vs safety trade‑offs. The negative efficiency term shapes behavior (Table 5), but its sensitivity and potential side‑effects (e.g., overly aggressive parallelism near constraint boundaries) are not fully characterized.


Minor questions:
1. Would a well broken‑down reasoning process or a proven multi‑agent system help in multi‑robot tasks?

2. What exactly are the “physical constraints” incorporated into the reward? Could this be explicitly encoded into the prompts and tested with LLMs?

3. Why Qwen2.5‑3B‑Instruct and Qwen3‑4B? If one instructed model and one thinking model are preferred, I think choosing Qwen3-4B-instruct and Qwen3-4B-thinking will add one more ablation and may derive an interesting conclusion.

4. Missing figure cross-ref on Line 343.

### Questions
This method is simple and interesting. I have listed my concerns and some minor questions in the weakness section. Look forward to the authors' rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
