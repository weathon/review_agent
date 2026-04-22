# When a Robot is More Capable than a Human: Learning from Constrained Demonstrators

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Learning from demonstrations enables experts to teach robots complex tasks using interfaces such as kinesthetic teaching, joystick control, and sim-to-real transfer. However, these interfaces often constrain the expert's ability to demonstrate optimal behavior due to indirect control, setup restrictions, and hardware safety. For example, a joystick can move a robotic arm only in a 2D plane, even though the robot operates in a higher-dimensional space. As a result, the demonstrations collected by constrained experts lead to suboptimal performance of the learned policies. This raises a key question: Can a robot learn a better policy than the one demonstrated by a constrained expert? We address this by allowing the agent to go beyond direct imitation of expert actions and explore shorter and more efficient trajectories. We use the demonstrations to infer a state-only reward signal that measures task progress, and self-label reward for unknown states using temporal interpolation. Our approach outperforms common imitation learning in both sample efficiency and task completion time. On a real WidowX robotic arm, it completes the task in 11 seconds, 10x faster than behavioral cloning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents LfCD-GRIP, a new way to do imitation learning/learning from demonstration that considers the fact that the input expert demonstrations likely have some amount of "constraint" to them. The proposed approach seeks to learn better policies by using "unconstrained" actions. This is done by estimating the "confidence" in the "proximity" of different states to the goal, finding high-confidence states, and interpolating the proximity of the intermediate states. In this way a more generalized model of proximity can be used to guide the PPO RL loss function, which allows learning to visit states and take actions that may not have shown up in the expert demonstrations. The authors show results in simulation and on a real robot against several baselines, showing faster task performance, particularly in the setting where expert actions were constrained.

### Strengths
Originality
- The approach of identifying high-confidence good states and then interpolating between them is novel to me
Quality
- The paper compares against several baselines and has ablations
Clarity
* The graphs and visualizations are clear
* The paper is generally well-written
* The literature review is appropriately broad and deep for a problem like this 
Significance
- All expert demonstrations are constrained in some way, so this general idea and the problem being tackled has broad applicability across imitation learning.

### Weaknesses
- The contribution of the paper, while novel, is fairly small. Proximity-based approaches are already published work (like the cited Lee et al. 2021). LfCD-GRIP only performs slightly better than the Proximity approach, with the small changes of estimating proximity confidence. Proximity interpolation is done by time, which is the same as Lee et al. 2021.
- There are so many figures in the paper that there is not much room for discussion of results.
* The results are not presented in a way that clearly shows that the proposed approach is better than prior work.
	* Graphs include error bars, but there is no indication of confidence levels or statistical significance.
	* No comparison to Ma et al. 2022, which was listed in the lit review and seems it seeks to solve almost the exact same problem.
- The explanation of the proposed approach is incomplete.
	* Line 251 says "we identify sub-trajectories where both endpoints are high-confidence, and use them as anchors for interpolation", but there is no description of how high-confidence endpoints are determined. I had to go to the code to determine how it is done, and it's not a fixed threshold but more complicated. This is an important part of the approach and should be described in detail.
* The experiments could more clearly illustrate the problem and proposed solution.
	* Most of the examples of a "constrained" expert are simply limiting the magnitude of the possible motion. This seems like an almost trivial constraint to overcome by using any amount of reward shaping during the RL process, like penalizing total time taken. However, I don't see evidence that this was tested against.
	* An experiment where the constraint was relaxed gradually (i.e. increasing the possible action interval in steps) to see that the proposed approach improves as the constraint is relaxed

### Questions
* Exactly how are "high-confident" states determined? I want to make sure I understood the code correctly.
* What is the intuition around how Proximity-Drop performs almost universally worse than Proximity? Did you test LfCD-GRIP without Drop - it seems like it might perform better?
* Conceptually, why wouldn't a technique like Proximity be able to learn to take diagonal actions in MiniGrid?

### Soundness
2

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
This paper proposes a framework for Learning from Constrained Demonstrations, Goal-proximity Reward InterPolation (LfCD-GRIP). LfCD-GRIP extends proximity-based IRL with confidence-guided reward propagation. The authors' formulation of LfCD is interesting, specifically focused on scenarios where the demonstrator is optimal within their constrained action space but where the robot has access to a broader set of actions. The methodology is well-designed and straightforward. The authors test their approach against several baselines and ablations of their proposed framework, and find that LfCD-GRIP has several benefits.

### Strengths
+ The paper has an interesting formulation of LfCD. I could see many applications where a robot may need to learn from LfCD.
+ The paper is well-written and clear.
+ The paper has ample results and a deployment on a real-world robot.

### Weaknesses
- The authors should better explain the difference between their framework and those that learn from suboptimal demonstrations. To my knowledge, works on learning from suboptimal demonstration often have similar motivation and can be applied to the same problem statement posed in this paper. This would also help better highlight the novelty in the proposed work.
- The results have very large standard deviations, and it is unclear whether LfCD-GRIP is actually outperforming other frameworks. A statistical significance analysis and explanation regarding large standard deviations would be beneficial.
- The key result noted in the intro (100 seconds to 12 seconds) seems an overclaim given the large standard deviation in Figure 8.

### Questions
1. Can the authors comment on why SSRR performs so poorly across these domains? From my knowledge of that framework, I cannot see a specific reason on why that framework should underperform GAIL by such a large margin.
2. How does the second term in Equation 2 help avoid overgeneralization?
3. Can you reply to the weaknesses noted above?

### Soundness
3

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
2

### Summary
The paper focuses on the setting where the demonstrator performance is hindered due to constraints in observations/actions.
The paper questions whether it is possible to create a policy, without said constraints, to outperform the constrained demonstrator using the generated demonstrations.
The paper proposes to learn a goal-proximity reward with a confidence-based proximity interpolator, that consists of a confidence estimator and a trajectory-wise interpolator.
The former estimates the confidence through Monte-Carlo dropout and aims to identify reliable observations, and the latter estimates low-confidence observations through interpolating the values between reliable observations.
The paper then conduct experiments on four simulated tasks and a real-world manipulation picking task, and demonstrate that the proposed method identifies out-of-constraint actions that result in shorter trajectories.

### Strengths
- The introduced problem is well motivated, especially in the robotic setting where the demonstrator may have less degree-of-freedoms compared to the actualy embodiment.
- The proposed idea is straightforward and easy-to-follow, and the writing is clear.
- The analyses on MiniGrid environment and out-of-constraint actions are great to demonstrate that the proposed method identifies better actions beyond the demonstrator.

### Weaknesses
I am happy to increase my score after these comments are addressed.
- While the confidence estimation module will provide low variance on "reliable observations", the uncertainty might come from another source, e.g., having multiple demonstrators gathering data.
	- As we scale the number of demonstrators, this might become a degenerate problem where none of the demonstrations contain high-confidence states.
- There are a handful of design/hyperparameter choices that I am unsure if it's well justified/experimented.
	- Decaying strategy for masking
	- What's considered as high confidence?
	- The choice of interpolation strategy
	- There are no sensitivity analysis on any hyperparameters, especially on the introduced modules
		- $\delta$, $K$, etc.
- Nit: It would be great if the paper shows the expert performance on all figures.

### Questions
- About Eq. 5: Is log-scale goal proximity distances the same as $log(r_{end}) - log(r_{start})$?

### Soundness
2

### Presentation
3

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
This paper proposes LfCD, a learning from demonstration method that specializes scenarios where the demonstrators are "constrained" compared to the robot. LfCD extends the IRL framework by 1) using a state-only, learned proximity reward and 2) a mechanism to enhance fidelity of the proximity reward on OOD observations. The core idea of 2) is to provides reward target computed from linearly interpolating confident states to the less confident ones.

### Strengths
+ The paper is well written and presented. The algobox provides a clear overview of the presented approach.
+ The core idea of creating pseudo supervision using confident predictions makes a lot of sense

### Weaknesses
- It seems the presented approach only works in setups where the state space is 1) observable / low-dimensional and 2) smooth. If 1) or 2) does not hold, linear interpolation might not be a proper way to obtain the supervision targets. 
- While the presented approach makes sense, I don't quite see how it relates to the motivation of expert being constrained. The proposed method seems rather generic.
- Similarly, I would like to see how the presented approach improves over the baseline, over various levels of the demonstrator being "constrained".
- I'd like to see a more detailed analysis on the sensitivity of the hyperparameters of the method, such as the confidence/non-confidence threshold, data / compute of the pretraining of the network vs training etc.

### Questions
Similar to the weakness section above. 

- Can the presented approach apply to non-smooth, or high-dimensional state space, or how can one extend the method to make it work on them?
- How does the presented approach improve over the baseline over various "constrained" level of the demonstrator?
- Can we get a more detailed analysis on the sensitivity of the hyperparameters of the method, such as the confidence/non-confidence threshold, data / compute of the pretraining of the network vs training etc.

### Soundness
3

### Presentation
3

### Contribution
3
