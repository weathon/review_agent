# Time Optimal Execution of Action Chunk Policies Beyond Demonstration Speed

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Achieving both speed and accuracy is a central challenge for real-world robot manipulation. While recent imitation learning approaches, including vision-language-action (VLA) models, have achieved remarkable precision and generalization, their execution speed is often limited by slow demonstration via teleoperation and by inference latency. In this work, we introduce a method to accelerate any imitation policy that predicts action chunks, enabling speeds that surpass those of the original demonstration. A naive approach of simply increasing the execution frequency of predicted actions leads to significant state errors and task failure, as it alters the underlying transition dynamics and encounters physical reachability constraints over shorter time horizons. These errors are further amplified by misaligned actions based on outdated robot state when using asynchronous inference to accelerate execution. Our method $\textbf{\textit{RACE}}$ address these challenges with a three-part solution: 1) using desired states as imitation targets instead of commanded actions, 2) replanning the timing of action chunks to execute them as fast as the robot's physical limits allow, and 3) employing a test-time search for an aligned action chunk that maximizes controllability from the current state. Through extensive experiments in both simulation and the real world, we show that our method achieves up to a 4x acceleration over the original policy while maintaining a high success rate

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes RACE, a method that accelerates robot policies beyond expert demonstration speed. RACE uses TOPP-RA to refine the state trajectories proposed by the learned policy, generating new trajectories that satisfy physical constraints. When inference delays occur as in real-world settings, it applies Best-of-N sampling over action chunks to maximize trajectory smoothness. Experiments on Lift, Can, Square, and Tool Hang tasks show that RACE outperforms baselines in both simulation and real-world environments.

### Strengths
1. The paper conducts extensive experiments across four manipulation tasks in both simulation and real-world environments. To evaluate RACE’s robustness and trajectory quality at test-time, the paper designs rich metrics including action smoothness, joint error, and consistency.
2. Real-world demos are provided, showing that RACE performs well at different control rates (15Hz, 30Hz, 45Hz).
3. RACE can accelerate various backbone policies learned by imitation with no additional training in the backbone policy, like diffusion policy and VLA.

### Weaknesses
1. RACE requires the backbone policy to output the entire state trajectory. It can't accelerate policies that only produce single actions or action sequences.
2. When using TOPP-RA, RACE needs accurate knowledge of the robot’s physics constraints, such as maximum torque and velocity.

### Questions
1. Does RACE's performance drop down when the backbone policy does not output the entire trajectory?
2. Several typos in the paper need correction: citation errors in line 46, 82, 85, "chang,e" in line 361,  missing a space in line 320.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
I believe that the three main contributions are primarily established control engineering techniques. The slow execution rate of demonstration data necessary for learning reflects a fundamental limitation of the imitation learning paradigm, which is ideally not addressed by downstream control engineering. If we consider the domains in which these techniques are applicable, it is necessarily quasi-static problems where the dynamics of the task are wholly controllable by the agent. This would exclude eventual dynamic tasks like playing ping pong, throwing and catching, or anything else involving external dynamics not rigidly attached to the robot. In this sense, there is relatively shallow vision aside from the integration of these techniques into VLA systems.

However, control engineering has always yielded reliable sub-systems for autonomous systems and, naturally, current industry engineers relying on VLA systems stand to benefit from learning about this work. Beyond this, it is challenging for me to envision how other researchers may fundamentally build upon what is presented.

### Strengths
1. Demonstrable speed up on door insertion in the real-world
2. Experimentation and result plots are clear and easy to interpret quickly
3. Clear motivation and takeaway

### Weaknesses
On contributions
1. “Imitation from states instead of actions” is effectively learning-from-observation but this area’s related work is not discussed at all. For instance, learning an inverse dynamics model might be a valid and more generalizable alternative to this works’ model-based trajectory optimization.

    * In the first place, if the “action” is not the robot actuation (e.g. motor commands) but some function of state, the learning problem did not originally have actions to begin with. This is an important distinction because almost all robot actions are underactuated relative to state. Hence, this is why modern VLAs rely heavily on controllers to do position control, but the modern language and this work seems to have equivocated delta state as actions. In other words, the first claimed contribution of “actions as state” implicitly existed already in standard VLAs.

    * In the way it is presented, I must assume that the model used for the Time-optimal planning pertains only to the robot (it would be a significantly different and substantial contribution to apply time-optimal planning for exteroceptively sensed states, like a manipulated hammer or ball). This also implies that following controllability and reachability analyses are done only on robot state. However, under this scope, these analyses effectively reduce to optimal control for trajectory tracking. Reachability is often otherwise used in safety analyses for domains such as autonomous driving or social navigation where unsafe and dynamic constraints can be highly non-convex and correlated across uncontrollable states (Bansal et al.).

2. Test-time search and controllability. As noted in the related works, this is very much equivalent to MPC.

3. The work acknowledges that many of the individual techniques exist with a substantially rich history in the related works. I appreciate that the authors do not overclaim the contributions. However, it is perhaps misleading to dedicate such large portions of the manuscript (sections 3.1-3.3) to existing techniques. The authors may save on space by citing references and discussing integration or system-wide challenges rather than the individual methods. For instance, section 3.2 uses state as the time-optimal model parameterization, but states in the context of VLAs and their pertinent tasks are much more than just the robot state.

4. Parts of the paper are not well-proofread and remain unpolished:
   * Caption of figure 7

### Questions
Given the extensive optimal control prior (model-based trajectory optimization), are iLQR or MPPI valid baselines? If so, why not compare them? I surmise by “explicit alignment to physical properties remains less explored”, the authors mean that physical constraints are not often included in MPC and CEM. This is certainly not true considering that MPC has traditionally been performed with models and actuation constraints and is actually often the baseline for model-based control. Here is an open-source project for manipulation with joint-level control under constraints: (Bhardwaj et al.). 


### References

Bansal, S., Chen, M., Herbert, S., & Tomlin, C. J. (2017). Hamilton-Jacobi reachability: A brief overview and recent advances. 2017 IEEE 56th Annual Conference on Decision and Control (CDC), 2242–2253. https://doi.org/10.1109/CDC.2017.8263977

Bhardwaj, M., Sundaralingam, B., Mousavian, A., Ratliff, N. D., Fox, D., Ramos, F., & Boots, B. (2022). STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation. Proceedings of the 5th Conference on Robot Learning, 750–759. https://proceedings.mlr.press/v164/bhardwaj22a.html

### Soundness
4

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
4

### Summary
The authors propose an approach to speed up policy inference at inference time. Their proposed method, RACE, uses states as imitation actions, optimizes for the timing of state chunks, and employs a sampling-based replanning mechanism for asynchronous execution. Experiments are conducted in both sim and real, testing the task success rate v.s. speedup rate. Ablations are done to showcase lower state deviation and higher smoothness of the method.

Overall, the motivation of the paper is clear, the method is sound, and the experiments results are strong. However, I have a few questions/concerns that could be addressed by the authors to further improve the paper. I'm happy to raise my score if the concerns are properly addressed.

### Strengths
* Speeding up policy execution while maintaining performance is an exiciting area of research and has real world impacts
* The paper is written well and easy to follow (besides some minor typos)
* Main experiment results are good, the choice of baseline and tasks are appropriate
* Real world results further strethen the authors' claim about the method

### Weaknesses
* The test-time search part of the algorithm seems to be disconnected from the rest, and the objective to optimize for smoothness seems a little arbitrary. Is this component added just to overcome some issues of asynchronous inference, or does it have some ties with the time-optimal planning (Section 3.2)?

* The authors should also try to explain whether the test-time search component can prevent issues from (Black et al. 2025, Figure 2), where bifurcation between consecutive chunks can exist.

* Following the previous point, there seem to be ablations missing: what is the performance when you remove time-optimal planning and only include test-time search, and vice versa?

* The method section describes multiple design choices and their reasoning (using state vs. actions, using a high-gain PD controller vs. time-optimal planning, naive asynchronous execution vs. test-time search), but they can be improved with examples. It will be beneficial to include visual explanations in addition to the text.

* There are apparent typos, incomplete captions, and citation errors throughout the paper. Here are some examples. The authors should fix these and spend effort double-checking the paper for errors.

  * line 46 citation missing

  * line 47–48 “it imitate” → “it imitates”, “it’s speed” → “its speed”

  * line 85 citation missing

  * line 178 “the controller will high force”

  * line 242 Equation 3.2 is missing

  * Captions for Figure 7(a) and 7(b) are placeholders

### Questions
* How does SAFE scale with the state space of the robot? if the robot is a humanoid, what are some of the considerations to adapt SAFE to it?
* What is the effect of increasing the size of controllable state set? will this result in better time parameterizations?
* What is the point of testing the methods with no inference delay?
* Is there any point of using test-time search when there is no inference delay? 
* Does SAFE work in non-quasistatic environments. For instance, playing air-hockey.

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
3

### Summary
RACE improves the test-time execution speed and control frequencies of action chunk policies. Given a predicted path, RACE uses Time-Optimal Path Parameterization to plan an alternative time-optimal trajectory within given physical constraints. To compensate for the state discrepancies caused by async inference, RACE samples multiple action chunks and selects the one that aligns the best with the current state based on trajectory smoothness, which also benefits time-optimal trajectory replanning. RACE is evaluated in multiple simulation and real-world domains and sees improvement in execution speed with limited degradation in success rate.

### Strengths
- The presentation of the paper is overall good, although some parts lack clarity.
- The proposed method is novel to my knowledge.
- RACE is empirically evaluated to improve over baselines in both simulation and real-world benchmarks, including tasks with high precision requirements that can be challenging to speed up.

### Weaknesses
- A baseline where you directly apply test-time search to action/state fast-forward to smooth the trajectories is missing. This baseline can better contextualize the contributions of both TTS and TOPP. Same for RACE w/o TTS. Figure 7b shows the improvement of TTS in some indirect metrics, but you can also add the evaluation result of RACE w/o TTS to, for example, Table 1.
- Paragraph starting from L183 could use more context and notation explanations.

### Questions
- Is it possible to establish a Pareto front by, for example, changing the constraints in TOPP?

Typos / Minor issues:
- Missing subfigure captions in Figure 7.
- L46, the citation is broken.
- L243, $a$, $b$, $c$ should be bold.
- L247, missing closing parenthesis.
- L361, "chang,e".

### Soundness
3

### Presentation
3

### Contribution
3
