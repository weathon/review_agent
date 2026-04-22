# DexNDM: Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-Wise Neural Dynamics Model

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 8

## Abstract
Achieving generalized in-hand object rotation remains a significant challenge in robotics, largely due to the difficulty of transferring policies from simulation to the real world. The complex, contact-rich dynamics of dexterous manipulation create a "reality gap" that has limited prior work to constrained scenarios involving simple geometries, limited object sizes and aspect ratios, constrained wrist poses, or customized hands. We address this sim-to-real challenge with a novel framework that enables a single policy, trained in simulation, to generalize to a wide variety of objects and conditions in the real world. The core of our method is a joint-wise dynamics model that learns to bridge the reality gap by effectively fitting limited amount of real-world collected data and then adapting the sim policy’s actions accordingly.  The model is highly data‑efficient and generalizable across different whole‑hand interaction distributions by factorizing dynamics across joints, compressing system-wide influences into low‑dimensional variables, and learning each joint’s evolution from its own dynamic profile, implicitly capturing these net effects. We pair this with a fully autonomous data collection strategy that gathers diverse, real-world interaction data with minimal human intervention. Our complete pipeline demonstrates unprecedented generality: a single policy successfully rotates challenging objects with complex shapes (*e.g.*, animals), high aspect ratios (up to 5.33), and small sizes, all while handling diverse wrist orientations and rotation axes. Comprehensive real-world evaluations and a teleoperation application for complex tasks validate the effectiveness and robustness of our approach. Website: [DexNDM](https://meowuu7.github.io/DexNDM/).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a neural joint dynamics alignment approach to address the sim-to-real challenge in dexterous manipulation. The authors first train a neural joint dynamics model on real-world data, then use this learned model to train a residual policy on top of a simulation-pretrained policy. The goal is to ensure that the predicted future hand state under the learned real-world dynamics matches that observed in simulation. The approach is evaluated on several rotation-style manipulation tasks, showing some empirical improvement.

### Strengths
The engineering effort to reproduce rotation manipulation tasks on a low-cost robotic hand is commendable.

The experimental section is relatively thorough, demonstrating multiple tasks and ablations.

### Weaknesses
**1. Soundness**. The main weakness of this paper lies in the soundness and validity of the proposed method. Conceptually, the approach overlooks several key aspects of manipulation dynamics: 

1A. Object-dependent dynamics. Unlike locomotion on a rigid terrain, where the dynamics primarily depend on the agent itself, manipulation dynamics are determined by both the hand and the object. The proposed neural joint dynamics model cannot predict finger–object contact events when the past frames correspond to free-hand motion. Because the model relies solely on proprioceptive input, it is impossible to infer or reconstruct the object’s geometry and contact state from proprioception (think about parallel gripper gripping a small marble-- you can only know there is something between the fingers. You can not tell the exact object location without touch).

1B. Based on 1A, we can further show that this method can be wrong in simplest scenarios. See comments in 4.

1C. Limited need for learned joint dynamics. For commercial robot hands driven by high-quality servo motors, the joint behavior is typically very repeatable (this is unlike humanoid robots, whose joint dynamics is more complex due to whole-body coupling). In such cases, traditional system identification (SysID) methods are often sufficient to close the sim-to-real gap in actuation dynamics. The authors should include SysID baseline results to determine whether their learned model actually provides any improvement.

1D. Unconvincing data transfer across object scales.
It is unclear why training on large spherical objects would generalize to manipulating much smaller objects. This data choice is questionable, since the resulting joint states for small-object manipulation are likely out of distribution. The claimed transferability is therefore unconvincing.

**2. Novelty.**

2A. The conceptual novelty of this work appears limited. Learning a neural joint dynamics model has already been explored in previous research, including for humanoid and manipulation systems.

2B. The idea of distilling multitask experience into a single model has been investigated by prior work such as DexterityGen, which also demonstrated teleoperation capabilities under a very similar setup. The authors also refer to their performance as "unprecedented" while this is not true. These related works are not cited or discussed adequately.

**3. Theoretical section**

The theoretical discussion in this paper is weak and disconnected from the proposed method. Theorem 3.1 is a well-known result in information theory. Theorem 3.2 is trivial because if $B\subset A$, then for any function or functional $h$, $\sup_{x\in B} h(x)\leq \sup_{x\in A} h(x)$. In the theorem, $A=[f],B=[f\circ g_X]$ and the authors did not make any assumptions on f. If all the $f, g_X$ are continuous (which is natural as the authors use neural networks), then $B\subset A$ and we are done with $\leq$ part.  The value of  showing $<$ is marginal, a nontrivial result typically requires $\gamma$ contraction ($\gamma < 1$).

Besides, these results appear to have little relevance to the main technical contribution and seem to have been added without meaningful connection. A more appropriate analysis would examine error propagation or model mismatch under dynamics error, similar to what has been done in model-based reinforcement learning literature. For instance, the authors should explicitly present the return bound relating $J_\pi$ and $J_\pi^{res}$ ($J_\pi$ is sum of reward when rolling out $\pi$), we should expect some form of $ J_\pi^{res} \geq J_\pi - C(\epsilon)$, where $\epsilon$ is the error of the model.

**4. Conceptual counterexample**

The proposed approach can fail even in simple cases. Consider a minimal manipulation world model with two parts of the state $s=(s_o,s_h)$: the object state $s_o$ (which can take values 0, -1, or 1, representing neutral, fail, and success) and the hand state $s_h$ (which can be 0 or 1). Together there are six possible world states. We have 2 actions, a=0 and a=1.

**In simulation, action a=1 always succeeds**. Specifically,

(0, 0):  a=0 -> (0, 1), receive reward 0.

(0, 0):  a=1 ->  (1, 1), receive reward 1, terminate.

(0, 1):  a=0 -> (0, 1), receive reward 0.

(0, 1):  a=1 -> (1, 1), receive reward 1, terminate.

Clearly, the best policy is to execute A=1 at any state. So, $\pi^*(s)=\pi_{sim}(s) = 1$.

**In the real world, action a=1 may immediately cause failure**:

(0, 0): a=0  -> (0, 1), receive reward 0.

(0, 0): a=1   ->  (-1, 1). **Fail**, receive reward -100, terminate. (e.g. an abrupt action in real world at initialization can lead to failure)

(0, 1): a=0   ->  (0, 1), receive reward 0.

(0, 1): a=1   -> (1, 1).   receive reward 1, terminate.

The best policy in real world takes a=0 at (0, 0), and then a=1 at (0, 1). The optimal simulation policy $\pi^*$ will run into (-1, 1) and receive -100 reward at the first step in this real world environment.

Now, we examine the proposed approach. We first train a neural joint dynamics model $F(s_h, a)=s_h’$ which only considers the dynamics of $s_h$. According to the real world data above, we have $F(0, a) = 1, F(1, a) = 1$ for $a= 0,1$. Then according to L291, we further train a residual policy $\pi_r$ so that $F(s_h, \pi_r(s)+\pi^* (s))=s’_h$ given any $(s, s’)$ drawn from the optimal simulation policy $\pi^*$'s trajectory distribution. Since the simulation policy’s trajectory is (0, 0): a=1 -> (1, 1), the requirement is $F(0, \pi_r(0)+1)=1$. Since $F(0, 1)=1$, $\pi_r(0)= 0$ can work and no correction is required. As a result, the corrected policy still outputs 1 and runs into failure state (-1, 1) at the first step.

This simple example shows that without explicitly modeling object-dependent dynamics, the proposed residual adaptation cannot reliably bridge the sim-to-real gap.

### Questions
See above.

I believe this paper would be significantly stronger if the authors removed the unnecessary (and wrong) theoretical and adaptation components, and instead focused on genuinely extending the capabilities of the considered skills while clearly presenting what works and what does not. These "novelties" feel very forced.

I appreciate the engineering effort and the demo. However, these results have been demonstrated by several previous works, and the authors did not show any significantly new capability (e.g., precisely rotate the object to a specific pose and stop). Therefore, given that the science part of this paper is also flawed, this paper can not be accepted. **A demonstration is only meaningful if it showcases substantially new capabilities or is supported by sound, novel technical contributions.** Personally speaking, I believe a strong paper in this field should emphasize capability breakthroughs. Since reproducing an existing robotic system is inherently challenging and involves many compounding factors (what if you remove the tape from your fingertip and object? what if you tune SysID better? what if your ROS interface is slightly less laggy?), it becomes difficult to clearly quantify the effectiveness of an additional component integrated into an already functional setup. Novel capabilities are definitely more measurable and visible. I encourage the authors to continue their solid engineering efforts and submit an improved version to a future robotics conference.

### Soundness
1

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper DexNDM proposes a neural sim2real framework for dexterous in-hand rotation. Unlike prior methods limited to simple geometries or constrained wrist poses, DexNDM enables a single policy to generalize across diverse objects, wrist orientations, and rotation axes. The method introduces two key innovations: (i) joint level dynamics to bridge sim2real gap and (ii) an reset-free, scalable data collection strategy (“Chaos Box”) that applies randomized loads. A residual policy is trained atop this learned model to adapt the simulation-trained base policy to real hardware. Experiments on the LEAP hand demonstrate impressive real-world in-air rotation of complex, small, and high-aspect-ratio objects across multiple wrist poses, outperforming strong baselines such as AnyRotate, Visual Dexterity, ASAP, and UAN, with additional applications in teleoperation tasks. This paper is solid in methodology, theory, and real-world experiment, shown a possible paradigm for dextrous manipulation.

### Strengths
1. Novel Sim-to-Real Framework. The introduced joint-wise neural dynamics model is novel in sim-to-real transfer for dexterous manipulation. By decomposing system dynamics into per-joint models, the method avoids reliance on noisy object-state estimation and achieves data efficiency through information contraction formalized via KL-divergence reduction

2. Autonomous and Scalable Real-World Data Collection. This idea is really creative, the “Chaos Box” procedure autonomously generates rich, diverse contact data using randomized loads without manual resets, which significantly mitigating the challenge of efficiently obtaining distributionally relevant real-world trajectory for dynamics model learning.

3. Comprehensive Empirical Validation.
Extensive experiments cover simulation, real-world, and cross-simulator settings, and the provided demo videos show impressive performance of rotating objects from small to large aspect ratio.

4. Strong Theoretical Underpinning.
Theoretical analysis  rigorously argues that the joint-wise projection reduces domain divergence (KL contraction) and narrows the generalization gap, supporting the empirical findings.

### Weaknesses
1. Limited Hardware Validation. While the method demonstrates strong performance on the LEAP hand, its hardware generality remains untested. The paper would be strengthened by deploying DexNDM on additional robotic hands (e.g., Allegro, XHand) to verify whether the joint-wise dynamics model and residual adaptation strategy transfer effectively across different actuation mechanisms and kinematic structures.

2. Scope of Generalization Remains Task-Specific. Although the model generalizes across object shapes and orientations, it is unclear whether the learned dynamics transfer to different manipulation tasks (e.g., regrasping, non-rotational dexterity, tool using).

### Questions
1. Sensitivity to Noise and Load Distribution. How robust is the dynamics model learning to variations in noise magnitude and randomized load distributions within the Chaos Box setup?

2. Performance–object weight Relationship. In the real-world experiments, could the authors report the physical weight of each test object and plot a performance curve (x-axis: object weight, y-axis: rotation performance) to examine whether and how DexNDM’s effectiveness degrades as object weight increases?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces DexNDM, a sim-to-real pipeline for dexterous in-hand rotation. A specialist-to-generalist policy is trained in sim, then adapted to real via a joint-wise neural dynamics model learned from fully autonomous “Chaos Box” data that does not require human resets. A residual policy uses the learned dynamics to correct the base policy’s actions, yielding strong real-world rotation across diverse objects, high aspect ratios, and varied wrist orientations—substantially outperforming strong baselines including AnyRotate and matching/surpassing Visual Dexterity on shared items.

### Strengths
The paper presents a novel sim-to-real design with a distinctive joint-wise dynamics perspective and extensive hardware evaluation. Additional strengths include:

i) Clear core idea with theoretical support. Factorizing into joint-wise dynamics (per-joint prediction from its own history) is argued to contract domain shift and improve generalization, with formal statements (e.g., Theorem 3.2) backing the information-contraction view.

ii) Practical, scalable data collection. The autonomous “Chaos Box” setup (soft-ball container, open-loop action replays, small Gaussian action noise) yields diverse, reset-free real data that are cheap and safe to collect.

iii) Consistent real-world gains. Across multi-axis and multi-wrist settings, DexNDM improves radians rotated and time-to-fall over direct transfer and whole-hand dynamics baselines.

### Weaknesses
Overall, the writing is clear, the presentation is strong, and the proposed methods are both promising and interesting. I have a few suggestions:

i) Tighten the narrative. The paper tries to cover a lot of useful details; consider trimming or re-structuring to keep the main storyline crisp.

ii) Clarify L207 (“works well on hardware”). Do you mean the distilled base policy from the specialized oracles already works on hardware, or only after adding the learned residual policy? If it’s the former, what is the algorithmic role of the neural dynamics/sim-to-real transfer (i.e., the main novelty of this work)? If it’s the latter, please state that explicitly and explain how the transfer enables generalization to more complex objects (shape, rotation axes).

iii) Data coverage from success-only rollouts. If oracle rollouts are all successful, the dataset likely underrepresents near-failure states and local corrections. Did you observe failures of this kind, and how were they handled? If not, did you apply simple augmentations (e.g., noise injection, reset-to-perturbed states, small pose/force perturbations) to broaden the distribution? 

iv) Terminology on “randomized loads.” In autonomous data collection the hand is placed in a container of soft balls; “randomized loads” may be misleading. Use a more precise term (e.g., “randomized contact interactions” or “stochastic contact conditions”) to reflect the actual setup.

v) Residual policy section needs intuition. Could the trained base policy, combined with the learned joint-wise dynamics model, be used directly—i.e., update the base policy to produce slightly adjusted actions—rather than training a separate residual policy? More broadly, the procedure resembles system identification: the policy operates through a dynamics model trained on real data. Please expand the intuition in “Bridging the Dynamics Gap via a Residual Policy”: clarify the role of the neural dynamics model—especially since it is not used at real-world execution time—and explain why training the residual policy through this model helps overcome the sim-to-real gap.

vi) Object-agnostic design and task specificity. The base policy, dynamics model, and residual policy are all object-agnostic. While I accept the claim that complex hand–object interactions can be inferred from observation history, this effectively makes the connection task-specific: without explicit object conditioning, contact dynamics are largely memorized via state transitions. From an application angle, integrating teleop is promising; algorithmically, however, this limits generalization to other type of hand-object interactions. This is common in recent in-hand reorientation work (i.e., the Leap hand paper), so it’s not a primary weakness here—but I’d like to hear the authors’ perspective and any plans to incorporate object-aware conditioning or descriptors.

### Questions
See Weaknesses.

I would like to increase my score if they are well explained / addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes DexNDM, a novel sim-to-real framework for dexterous in-hand object rotation. The core component is a joint-wise neural dynamics model (NDM) that factorizes the system dynamics across individual joints instead of modeling the entire system, using the data collected in the real world. To collect these data without human supervision, the authors design an autonomous data collection where the robot hand interacts with random soft loads to generate diverse, object-loaded contact experiences. The learned dynamics model is then used to train a residual policy that adapts a simulation-trained base policy to real-world physics, closing the sim-to-real gap. The authors show extensive experiments on the LEAP hand demonstrate the ability to rotate a wide variety of objects under diverse wrist orientations and rotation axes. DexNDM significantly outperforms strong baselines such as AnyRotate, Visual Dexterity, ASAP, and UAN. The system also enables teleoperation applications for tool use and assembly, showcasing versatility beyond single-object rotation.

### Strengths
(+) The joint-wise neural dynamics model is conceptually elegant and practically effective. It addresses dynamics mismatch by factorizing high-dimensional motion into independent low-dimensional predictions, with a solid theoretical rationale for generalization through information contraction.

(+) The “Chaos Box” design for collecting diverse contact data is simple yet effective. It removes human intervention and provides a realistic, high-coverage dataset for learning dynamics models in dexterous manipulation.

(+) The system achieves in-air rotation of challenging objects across multiple wrist orientations, as well as detailed ablation studies that validate each key design choice.

(+) Applications beyond benchmark tasks: The demonstration of teleoperated long-horizon tasks (e.g., using tools and assembling furniture) highlights the real-world relevance and generality of the learned policy.

### Weaknesses
(-) Complexity of presentation: The paper packs many contributions (specialist-to-generalist policy, joint-wise model, residual policy, and autonomous data collection). While each is justified, the narrative can be dense, and the hierarchy between contributions could be clarified further.

(-) The approach relies mainly on proprioception and does not yet leverage tactile or vision-based feedback for joint-wise modeling. This may limit performance in highly ambiguous contact configurations.

(-) The project website should more clearly indicate which motions in the teleoperation demos are executed by the learned policy versus human guidance, as this distinction affects perceived autonomy.

(-) While the joint-wise model improves data efficiency, training multiple per-joint networks and the residual policy could be resource-intensive, and runtime performance scaling is not discussed.

### Questions
How sensitive is the performance to the number of joints modeled jointly (e.g., finger-wise vs. joint-wise factorization)?

How long does the autonomous data collection (“Chaos Box”) take to gather sufficient data, and how does this compare to task-aware data collection in wall-clock time?

For the teleoperation tasks, are the actions purely executed by the learned policy after human high-level input, or is there continuous human feedback control?

How transferable is the joint-wise model to other hands with different kinematics or actuation? Would retraining be required, or could it generalize zero-shot with minimal fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3
