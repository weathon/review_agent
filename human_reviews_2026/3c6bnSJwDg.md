# DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
We study the problem of functional retargeting: learning dexterous manipulation policies to track object states from human hand-object demonstrations. We focus on long-horizon, bimanual tasks with articulated objects, which are challenging due to large action space, spatiotemporal discontinuities, and the embodiment gap between human and robot hands. We propose DexMachina, a novel curriculum-based algorithm: the key idea is to use virtual object controllers with decaying strength: an object is first driven automatically towards its target states, such that the policy can gradually learn to take over under motion and contact guidance. We release a simulation benchmark with a diverse set of tasks and dexterous hands, and show that DexMachina significantly outperforms baseline methods. Our algorithm and benchmark enable a functional comparison for hardware designs, and we present key findings informed by quantitative and qualitative results. With the recent surge in dexterous hand development, we hope this work will provide a useful platform for identifying desirable hardware capabilities and lower the barrier for contributing to future research. Videos and more at \url{project-dexmachina.github.io}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the functional retargeting problem, requiring robots to reproduce human-demonstrated trajectories. A virtual-controller-based curriculum with multiple reward functions is proposed to improve performance. In addition, the authors introduce a new simulation benchmark featuring diverse tasks and objects, through which they analyze the influence of hand size and degrees of freedom (DoF) selection.

### Strengths
1. The virtual-controller-based curriculum is an interesting approach that demonstrates performance improvements across a range of tasks and hand configurations.
2. The analysis of hand size and DoF selection is an interesting direction.

### Weaknesses
1. Scalability of the pipeline: Although the proposed method does not require task- or hand-specific designs, it still appears to rely on object-specific environment setups, which may limit its scalability to large-scale applications.
2. Insight on hand size and DoF: The analysis regarding the impact of hand size and degrees of freedom (DoF) is conducted on a limited number of tasks, which may reduce the strength and generality of the conclusions.
3. Missing baseline: From my perspective, paper [1] also addresses the functional retargeting problem and should be considered as a relevant baseline for comparison.

[1] Liu X, Adalibieke J, Han Q, et al. DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References[C]//The Thirteenth International Conference on Learning Representations.

### Questions
Performance on long-horizon tasks: Since the tasks in ARCTIC should involve more than 300 steps, it would be valuable to evaluate how DexMachina performs on extremely long-horizon scenarios.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
DexMachina studies the function retargeting problem with the goal of transferring human-object manipulation demonstrations to robot action sequences that can manipulate the object accordingly. The core of the method is a virtual object controller and an automatic curriculum learning strategy. The object is at first controlled by a virtual controller whose effects are gradually tuned down along the optimization process. Utilizing VOC, the optimization difficulty is largely decreased, facilitating an easy transfer from human-object kinematic demonstrations to the robot executable control signals.

### Strengths
- Well-motivated method. Introducing the object virtual controller to ease the optimization problem is a well-motivated and interesting idea. 
- Thorough experimental studies. The authors conduct a series of experiments using different types of robot hands to validate the broad effectiveness of the proposed method. A detailed study that compares the functionality of various robot hands is conducted, which also provides valuable insights. 
- Good presentation.

### Weaknesses
- Limited effectiveness. From qualitative results, the effectiveness is only showcased on relatively easy examples with bulky objects. The effectiveness on harder examples, like those involving thin objects with the need to grasp from the table, is not demonstrated (like those shown in dextrack). Besides, the final hand motion is quite unnatural compared to prior works, including maniptrans and dextrack. 
- [Minor for ICLR submission] No real-world experiments. The effectiveness is not validated in the real world. And the real-world applicability is quite questionable, considering its unnatural and jittering hand motions.

### Questions
- How about the efficiency of the whole training pipeline?
- What's the criterion of the success?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a curriculum-based reinforcement learning method that converts a single hand-object demonstration into a bimanual dexterous manipulation policy capable of reproducing the demonstrated trajectory, and showing this paradigm can be used for different robot hands and articulated objects. The key innovation is the use of a virtual PD controllers applied to the object’s 6-DoF joint, which initially guide the object along the human trajectory and gradually decay as learning progresses -- The curriculum reduces PD gains when deque-averaged rewards surpass thresholds, ensuring the final policy operates autonomously. Experiments on ARCTIC datasets involving five articulated objects and six robot hands show that DexMachina outperforms kinematic replay, task-only RL, non-curricular variants, especially on long-horizon tasks. Further analysis also reveals that larger, fully actuated hands learn faster and achieve higher success.

### Strengths
The key insight is interesting -- using the curriculum to stabilize early training in long-horizon, contact-rich tasks via PD control on pose and articulation with reward-gated exponential decay.

The long-horizon performance looks strong qualitatively, and it also outperforms non-curricular baseline to verify the insight

The benchmark is comprehensive, it works across six robot hands, crossing different actuation and size.

### Weaknesses
The policy relies on a single demonstration and must be retrained for each task or object, making it impractical even for retargeting. The benchmark covers few objects and demonstrations, with no explicit tests on unseen objects, demonstrations, or cross-task transfer, only different hand-embodiment study is provided. The approach also assumes high-quality MANO-based hand motion data, which is often unrealistic in real-world capture settings, further limiting its applicability for practical retargeting.

The central idea of using virtual or residual forces to stabilize learning resembles prior motion imitation methods such as Residual Force Control (Yuan, Ye, and Kris Kitani. "Residual force control for agile human behavior imitation and extended motion synthesis." Advances in Neural Information Processing Systems 33 (2020): 21763-21774.)

### Questions
How sensitive are the results to the decay thresholds, reward-deque window sizes, and decay ratios? Do these hyperparameters require extensive tuning or task-specific adjustment to achieve stable performance?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the challenging problem of functional retargeting: training dexterous robot manipulation policies from human demonstrations to track object states. The focus is on long-horizon, bimanual tasks involving articulated objects, which present issues like large action spaces and the embodiment gap. The authors propose DexMachina, a curriculum-based reinforcement learning algorithm that utilizes object-tracking virtual controllers with decaying strength to stabilize learning. DexMachina successfully achieves higher success rates than existing state-of-the-art methods across several complex manipulation tasks in simulation.

### Strengths
1. **High Domain Foresight (Dual-Arm Dexterous Hand)**: The research demonstrates strong foresight by tackling the manipulation of articulated objects using sophisticated dual-arm dexterous hands, addressing a critical and complex area of robot manipulation.

2. **Novel Curriculum-Based RL Framework**: The method introduces a curriculum-based Reinforcement Learning approach that successfully mitigates the inherent instability and local optima issues common in single-framework RL learning, offering a novel and robust learning paradigm.

3. **SOTA Performance in Simulation**: The proposed method, DexMachina, significantly outperforms comparable advanced baselines like ObjDex and ManipTrans across a variety of long- and short-horizon manipulation tasks, positioning it as a competitive new choice in the field.

### Weaknesses
1. **Lack of Real-Robot Validation**: The most significant drawback is the absence of real-robot experiments. Since comparable baselines like ObjDex and ManipTrans include hardware validation to support their claims, DexMachina's reliance solely on simulation limits the persuasiveness of its reported performance leadership.

2. **Reward Function Complexity and Generalization Risk**: The design utilizes numerous auxiliary reward components, which is common in RL but makes the method highly dependent on extensive hyperparameter tuning. This complexity inherently raises doubts about the method's ability to generalize to objects more complex than articulated ones, such as high-dimensional deformable objects (e.g., fabric), without major refactoring.

3. **Simulation Complexity and Scalability**: The core components—bimanual control, dexterous manipulation, and curriculum-based RL—combined with the long-horizon tasks inherently result in a highly complex and computationally demanding training setup, potentially creating a significant barrier to entry for researchers attempting to reproduce the results or scale the method.

### Questions
1. **Hardware Transfer Plan**: What is the specific, proposed plan for transferring DexMachina policies to a physical robot platform, and what additional challenges are anticipated?

2. **Reward Function Ablation**: Can the authors provide an ablation study to quantify the performance sensitivity of DexMachina to the individual components within the complex reward function?

3. **Deformable Generalization**: How can the DexMachina framework be adapted or extended to address manipulation tasks involving high-dimensional deformable objects (e.g., cloth or rope)?

### Soundness
3

### Presentation
3

### Contribution
3
