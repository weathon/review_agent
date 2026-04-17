# Free-View Robot Manipulation: Visuomotor Policy by Calibration Diffusion

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Visuomotor policies have demonstrated great potential in robot manipulation tasks.
However, current robot manipulation tasks are often observed from fixed viewpoints. 
Once the viewpoints change, the trained policy becomes ineffective.
This limitation curbs the generalization of robot manipulation and impedes its application.
To address this issue, we make a comprehensive study by presenting novel free-view manipulation tasks that enables the robot to perform actions from any viewpoint.
Firstly, we construct a free-view dataset, which encompasses 8 tasks with over 5,000 episodes sourced from the ISAAC SIM simulation  environment. 
Each episode records robot manipulation behaviors from different viewpoints.
Secondly, we propose a calibration diffusion policy, which utilizes an additional calibration network to enhance the adaptability of the diffusion policy to different viewpoints. 
In particular, we adopt two-stage curriculum training to make the calibration diffusion policy converge rapidly.
Finally, we conduct a wealth of experiments on the free-view dataset. 
The obtained results demonstrate the effectiveness of the calibration diffusion policy. 
This also means that we have built a new benchmark for free-view manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the free-view robot manipulation task to overcome the fixed viewpoint limitation in current visuomotor policies. The authors construct a new dataset with 8 tasks and over 5,000 simulation episodes, each from varying viewpoints. Their solution is a calibration diffusion policy, which uses a novel calibration network and a two-stage training curriculum to achieve effective, viewpoint-invariant manipulation. The method's success is validated through extensive experiments, establishing a strong new benchmark for this problem.

### Strengths
1. the topic the paper focuses on is an interesting and crucial problem in current visuomotor policy for robot manipulation.
2. the paper is well-written with little typos.
3. the author try to build a benchmark to systematically analyze the free-view robot manipulation problem.

### Weaknesses
1. The contribution of the proposed dataset is undermined by its limited scale and lack of detailed specification. The paper would be strengthened by providing a comprehensive description of the camera pose distribution used for data collection, including the ranges for azimuth, elevation, and distance from which viewpoints were sampled. 
2. The method lacks of specific design for the free-view robot manipulation problem. The author simply add the calibration parameters to the current diffusion policy pipeline without any insight of this problem. 
3. The statement of the two-stage training pipeline is confused.  The purpose of introducing random noise states in the first stage is unclear and requires elaboration. Please clarify the  training strategy with figure 3. 
4. The experimental evaluation lacks critical details regarding the viewpoint split between training and testing. It is essential to clarify whether the tested viewpoints were entirely unseen during training or were simply a held-out set from the same distribution. Please present more details of experiment evaluation.
5. The most significant weaknesses of the paper is lack of real-world experiments. The absence of real-world experiments leaves the method's practicality, robustness to perceptual noise unproven.

### Questions
Please see the weaknesses.

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
This work aims to deal with the viewpoint generalization problem for robotic manipulation. The author constructs a free-view dataset, which encompasses 8 tasks with over 5,000 episodes sourced from the Isaac Sim simulation environment. Then, a calibration diffusion policy training method is introduced to enhance the adaptability of the diffusion policy to different viewpoints. To verify their method, the author does some experiments based on Issac Gym.

### Strengths
1. The method is novel and the problem you concern is important.
2. For simulation, the author introduce well-designed free-view settings.

### Weaknesses
1. No multi-view related method is included as baseline. There have been many methods proposed to deal with viewpoint disturbance setting, like RoboUniview[1], Maniwhere[2], ReViWo[3] or MV-MWM[4]. However, no kind of such baselines is included.

2. We desire a real-world experiment to validate the effectiveness of your method.

3. For real-world tasks, despite using self-collected data, we wonder if some open-source multi-view data could also be used to better improve your model capability.

References:
[1] RoboUniView: Visual-Language Model with Unified View Representation for Robotic Manipulation. 
[2] Learning to Manipulate Anywhere: A Visual Generalizable Framework For Reinforcement Learning.
[3] Learning View-invariant World Models for Visual Robotic Manipulation.
[4] Multi-View Masked World Models for Visual Robotic Manipulation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a Free-View Robot Manipulation framework that enables robots to perform visuomotor manipulation tasks from arbitrary camera viewpoints. It builds a new Free-View Dataset (8 tasks, 5000+ episodes) within Isaac Sim and proposes a Calibration Diffusion Policy (Cali. DP) that integrates camera calibration parameters into diffusion-based visuomotor policy learning. Extensive simulation experiments demonstrate improved generalization to viewpoint variations compared to baselines like ACT, Diffusion Policy, and Flow Policy.

### Strengths
1. The paper addresses an important and underexplored challenge (viewpoint generalization) in visuomotor policy learning, with clear motivation and relevance to real-world deployment.
2. It constructs a well-designed free-view dataset with diverse manipulation tasks and calibration annotations, providing a valuable benchmark for future research.
3. Experimental results are comprehensive, including comparisons with strong baselines, ablation studies, and multiple evaluation settings, demonstrating meaningful performance gains.

### Weaknesses
1. All experiments are conducted purely in simulation (Isaac Sim) without any real-world validation; thus, the claimed generalization to varying viewpoints remains unverified under physical-world noise and imperfections.
2. The proposed Calibration Diffusion Policy shows limited methodological novelty. It mainly adapts ControlNet-like conditioning to diffusion policy rather than introducing a fundamentally new idea.
3. The dataset’s task diversity and complexity are relatively low (no deformable or dual-arm tasks), limiting its generalizability and practical relevance.
4. The paper’s claim that existing methods fail under viewpoint changes is insufficiently supported; it lacks direct comparisons with modern multi-view or hybrid-view frameworks (e.g., OpenVLA, RDT, or $\pi_0$) that already handle varying viewpoints effectively.

### Questions
1. Can the proposed Calibration DP be validated on a real robot to assess robustness under real-world calibration noise?
2. How does the method perform when calibration parameters are inaccurate or unavailable, as often occurs in practice?
3. Could comparisons be extended to recent Vision-Language-Action systems or hybrid-view approaches to strengthen the motivation?
4. Is the calibration network truly necessary, or could similar improvements be achieved with modern feature alignment or camera-pose estimation modules?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new framework for viewpoint-invariant robot manipulation by introducing a calibration diffusion policy. The authors also propose a free-View robot manipulation dataset constructed in the Isaac Sim environment, covering eight manipulation tasks. The method integrates a calibration network into the diffusion policy to adapt to variable camera viewpoints. Extensive experiments and ablation studies are conducted to evaluate the approach against several baselines, including BC-T, ACT, Diffusion Policy, Flow Policy, and DP3.

### Strengths
1. The paper identifies a relevant and underexplored issue, that is the sensitivity of visuomotor policies to camera viewpoints, and formalizes the “free-view manipulation” as a benchmark task. This is novel to me.
2. The Free-View dataset is clearly structured, includes calibration parameters for each episode, and provides an important testbed for evaluating viewpoint generalization.
3. The experiments cover comparisons with multiple strong baselines, data size ablations, structure ablations, and calibration noise analyses. The evaluation protocol is clearly explained. and overall it is generally well organized and includes clear visualizations.

### Weaknesses
1. The entire evaluation is done in simulation of Isaac Sim. There lack of physical robot validation limits the practical credibility of the results.
2. The description of the calibration network could be improved to make it more clear. The method section is difficult to follow, with unclear notations and redundant equations. Key architectural details are missing
3. The method requires accurate calibration parameters between the robot and each camera. As the authors themselves admit in Section A.3, this dependency makes the approach impractical for real-world scenarios, where calibration is noisy or unavailable.

### Questions
1. How is the camera calibration represented in real-world tasks, and could the model generalize to approximate or partially incorrect calibrations?
2. How scalable is the proposed two-stage training strategy when the number of viewpoints or tasks increases significantly?
3. Could the calibration features be learned implicitly, rather than explicitly provided?

### Soundness
3

### Presentation
3

### Contribution
3
