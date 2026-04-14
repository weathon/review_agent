# Agent-to-Sim: Learning Interactive Behavior Models from Casual Longitudinal Videos

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 8

## Abstract
We present Agent-to-Sim (ATS), a framework for learning interactive behavior models of 3D agents from casual longitudinal video collections. Different from prior works that rely on marker-based tracking and multiview cameras, ATS learns natural behaviors of animal agents non-invasively through video observations recorded over a long time-span (e.g. a month) in a single environment.
Modeling 3D behavior of an agent requires persistent 3D tracking (e.g., knowing which point corresponds to which) over a long time period. To obtain such data, we develop a coarse-to-fine registration method that tracks the agent and the camera over time through a canonical 3D space, resulting in a complete and persistent spacetime 4D representation. We then train a generative model of agent behaviors using paired data of perception and motion of an agent queried from the 4D reconstruction. ATS enables real-to-sim transfer from video recordings of an agent to an interactive behavior simulator. We demonstrate results on animals given monocular RGBD videos captured by a smartphone. Project page: gengshan-y.github.io/agent2sim-www.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes a 4D reconstruction method and environment-aware behavior generation method for 3D agents.  It represents the canonical structures for the scene and agents by MLPs as in NeRF, and represents the time-varying structure with camera pose, root pose and bone transformations at each frame. These structures  are all learned from casually captured videos. After recovering the structures, three diffusion models are trained to generate goal, path and body poses conditioned the 3D feature field of the scene.  

The main contribution of this paper is a complete pipeline to reconstruct and simulate interactive behaviors of 3D agents in a dynamic scene.

### Strengths
I appreciate the careful 4D agent-to-sim pipeline design, including canonical and time varying structure design, the camera localization and the optimization procedure. It enables the improvement of the interactive behavior simulation results as verified in the rendering quality and Interactive behavior prediction. 

1. the simulated behaviors or animations of cats and humans looks realistic and interacts with the scene geometry in a physics-plausible way, no obvious penetration between the agent and the scene geometry in the video demos. 
2. The hierarchical behavior simulation model, from goal, path to body poses, and egocentric perception features in this paper are verified to be effective in improving the accuracy of the behavior prediction.

### Weaknesses
1.  It is not clear how to collect the training data for neural camera localization. Does the camera move along with the agent or fixed during capturing？  If it is fixed, there are only a few camera poses that can be used in the training.  If it can move, that means we already have an algorithm to obtain accurate camera poses for training. 
2. The symbol \sigma is used to represent density in Sec. 3.1 and noise in Sec. 3.3, a kind of misleading when reading the paper.

### Questions
I have some questions  on the paper writing:

1. In eq.8, both \sigma and \epsilon indicate the noise?  since it is mentioned in line 303 that \sigma is a noise value. If it is the case, please clarify: Does \epsilon indicate the variance of the noise? 
2. In line 310,  how large is the volume queried from the 3D feature volume?
3. What is the definition of the generated path? Is it the root translations in the scene？

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a method for learning interactable behaviour of agent from casual video. To this end, the reconstruct the video (agent, scene and observer) in 4D. The representation has canonical structure in 3D for time independent element using NERF and a time varying structure (observer and agent). The paper proposes a coarse to fine registration of agent and environment into a canonical 4D space. This 4D representation further helps in behaviour representation further used in understanding the behaviour of agents and goal conditioned trajectory generation.

### Strengths
The paper shows an original pipeline to reconstruct 4D representation from multiple videos over longer timespan like a month. This is a complicated problem and the method solves this with good qualitative and quantitative results. 
The choice of representation in every stages is carefully thought through and use of the body of knowledge is good. For example, the study by Banani et.al, which shows that large image models have good 3D and viewpoint awareness, is exploited nicely in this paper.
The decoupling of static and dynamic structure and registration is well designed and merged to solve the complicated relationships exist in a long scene with different variation in layout or color over time.

### Weaknesses
Despite the strength, the paper writing is not very good for a reader. Many places the approach is not well understood. In many places argument is placed without much conviction, for example, neural localiser finds more robust than geometric correspondence while being computationally efficient is not well grounded through technical principles or empirical evidences. The ego perception of the world and scene, observer, and past encoding needs more elaboration in writing to make them understand. It is not clear how the path is generated given a goal? is this a shortest path or any other path. More doubts are raised as questions below.

Because the writing, required illustrations, and the flow of thought is bit convoluted, I am tending towards the low score of the paper.

### Questions
1. I assume Equation 1 and 2 are through NERF context. it is not clear to train them, how to get the poses to train them?
2. How to train Equation 2 when agent is in motion. Elaboration is needed.
3. A picture is needed to show the representation of agent. It is not clear what is meant by root and how that is identified from video automatically for any morphology. 
4. Satio et. al. uses SMPL morphology where getting the skinning is understood. Here the agent can be anything. So the skinning part needs more elaboration to understand by the reader.  What is meant by "bones have time varying centers" and how this impact the skinning? Need picture with explanation.
5. What do we mean by set of 3D Gaussian as bones? How they are fitted in video? How they are initialised?
6.  Scene specific neural localiser  regresses the camera poses, how to get the GT poses? In the following paragraph it is written that "to learn neural localiser we first capture a walkthrough video and build a 3D map.." how to build this 3D map? where the poses are coming from? Is this video is the same we are dealing for 4D reconstruction? Lots of confusion here.
7.If the above 3D is in place , we can sample R*, T* and solve Eqn 5. But what is theta there and what is the importance of that parameter? Agent can move while R* can be fixed.  This is not well understood.
8. Dynamic 3DGS also require poses. How to get them?  Equation 5 needs more illustrative explanation.
9. Because the writing of registration section is bit convoluted questions are arising for Behavious representation and Goal generation. Why score based method was needed for generating goal? VAE could be good enough?
10. Scene observer and past encoding is a complex design. When the scene is already encoded as a MLP(Eq 1,2) it is not clear why another encoding? Or am I missing something?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Agent-To-Sim (ATS), a framework for learning models of agent behavior conditioned on the observer (ego-perspective) and the static background scene. The scene and the agent are represented using neural radiance field in a canonical state. Additionally, the dynamic motion of the observer and the agent as well as the agent's body deformation is represented using a sequence of SE(3) poses. The latter is achieved by representing the motion as a set of 3D Gaussians in the agent frame and mapping locations by blend-skinning. The NeRF representations are combined and rendered in the respective poses using ray integration. The dynamic scene reconstruction is obtained from video sequences using bundle-adjustment like optimization, initialized with a trained camera pose regressor. Behavior prediction for the agent is trained using a 3-stage diffusion model which subsequently predicts the goal sequence, the path of the agent, and finally its body pose sequence. The approach is evaluated for camera registration, 4D reconstruction, and behavior prediction accuracy quantitatively on one test video sequence for a cat. It is also compared with previous methods as baselines.

### Strengths
- The proposed approach to combine 4D scene reconstruction and training diffusion models for behavior prediction seems novel and is interesting.
- The paper is mostly well written and is easy to follow.
- The paper presents two contributions: 1. 4D scene reconstruction including body pose estimation of an agent using NeRF-based representations. 2. A hierarchical diffusion model for behavior prediction in the 4D scene representation.
- The related work section provides a compact overview of the state-of-the-art for related fields.
- The experimental evaluation demonstrates improvements over previous methods for 4D scene reconstruction and agent motion prediction. It also demonstrates improvements over several ablations of the method.

### Weaknesses
- The paper is too packed by presenting two contributions at once and by this important details (see below comments) are missing which hinder reproducibility of the approach.
- l. 199ff, it is unclear how the rendering works. How are the NeRF representations of scene and agent combined in detail? Please provide further information either in main paper or supplemental material.
- The paper references directly to several Figures in the appendix which is attached to the main paper without noting that this is supplemental material. Please indicate clearly in the main paper text that they are supplemental material when referencing the figure. Currently, it seems the paper would try to circumvent the page limit. 
- What is the orientation representation for regression in Eq. 5 ?
- It is not clear why there is a need for a joint optimization of poses and representations T and D as indicated in Eq 6, because the video sequence is annotated with camera poses from a different (unspecified) system (l. 225). Please clarify which other method is used to obtain the camera motion trajectory. Why is it necessary to optimize the poses in Eq. 6 ? How would the approach perform without this post-optimization? Please add an ablation study. What is the point of training a neural localizer if the camera poses are known from a different system? 
- How are the agent root and body part poses intiialized for Eq. 6? 
- How are agent and static scene segmented?  
- l. 251ff, Is the training and swapping of beta performed across various scenes (i.e. different types of rooms) or only within specific scenes? 
- l. 263, why was T*=5.6s chosen and how does the performance of the method depend on the horizon length? This should be evaluated as supplemental results.
- Fig. 2, it is not clear what the thin black arrows mean (e.g. between the motion fields and between score map and goal/path). What is the motion field depicting? Where is this output in the proposed method?
- l. 304ff., the note on generalization to different actions seems problematic. The datasets used in the experiments seem small. The biggest dataset (cat) has only 22 training videos of a few minutes each. The anecdotal result pointed to in the supplemental material (Fig. 11) might be a rare example. Please discuss and revise the claims.
- Eq. 11, what is \Psi_s ? Please define.
- l. 314ff, the text mentions trajectories, but the equation only transforms a single pose (\xi^a). Please clarify.
- l. 318ff, please explain why the approach uses the observer trajectory perceived by the agent and not the other way round. Please also provide an evaluation if this makes a difference.  
- l. 360, a segmentation method is mentioned,but it is not stated how it is used.
- l. 401. GT relative camera pose is obtained by some unspecified process. How does it work in detail and how accurate is this process? Please provide this description in the supplemental material.
- l. 428ff / Tables 4 and 5: the motion prediction is conditioned on ground-truth goal and path for the evaluation. However, this should also be evaluated for regressed goal and path to fully evaluate the proposed method without such oracle information.
- How were ground-truth goal and path annotated in the data ?
- It seems the paper lists several datasets (bunny, human, etc), but only shows results for the cat videos.
- If possible, the datasets used in the evaluation should be made publicly available for future comparison with the proposed method. Is it planned to release the datasets publicly ? Please also specify how this data was collected. Are there potential personal data protection issues for releasing the data?
- The datasets used for evaluation are rather small and, hence, results seem anecdotal and significance is unclear. The method should be evaluated quantitatively on more and longer sequences and a larger variety of agents (e.g., 10-20 different animals and human subjects) and scenes (e.g. 10-20 different rooms) to increase significance of results and to provide further insights into the performance and limitations of the proposed method. Also the variety of daily activities should be increased and annotated. Cultural and gender diversity of subjects should also be considered when designing a benchmark dataset. With larger datasets, also the dependence of the performance of the method on the amount of available training data could be evaluated.  

Further minor comments:
- p. 2, please add a label/caption to the lower figure and reference it from the text.
- l. 76ff, the notation for \sigma, c, \psi and the variants c_s/a, \sigma_s/a, and \psi_s/a is not accurate. Are the subscripts referencing parts of the variables without subscript? What kind of mathematical structure are they? 
- l. 210 "buil" => "build"
- Fig. 2 caption "fully" => "full"
- Eq. 11, "X_w" should be "X^w"
- l. 369, the reference to Eq 5 should point to Eq 6

### Questions
- Please address the questions raised in paper weaknesses in the rebuttal. Please address the lacking details of the method description and the raised shortcomings of the evaluation.

### Soundness
2

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
This paper proposes a framework called Agent-to-Sim to learn the interactive behaviors of 3D agents in a 3D environment from casually captured videos. Specifically, a coarse-to-fine registration method is developed for persistent and complete 4D representations. A generative model of agent behaviors of agent behaviors is trained to enable the generation of agent's reactions to the observer's motions. Interesting demos show the effectiveness of the proposed system.

### Strengths
1. This paper proposes a framework to learn interactive behaviors of agents in 3D worlds, which I believe is an interesting and promising direction. With the rise of embodied AI, this paper offers a novel approach to leveraging real-world videos for understanding interactions between agents and their environment.

2. This paper is clearly written and well-presented, with an impressive demo that effectively showcases the benefits of the proposed framework.

3. The proposed methods are logical for building the entire system and have been thoroughly validated through experiments.

### Weaknesses
1. In terms of behavior generation, there is extensive existing work on generating realistic human motions based on intentions or trajectories, with most approaches utilizing parametric models like SMPL. Given that the agent in the demo appears somewhat unrealistic, I was curious why parametric models weren't used for agent representation. For example, there are parametric models available for animals, such as [1,2].

[1] Zuffi et al.: 3D Menagerie: Modeling the 3D Shape and Pose of Animals, CVPR 2017

[2] Zuffi et al.: Lions and Tigers and Bears: Capturing Non-Rigid, 3D, Articulated Shape from Images, CVPR 2018

2. Regarding the egocentric perception: is the egocentric visual/video data encoded as well? It is mentioned "a latent representation from a
local feature volume around the agent", but an agent like a cat can not see the scene behind it. Besides, the proposed egocentric encoding to transform the world to the egocentric coordinates to avoid over-fitting as well as past encoding to capture the past motion sequence seem to have been proposed and well discussed in [3], which also predicts the motions from spatial controls.

[3] Jiang et al.: EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere, ECCV 2024

3. The proposed system is trained and tested on the collected dataset. However, based on the information provided in the paper, this dataset appears quite small and lacks diversity, as it includes only 4 agents, 3 scenes, and has a limited duration. Given the availability of numerous large datasets featuring moving agents and varied environments, incorporating these existing datasets for evaluation would strengthen the findings. Additionally, I am curious about the generalization capability of the proposed method, as the experiments were conducted by training on only 22 videos and testing on a single remaining video. Do the training and testing data contain the same scenes? If so, this might lead to potential overfitting issues.

4. I am interested in the potential applications of the proposed systems beyond the general areas mentioned in the paper, such as driving, gaming, and movies. It would be valuable to explore specific existing tasks to demonstrate how the system can provide practical benefits.

### Questions
See above

### Soundness
3

### Presentation
4

### Contribution
3
