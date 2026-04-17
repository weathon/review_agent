# Ctrl-World: A Controllable Generative World Model for Robot Manipulation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Generalist robot policies can now perform a wide range of manipulation skills, but evaluating and improving their ability with unfamiliar objects and instructions remains a significant challenge. Rigorous evaluation requires a large number of real-world rollouts, while systematic improvement demands additional corrective data with expert labels. Both of these processes are slow, costly, and difficult to scale.
World models offer a promising, scalable alternative by enabling policies to rollout within imagination space. 
However, a key challenge is building a controllable world model that can handle multi-step interactions with generalist robot policies. 
This requires a world model compatible with modern generalist policies by supporting multi-view prediction, fine-grained action control, and consistent long-horizon interactions, which is not achieved by previous works.
In this paper, we make a step forward by introducing a controllable multi-view world model that can be used to evaluate and improve the instruction-following ability of generalist robot policies. Our model maintains long-horizon consistency with a pose-conditioned memory retrieval mechanism and achieves precise action control through frame-level action conditioning. Trained on the DROID dataset (95k trajectories, 564 scenes), our model generates spatially and temporally consistent trajectories under novel scenarios and new camera placements for over 20 seconds. We show that our method can accurately rank policy performance without real-world robot rollouts. Moreover, by synthesizing successful trajectories in imagination and using them for supervised fine-tuning, our approach can improve policy success by 44.7\%. Videos can be found at https://sites.google.com/view/ctrl-world.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Ctrl-World, a controllable multi-view generative world model for robotic manipulation that enables policy-in-the-loop imagination rollouts for both policy evaluation and improvement. By integrating multi-view joint prediction, frame-level action conditioning, and pose-conditioned memory retrieval, the model achieves spatially consistent, controllable, and long-horizon video generation. Trained on the large-scale DROID dataset, Ctrl-World can simulate policy behavior across diverse camera views and environments, achieving close alignment with real-world performance and improving policy success rates by 44.7% through synthetic fine-tuning. Overall, the work offers a significant step toward scalable simulation-based evaluation and data-efficient improvement of generalist robot policies.

### Strengths
- This paper explores an interesting and important problem in the field of robotic manipulation, action-conditioned embodied video generation. Leveraging video generative models to simulate action policies represents a promising research direction, as it enables more efficient and scalable simulation, potentially facilitating the future development of RL in VLA systems.

- The proposed pipeline is well-designed and comprehensive. Through the incorporation of memory mechanisms, multi-view representations, and action-conditioning, it effectively ensures controllability between the generated videos and the corresponding actions.

- The paper also provides extensive real-world experiments. The authors validate the effectiveness of the proposed Ctrl-World framework in simulating policy models across different robotic platforms.

### Weaknesses
- The paper addresses an important problem in action-conditioned embodied video generation for robotic manipulation. However, I notice that similar approaches, such as EVAC [1], have also explored multi-view representations and sparse memory mechanisms. The paper would benefit from a more detailed discussion and comparison with such related work.

- Another concern lies in the simulation fidelity of object dynamics. For computational efficiency, the proposed method employs a relatively compact video diffusion network. It remains unclear whether the generated videos are sufficiently accurate for downstream simulation purposes—especially for deformable objects, where precise modeling of shape deformation is essential, and for articulated objects, where accurate representation of hinge or rotational motion is critical.

- Finally, the practical applicability of the proposed model requires further clarification. It would be valuable to discuss whether the generated data can be directly used for policy training or whether the model itself could function as a video-based simulator to enable scalable reinforcement learning. This aspect is crucial for assessing the long-term impact of the approach, but it is not thoroughly validated in the current version.

- A current limitation is that all demonstrated examples and evaluations focus exclusively on successful policy trajectories, lacking simulation or analysis of failure rollouts. This prevents verification of whether the generated videos truly reflect accurate action–visual mappings. The paper should further compare and analyze cases where, under identical initial states, both successful and failed action policies are executed to assess the model’s fidelity and controllability.

[1] EnerVerse-AC: Envisioning Embodied Environments with Action Condition. arxiv:2505

### Questions
This paper presents an interesting exploration of action-conditioned embodied video generation. My main questions are:

- How does the method differ from existing approaches such as EVAC, which also employ multi-view and sparse-memory designs?

- Can the model accurately generate videos for complex cases like deformable or articulated objects, given its lightweight diffusion architecture?

- Can the generated data be directly used for policy training, or can the model serve as a video-based simulator for scalable RL?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method for building a multi-view world model that supports fine-grained action control and stable long-horizon generation. The authors conduct extensive experiments to verify the effectiveness of Ctrl-World in visual generation compared with baselines, and perform ablation studies to demonstrate the effectiveness of the three proposed techniques. In addition, the authors carry out policy evaluation and synthetic data experiments to validate the effectiveness of Ctrl-World for robot learning.

### Strengths
1. This paper addresses a practical problem and demonstrates the effectiveness of wrist-mounted cameras for robot world models.
2. The authors conduct detailed real-world experiments to verify the effectiveness of using world models for policy evaluation.
3. The synthetic data experiments help expand robotic arm datasets, and the experimental results in the paper show strong potential.

### Weaknesses
1. The technical novelty is relatively weak. Adding a wrist camera for joint prediction is not very challenging, and similar multi-view world model ideas already exist in autonomous driving, but are not discussed here [1].
2. The idea of Frame-level Action Conditioning (lines #185–#191) has already been used in [2][3], but here it is not properly cited or explained, which may confuse the source of the contribution.
3. Some doubts about Section 5.3: policy evaluation should test the same architecture with different checkpoints or slightly different training strategies. The world model should be able to reflect these differences; only then can it replace the real world to evaluate policies during development, instead of evaluating completely different architectures. Of course, testing on more architectures is good, but the key is whether the world model can detect subtle differences in success rates, similar to [4].
4. Some doubts about Section 5.4: the source of performance improvement is unclear — is it from rewritten instructions or from the world model? My main concern is reproducibility and real validity. This experiment is very similar to [5], but lacks detailed discussion and comparison.

[1] Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models. Arxiv.

[2] IRASim: A Fine-Grained World Model for Robot Manipulation. ICCV 2025.

[3] Pre-trained video generative models as world simulators. Arxiv.

[4] SimplerEnv: Simulated Manipulation Policy Evaluation Environments for Real Robot Setups. CoRL 2024.

[5] DREAMGEN: Unlocking Generalization in Robot Learning through Video World Models. CoRL 2025.

### Questions
1. Some papers [1][2] discuss the issue of visual degradation, and their solution is to add small amounts of noise to the conditioning frames to mitigate distribution shift. What is the fundamental difference between that approach and the Pose-conditioned Memory Retrieval Mechanism proposed in this paper? Which method is more effective for addressing visual degradation?

2. The main difference from other action-conditioned world models is the addition of language instructions. This means video generation is jointly constrained by both the text and the action chunks. In this setting, what is the role of the text input? I am curious about the motivation for this design and whether there is strong evidence supporting its necessity.

[1] IRASim: A Fine-Grained World Model for Robot Manipulation. ICCV 2025.

[2] Training Agents Inside of Scalable World Models. Arxiv.

### Soundness
2

### Presentation
3

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
The paper proposes CTRL-WORLD, a controllable, multi-view video-diffusion world model for robot manipulation designed to run policy-in-the-loop. Key components are: (i) joint multi-view prediction (including wrist views), (ii) frame-level action conditioning, and (iii) a pose-conditioned memory retrieval mechanism. The model is initialized from Stable Video Diffusion, fine-tuned on 95k trajectories, 564 scenes, and interacts with $\pi$0/$\pi$0-FAST/$\pi$0.5 policies via an adapter that maps joint-velocity actions to Cartesian poses. The authors report better video metrics than prior single-view world models, correlation between imagination-space and real-world rollouts, and a +44.7% improvement in instruction following for $\pi$0.5 after fine-tuning on curated synthetic rollouts.

### Strengths
1. The paper cleanly adapts a large video diffusion backbone into an action-conditioned, multi-view world model with explicit mechanisms for long-horizon consistency and control. The system is carefully described, including history frames, frame-wise cross-attention with poses/actions, and the memory stride idea.

2. The adapter that converts joint-velocity chunks into future Cartesian poses enables closed-loop interaction with modern VLA policies, which is practically useful.

3. Ablations show significant drops on both computation-based and model-based video metrics. Quantitative tables and qualitative rollouts are informative.

4. The paper demonstrates a concrete synthetic-data workflow, i.e., instruction paraphrasing with initial-state resets, as well as human-preference curation, that improves instruction following of $\pi$0.5 on downstream tasks.

### Weaknesses
1. Both the evaluation labels and the selection of synthetic trajectories rely on human preference judgments; the protocol, rater agreement, and quality controls are unspecified. Because only successful imaginations are kept for fine-tuning, there is a risk of optimistic bias in the reported improvement, especially absent size-matched controls versus random or uncurated synthetic data. 

2. Policies act in joint-velocity space, whereas CTRL-WORLD conditions on Cartesian poses via an MLP + FK adapter. There is no error analysis or ablation isolating the adapter’s contribution. Given that several physics errors (e.g., collisions/sliding) are noted, the adapter may be a major bottleneck that is not rigorously evaluated.

3. The multi-view token concatenation, frame-level conditioning, and memory retrieval are each known ideas. The paper’s novelty is mainly their combination and scale on DROID. Stronger baselines, e.g., extending IRASim/WPE to multi-view, and theoretical insights are absent, the contribution feels incremental. 

4. The text highlights generalization for “over 20 s”, but most quantitative evaluation reports 10 s trajectories. The horizon sensitivity is not analyzed, e.g., the memory stride/length hyperparameters.

### Questions
1. What happens if CTRL-WORLD conditions directly on joint-space action chunks, i.e., with or without poses? Please provide adapter error statistics and an end-to-end ablation isolating its effect on Fig. 7 correlations.

2. Can you extend IRASim/WPE to multi-view and re-run Table 1 to confirm that the gains are not primarily from adding wrist view rather than the proposed mechanisms?

3. Provide results varying rollout horizon (10–30 s) and memory parameters (k, m), showing how PSNR/FVD and instruction-following correlation degrade with horizon.

4. Please include size-matched controls: (a) random uncurated imaginations, (b) paraphrase-only vs. reset-only vs. both, (c) selection via a reward-model instead of humans, and (d) real DROID successes of equal count.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a controllable world model designed for policy-in-the-loop learning, capable of generating multi-view, spatially, and temporally consistent rollouts conditioned on past robot states, observations, and future actions. The model is trained on the DROID dataset and validated in real-robot experiments. The proposed approach reportedly achieves substantial improvements in policy success rate through trajectory synthesis for supervised fine-tuning. Overall I think this is a solid paper with novel idea that using world-model-based data synthesis for policy fine-tuning, which holds strong potential for advancing applications of world models.

### Strengths
1. The idea of controllable world model with correct action conditioning and cross-frame memory retrieval is important for practical world model application, this paper provides a straightforward yet effective solution towards this goal.
2. The presentation of this paper is good. Sections are well-organized, results are delivered clearly with figures and plots. Illustrations are well-made and helpful.
3. The proposed world model can be taken as a "plug-in" strategy for various pre-trained policies, which may have wide applications.
4. The performance of the proposed world model looks great, the frame-level pose condition and memory retrieval mechanism seems effective.
5. The training cost of this world model is relatively small.

### Weaknesses
1. The proposed world model purely focus on the robot motion and takes interaction with other objects in the scene as environmental dynamics, which may straggle with tasks that involves multi objects or complex interactions.
2. Based on the videos, there are still gaps between real rollout and imagined rollout, especially when the end-effector interacts with objects.
3. The demonstrated tasks are still limited to "pick-and-place" style motions, it would be great to see failure cases on more complex robot tasks.

### Questions
1. It will be great to see failure cases for each experiments on the website.
2. Are the objects in the experiments ID or OOD? How does it generate to very novel objects?
3. What is the inference time?

### Soundness
4

### Presentation
4

### Contribution
3
