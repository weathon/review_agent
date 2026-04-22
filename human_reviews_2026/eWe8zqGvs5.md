# Cortical Policy: A Dual-Stream View Transformer for Robotic Manipulation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
View transformers process multi-view observations to predict actions and have shown impressive performance in robotic manipulation. Existing methods typically extract static visual representations in a view-specific manner, leading to inadequate 3D spatial reasoning ability and a lack of dynamic adaptation. Taking inspiration from how the human brain integrates static and dynamic views to address these challenges, we propose Cortical Policy, a novel dual-stream view transformer for robotic manipulation that jointly reasons from static-view and dynamic-view streams. The static-view stream enhances spatial understanding by aligning features of geometrically consistent keypoints extracted from a pretrained 3D foundation model. The dynamic-view stream achieves adaptive adjustment through position-aware pretraining of an egocentric gaze estimation model, computationally replicating the human cortical dorsal pathway. Subsequently, the complementary view representations of both streams are integrated to determine the final actions, enabling the model to handle spatially-complex and dynamically-changing tasks under language conditions. Empirical evaluations on RLBench, the challenging COLOSSEUM benchmark, and real-world tasks demonstrate that Cortical Policy outperforms state-of-the-art baselines substantially, validating the superiority of dual-stream design for visuomotor control. Our cortex-inspired framework offers a fresh perspective for robotic manipulation and holds potential for broader application in vision-based robot control.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To address the issue that static-view observations in robotic 3D imitation learning cannot fully capture dynamic spatial information, and inspired by the dual-stream mechanism of spatial perception in the human cerebral cortex, this paper proposes Cortical Policy — a method based on view transformers that jointly encodes static 3D observations and dynamic world observations.

The model enforces geometric consistency alignment between the two observation streams in the feature space, thereby enhancing the understanding of the physical geometry of the world and improving the action generation capability of the policy model.

The authors first analyze the cortical principles of human visual-motor control, pointing out that in addition to the ventral stream responsible for reasoning about spatial scenes, the dorsal stream for dynamic understanding is equally important and mutually complementary.

Inspired by this, they argue that 3D imitation learning should also incorporate a dynamic observation stream to enhance dynamic spatial perception.

For the static stream, the authors introduce a cross-view consistency keypoint prediction mechanism, using VGGT to predict shared keypoints across multiple viewpoints and supervising the model to align representations of these keypoints across different views, thereby achieving cross-view geometric consistency.

For the dynamic stream, they propose a dynamic prediction mechanism for the GLC model, enabling it to accurately predict the end-effector pose after an action is executed. These predictions provide valuable features for subsequent policy learning.

Through feature extraction and fusion from both streams, the proposed model achieves significant performance improvements on the RVT-2 backbone and attains state-of-the-art results on both simulation and real-robot tasks.

### Strengths
1.	The paper introduces a dual-stream observation mechanism inspired by the human brain, which substantially improves the model’s ability to perceive dynamic changes in the environment during task execution and addresses limitations of previous work.
2.	Both static and dynamic streams emphasize cross-view feature alignment, and an efficient GLC-based dynamic prediction mechanism is designed to guide policy action generation.
3.	The proposed dynamic perception mechanism has strong potential for mobile manipulation and complex dynamic scene tasks, suggesting promising directions for future research.

### Weaknesses
1.	In the training of the static-view stream, the proposed cross-view alignment mechanism aims to unify geometric features of keypoints across viewpoints. However, it remains unclear whether this alignment may interfere with the encoder’s original spatial semantic understanding — which is critical for the static stream. This issue warrants further investigation.
2.	The paper does not evaluate the model’s generalization ability on new or unseen tasks (zero-shot transfer), leaving the claimed “cortical” system’s abstract reasoning capacity unsubstantiated.

### Questions
The tasks chosen in the RLBench simulator mostly rely on static observations to achieve good performance. Would it be possible to evaluate the proposed model on more challenging or dynamic simulation environments to better highlight the contribution of the dynamic stream?

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
4

### Summary
The paper proposes Cortical Policy, a dual-stream view transformer for robotic manipulation inspired by the ventral (static) and dorsal (dynamic) visual pathways. The static-view stream enforces cross-view geometric consistency using supervision from a 3D foundation model to improve 3D spatial reasoning; the dynamic-view stream uses a wrist/egocentric camera and a position-aware, gaze-estimation backbone to guide action with motion cues. The two streams are fused in an action head to predict 6-DoF gripper poses, state, and collision flags. Experiments are conducted on RLBench and several real-robot tasks. Authors report that the approach outperforms recent view-transformer baselines and that ablations suggest benefits from both the geometric-consistency loss and the dynamic stream.

### Strengths
1. The paper specifies supervision generation, the SmoothAP-based cyclic consistency loss $L_{cgc}$, the dynamic rendering/pretraining pipeline, and shows ablations isolating architecture, pretraining, and heatmaps.
2. On RLBench, authors claim higher average success than RVT-2 and improved performance on spatial-reasoning and dynamic scenarios; they also include small-scale real-robot tests.

### Weaknesses
1. Despite an elaborate pipeline, the trajectory-learning advantage may be modest. Even in the authors’ table, some tasks see limited gains or regressions, raising the question of whether the architectural complexity is justified by the overall deltas.
2. It’s not yet conclusive that the dorsal (dynamic) stream is the key driver of improvement; ablations show mixed patterns, and the net gain over a strong static baseline can be small.

### Questions
Why adopt the dual ventral/static and dorsal/dynamic streams with VGGT cross-view geometric consistency and gaze/pose priors? Can you provide capacity- and compute-controlled ablations and stress tests showing each component is necessary and yields independent gains (not just capacity or pretraining effects)?

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
3

### Summary
Visual robotic manipulation in unstructured environments is challenging, in part, due to poor spatial understanding in 2D RGB image encoders. Recent works improve performance of visual imitation learning policies by using either (1) an explicit 3D representation which is compute-intensive but effective, or (2) fusing features from multiple static camera-views in an implicit manner (e.g. view transformers) which has been shown to perform comparably to explicit representations but without the added computational costs. However, this leads to two key failure modes: inadequate spatial reasoning and dynamic adaptation failure. Inspired by human ventral (static) and dorsal (dynamic) visual pathways, the paper proposes Cortical Policy, a dual-stream view transformer that first encodes static and dynamic view information using two separate streams, and then fuses their respective features before making an action prediction. Beyond this dual-stream architecture, the paper also incorporates several representation learning techniques including a feature/view auxiliary loss that aims to improve 3D keypoint (produced by a 3D foundation model) consistency across views, as well as a pretrained visual backbone for dynamic view feature extraction (initialized from GLC, a human gaze estimation model). Experiments are conducted on RLBench and a real hardware setup, and experiments indicate that the proposed method (Cortical Policy) performs better than RVT-2, a strong view transformer baseline that only uses static views, upon which the proposed method is implemented.

### Strengths
- I believe that this paper studies a relevant and timely problem (imitation learning from multiple RGB cameras, and in this case a combination of static and dynamic views), and is likely to be of interest to the community. The problem is clearly defined and I believe that the shortcomings of prior work is described in enough detail for an unfamiliar reader to appreciate the technical contributions. The paper is generally well written and easy to follow throughout, although the method section is a bit verbose.
- Simulation experiments are conducted on a variety of tasks from RLBench, which has become a common benchmark for imitation learning algorithms for robotic manipulation, particularly works that focus on multi-view policy learning. The authors compare their proposed method against a series of strong baselines, including RVT-2 upon which the proposed method is implemented. Benchmark results indicate that the proposed method, on average, performs better than this set of baselines across RLBench tasks, and also generalizes better to unseen scene configurations in a cube stacking task on real hardware.
- The ablations in Table 2 are helpful for understanding the relative importance of the proposed changes. This is rather important as the proposed method can be viewed as a combination of different architecture, objective, and modality changes on top of RVT-2.

### Weaknesses
My initial assessment of the paper is fairly neutral. I believe that the paper and contributions are interesting, but I also do have some concerns that I would like the authors to address:

- Since this paper appears to follow the PerAct experimental setup, I was a bit surprised to not see PerAct listed as a baseline. While I understand that this work focuses on implicit view fusion rather than the explicit 3D representation of PerAct, I do believe that the comparison would be useful to readers (it currently is not clear how they compare in terms of task performance).
- The proposed method is rather complex and contains multiple new design choices on top of RVT-2. However, the ablations in Table 2 indicate that the majority of the performance improvements stem from the auxiliary feature/view consistency loss (RVT-2: 77.5%, +consistency loss: 80.1%, +everything else: 81.0%), which (to me) seems inconsistent with the paper's main claim that a dual-stream view transformer is the key to implicit 3D spatial understanding. I would appreciate if the authors can please clarify whether my understanding of this is correct.
- The real robot experiment only considers a single task but with several distinct scene configurations such as object displacement. I believe the real robot results would be more convincing if it contained more task variety like in the simulated tasks. Additionally, it is not clear from the paper whether a single agent is trained and evaluated on all four task variations in a multi-task setting like in simulation, or if it is four separate agents (from Appendix B.2: *"Each task collects 45 human-teleoperated demonstrations with placement variations"* is the only information I could find on this, and it is a bit ambiguous wrt this).
- If generalization / robustness is a motivating reason for better spatial understanding in view transformers, I am a bit perplexed where there seemingly is no such evaluation in simulation nor the real world. For example, it seems pretty easy to evaluate the trained agent on e.g. unseen scene configurations, especially with the existence of generalization benchmarks such as Colloseum (https://arxiv.org/abs/2402.08191) which are based on RLBench.
- Lastly, I am not fully convinced by the failure cases provided in Figure 1. In particular, the first example "Stack 2 blocks in between the bottles" could indeed be explained as a failure to understand the spatial relationship between objects, but it seems equally likely that the culprit may be a mode collapse or a failure to understand the language instruction, especially given the very small number of tasks/instructions.

It is possible that some of my concerns above stem from misunderstandings. If so, I would appreciate it if the authors can clarify those aspects of their paper and commit to revising the text to make it more clear.

### Questions
I would really appreciate it if the authors can address my comments in the "weaknesses" section above using written arguments and potentially additional experimental results (if applicable). I believe that most of my comments (e.g. robustness evaluation) can be addressed without substantial compute.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a dual‑stream view transformer for language‑conditioned robotic manipulation. A static‑view stream adds a cross‑view geometric consistency objective guided by VGGT to learn 3D‑aware features; a dynamic‑view stream reuses an egocentric gaze model (GLC) to produce action‑oriented feature maps and saliency heatmaps from a wrist‑like camera. The two streams are fused to predict 6‑DoF actions. It shows SOTA performance on RLBench and real robot.

### Strengths
Clear motivation to force 3D consistency and fuse dynamic cues. 

Useful ablations: removing the geometric loss drops performance; end‑to‑end fine‑tuning the gaze model underperforms freezing; and heatmaps matter for the dynamic stream.

### Weaknesses
The framing on Cortical policy is unnecessarily complicated. My understanding is that it produces saliency map about end effector position to get inductive bias. Unsure if we need to fine-tune from a gaze model. We could also just exact the effector location from robot forward kinematics and register on camera images, which seems to be an easy baseline that may perform similarly.

### Questions
1. What if just provide dynamic view  as a separate image to RVT-2?

2. Can we extend dynamic view to gaze not just end effector position but also other useful objects?

### Soundness
3

### Presentation
3

### Contribution
3
