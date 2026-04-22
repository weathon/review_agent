# Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
This work introduces Robots Imitating Generated Videos (RIGVid), a system that enables robots to perform complex manipulation tasks—such as pouring, wiping, and mixing—purely by imitating AI-generated videos, without requiring any physical demonstrations or robot-specific training. Given a language command and an initial scene image, a video diffusion model generates potential demonstration videos, and a vision-language model (VLM) automatically filters out results that do not follow the command. A 6D pose tracker then extracts object trajectories from the video, and the trajectories are retargeted to the robot in an embodiment-agnostic fashion. Through extensive realworld evaluations, we show that filtered generated videos are as effective as real demonstrations, and that performance improves with generation quality. We also show that relying on generated videos outperforms more compact alternatives such as keypoint prediction using VLMs, and that strong 6D pose tracking outperforms other ways to extract trajectories, such as dense feature point tracking. These findings suggest that videos produced by a state-of-the-art off-the-shelf model can offer an effective source of supervision for robotic manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Summary:

RIGVid enables a robot to execute real‑world manipulation by imitating AI‑generated videos—without physical demonstrations or robot‑specific training. Given a language command and a scene image, the system (1) generates a candidate video, (2) filters low‑quality generations with a vision‑language model (VLM), (3) extracts a 6‑DoF (6D) object‑pose trajectory, and (4) retargets that trajectory to different robot embodiments for closed‑loop execution.

Contributions:

End‑to‑end from generated video: First framework to execute real‑world manipulation directly from generated videos, requiring no physical demos or robot‑specific training.

Generated videos as supervision: Empirically shows that high‑quality, VLM‑filtered generated videos can substitute for real demonstrations in visual imitation.

Simple, effective recipe: Video generation → VLM filtering → 6‑DoF pose extraction → embodiment‑agnostic retargeting; this pipeline outperforms SOTA alternatives based on VLM abstractions, point/flow tracking, feature fields, or generated goal images.

Closed‑loop, robot‑agnostic control: Robust to disturbances via real‑time re‑tracking and recovery.

Insights:

Filtering is effective. VLM‑based filtering correlates strongly with human judgments and, when it errs, tends to be conservative (false negatives), which is the safer failure mode.

State extraction is key. The 6‑DoF pose trajectory extracted from generated videos serves as a compact, transferable dataset format and appears to be the minimal supervision needed for successful grasping.

Overall assessment:

This is a strong paper with a few remaining questions. If the authors are willing to address them, I would be happy to raise my score accordingly.

### Strengths
Closed-loop robustness. Real-time re-tracking and an explicit backtracking rule (≥3 cm or ≥20° deviation) enable recovery from perturbations—practically valuable and often missing in imitation-from-video work.

Thorough comparative evaluation for trajectory extraction. Benchmarks span tracks (Track2Act), flow (AVDC), feature fields (4D-DPM), and generated-goal supervision (Gen2Act). The 6D-pose rollout consistently performs best, especially on harder cases (thin/occluded objects, depth discontinuities).

Useful ablations on video quality and filtering. Better generators (e.g., Kling v1.6) and automatic VLM-based filtering materially improve success; filtered generations approach real-video performance—an actionable result for the community.

 Demonstrates portability (xArm7 → ALOHA, including a bimanual vignette) via an object-to-end-effector transform, suggesting the representation is robot-agnostic.

### Weaknesses
Scale and statistical power. Main quantitative results cover only four tasks and, for some comparisons, use 10 videos per task per source with human-judged success. That’s limited for strong claims like “on par with real demonstrations,” and no confidence intervals are reported.

Compute and latency under-specified. The pipeline generates and filters videos (up to five attempts), runs depth estimation and 6D tracking, and then executes in closed loop. The paper acknowledges high cost but does not quantify wall-clock time, attempts-per-success, or throughput—metrics that matter for practical deployment.

### Questions
Question 1 :What do generated videos add to robotic manipulation?

My current understanding is that generated videos provide a tighter visual grounding between the object and the robot than trajectory-only supervision. If the robot is ultimately learning a pick-and-place trajectory, why not supervise directly on trajectories? My hypothesis is that VLM-based filtering removes irrelevant content and that generated videos interface more naturally with VLMs than trajectories do.

Request: Please show an experiment that visualizes which generations are filtered out and why. Even better, compare (a) VLM filtering on generated videos vs. (b) an LLM (text-only) filtering of candidate trajectories, holding downstream training constant. If (a) > (b), that would substantiate the claim—and I’m inclined to raise my score if I can see this result.


Question 2:
Please include more qualitative failure cases of the full method (e.g., failure modes in re-tracking, depth errors on transparent/reflective objects, backtracking oscillations, or embodiment-transfer edge cases), along with brief diagnoses and suggested fixes.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes RIGVid, a system that enables robots to imitate AI-generated videos instead of real demonstrations. Using video generation, VLM-based filtering, and 6D pose tracking, the robot executes manipulation tasks directly from synthetic examples. Experiments show synthetic videos can match real demonstrations and outperform keypoint-based and optical-flow baselines. The work suggests high-quality generated video can serve as effective supervision for robotic manipulation, reducing the need for real data.

### Strengths
1. The method requires no robot-specific training or real demonstrations, showing strong potential for scalable imitation learning directly 
from generated videos.
2. The experimental analysis is thorough and insightful, revealing key factors such as the critical role of depth estimation in successful video-based control.

### Weaknesses
1. The paper suffers from dense formatting and uneven space allocation: related work and baseline descriptions are disproportionately long, while the core experimental section is compressed and difficult to follow. This tight layout and aggressive space packing visibly violate ICLR formatting expectations and reduce readability. A clearer narrative structure and more balanced discussion would improve clarity and presentation quality.

2. Although the method follows a different paradigm than Vision-Language-Action (VLA) models, a direct discussion or empirical comparison is still needed. This would contextualize the contribution and help readers understand how generated-video-based imitation currently stands relative to dominant VLA pipelines, clarifying strengths, limitations, and future directions for this alternative trajectory.

### Questions
How well does this method generalize? If applied to a scene that has not been tuned or calibrated, is it still likely to succeed?

### Soundness
2

### Presentation
2

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
This paper pilotly explores using purely AI-generated videos for robots to perform tasks. The pipeline is very simple but effective: 1. Generate videos according to task instructions; 2. Lift the videos into 3/4D; 3. Performation robot pose estimation on the generated point cloud sequences. Extensive experiments are conducted.

### Strengths
1. The idea is insightful. In fact, the reviewer personally also believes in the proposed solution path.
2. The experiments are superior, with extensive ablation studies and real-world evaluations.

### Weaknesses
1. Ablations on 3D reconstruction are missing. How about using other SOTA 4D reconstruction methods, instead of just monocular depth estimation? Will it improve the performance?  
2. Similarly, the tracking method also needs to be ablated. Especially, what if we just use ICP on the generated sequences? This may help to better evaluate the quality of the generation. 
3. If some robot data is used to finetune the video-gen (instead of directly testing), will it improve the performance? 
4. Open question: If the quality of the previous video-gen is low, are there any possible designs to post-refine the result, maybe from another action policy?

### Questions
Although the proposed method itself is not very impressive, the problem that the paper aims to explore is meaningful. The experiments are very comprehensive, making this paper a good *study* research, instead of a *method* paper.

Therefore, as for the current submission, not only the title but also the way to present the paper should be adjusted to a "study"-style paper, as there is no very novel framework is proposed, and no finetuning is provided, but only different video-gen models are tested. 

Moreover, some more experiments mentioned in the weaknesses are expected to be provided to strengthen the paper.

### Soundness
3

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
4

### Summary
This paper presents RIGVid, a demonstration-free robotic manipulation framework that enables a robot to execute tasks purely by imitating AI-generated videos. Given a textual command and a single RGB-D observation, the system generates a video via a diffusion model, filters it with a VLM (GPT-4o) for semantic correctness, estimates per-frame depth, extracts a 6D object pose trajectory using FoundationPose, and retargets this trajectory to the robot for execution. The method is evaluated on four real-world manipulation tasks (pouring, lifting, placing, sweeping) and compared against several strong baselines including ReKep, Track2Act, Gen2Act, and optical-flow or feature-field-based trackers. Experiments demonstrate that filtered generated videos can yield comparable success rates to real human videos, supporting the claim that synthetic visual demonstrations can replace physical ones under certain conditions.

### Strengths
The paper proposes a demonstration-free robotic manipulation framework that relies solely on generated videos. While the approach is currently constrained by the capability of existing video generation models, it points toward a compelling direction: as world-model generators such as SORA and Veo continue to evolve, this paradigm could become a practical and scalable way to get rid of robot teleoperation or imitation data.

The system is solidly engineered. It combines existing modules for depth estimation and 6D pose tracking in a sensible way to get a high-level “pose flow” that drives the robot’s actions. While I don’t think pose is the final or ideal representation for manipulation (as I’ll mention later), this design makes sense at the current era of the  co-evolving embodied AI and generative world models. At least for the four manipulation tasks shown in the paper, the whole pipeline works coherently and delivers consistent results.

The experiments cover a diverse set of real-world manipulation tasks and include detailed ablations on video quality, filtering accuracy, and trajectory extraction methods. The results provide convincing quantitative and qualitative evidence for the framework’s effectiveness.

The paper is clearly written and easy to follow. The figures and descriptions effectively illustrate the overall pipeline

### Weaknesses
The use of FoundationPose requires access to the CAD or mesh models of target objects, or at least works significantly better when such models are provided. This requirement limits the applicability of the system in unstructured or open-world environments where object models are unavailable. In addition, while modern video generators such as Kling can potentially produce videos containing deformable objects, FoundationPose cannot reliably track non-rigid shapes, which further constrains the border usage of the proposed pipeline.

Lack of details on handling pre- and post-grasp phases. Since the method uses the object trajectory directly as the source for retargeting, the robot gripper is assumed to remain fixed at the grasp position. It is unclear how the system handles the pre-grasp phase (approaching and aligning for grasping) and the post-manipulation phase (safely releasing or retracting the gripper). These should ideally be integrated into the overall policy for realistic and safe execution.


The paper does not evaluate any explicit recovery behavior after execution failures. It would be helpful to know how long the recovery takes, and whether the system (or the video generator such as Kling) can re-generate corrective videos to handle failed trials or unexpected conditions.


Current video generation models are limited by the duration of the generated content. The paper does not analyze how such limitations affect tasks with longer execution horizons, where the manipulation may take more time than the video generator can represent.

### Questions
The current pipeline extracts trajectories based on human-hand motion in the generated videos.
Have the authors considered directly generating robot gripper poses from Kling or similar models, which might align better with robotic control?

### Soundness
3

### Presentation
3

### Contribution
2
