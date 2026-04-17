# Point Bridge: 3D Representations for Cross Domain Policy Learning

- Decision: Reject
- Scores: 6, 4, 0, 4

## Abstract
Robot foundation models are starting to realize some of the promise of developing
generalist robotic agents, but progress remains bottlenecked by the availability of
large-scale real-world robotic manipulation datasets. Simulation and synthetic data
generation are a promising alternative to address the need for data, but the utility
of synthetic data for training visuomotor policies still remains limited due to the
visual domain gap between the two domains. In this work, we introduce POINT
BRIDGE, a framework that uses unified domain-agnostic point-based representa-
tions to unlock the potential of synthetic simulation datasets and enable zero-shot
sim-to-real policy transfer without explicit visual or object-level alignment across
domains. POINT BRIDGE combines automated point-based representation ex-
traction via Vision-Language Models (VLMs), transformer-based policy learning,
and inference-time pipelines that balance accuracy and computational efficiency
to establish a system that can train capable real-world manipulation agents with
purely synthetic data. POINT BRIDGE can further benefit from co-training on small
sets of real-world demonstrations, training high-quality manipulation agents that
substantially outperform prior vision-based sim-and-real co-training approaches.
POINT BRIDGE yields improvements of up to 44% on zero-shot sim-to-real trans-
fer and up to 66% when co-trained with a small amount of real data. POINT
BRIDGE also facilitates multi-task learning. Videos of the robot are best viewed at:
https://pointbridge-anon.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes **Point Bridge**, a framework that bypasses the visual domain gap by learning policies on a unified, domain-agnostic 3D point-based representation. Instead of training on raw pixels, the policy operates on 3D point clouds representing the task-relevant objects and the robot's end-effector.
The method achieves zero-shot sim-to-real transfer and significantly improves the task success rate over the baseline using raw pixel inputs. They also show that the performance could be further boosted by co-training with a small amount of real-world data.

### Strengths
1. **Clear, Well-motivated Idea**: Using compact 3D keypoint representations to move the sim/real gap from image appearance to a geometric representation is intuitive and practically appealing. The idea connects naturally to data-generation systems like MimicGen that scale up simulated demonstrations. 

2. **Strong Empirical Results**: Extensive real-robot rollouts, zero-shot sim-to-real experiments, and sim+real co-training analyses with multiple ablations give evidence the approach improves success rates compared to an image-based baseline.

3. **Clarity**: The paper is exceptionally well-written and easy to follow.

### Weaknesses
1. **Dependence on Foundation Models**: The perception pipeline is composed of multiple foundation models (Gemini, Molmo, SAM2, and Foundation Stereo) without a verification mechanism. The performance may be affected if any model fails.

2. **The Domain Gap is Shifted, Not Removed**: By discarding RGB and using only geometry, the method reduces appearance mismatch but increases reliance on accurate depth and occlusion handling. Physics and contact dynamics gaps between sim and real remain, so claims about “closing the sim-to-real gap” should be tempered.

3. **Limited Task Scope**: The experiment tasks are mostly pick-and-place tasks without complex dynamic, geometry, and environment constraints. Although the paper includes "Put bowl in oven" as the articulation task, the robot does not seem to learn the articulation to close the oven door in Figure 3 and the video.

4. **Weak Baseline Comparison**: The image-based baseline is trained on a dataset generated from MimicLab, whose visual rendering gap is notably bigger than the current state-of-the-art simulators (e.g., Issac Sim). The baseline model does not fully demonstrate the limit of image-based sim-to-real transfer. There is also a missing direct empirical comparison to other keypoint/point-based representations or the closest prior works that use learned keypoints or structured geometric inputs.

5. **No Failure Analysis**: Results are reported as success counts without a structured failure taxonomy, which helps understand robustness and reproducibility.

### Questions
1. The policy uses only 3D keypoint positions without RGB. Does this create ambiguity for more complex tasks? How does this design handle tasks that require appearance cues (e.g., sorting by color or distinguishing visually similar objects with the same geometry)? Prior work [1] indicates that combining visual features with geometry can help. Could you elaborate on this?

2. What are the independent failure modes and success rates of the perception pipeline? How do compounded errors (e.g., Gemini failing to identify the correct object, SAM2 failing to segment, or FoundationStereo producing noisy depth) propagate to the policy? Is it possible to develop a verification mechanism to make it more robust?

3. Does the current perception pipeline really capture "task-relevant" keypoints? The current method seems to be simply sampling all points on the task object without understanding the task. For instance, in the task "closing the drawer", the points on the handle and the corners of the drawer should be enough to solve the task and infer the articulation. Would a small set of semantically relevant keypoints perform as well or better? How does the method generalize to variations in object geometry (wider/taller/deeper drawers) without infinitely creating more assets in the simulation?

4. Since the model uses geometry only, what specific benefits does a small amount of real data provide? Is the improvement due primarily to the exact same geometry of the object?

**Typos**:
1. In Table 2’s caption, should “single task” be “multitask” (line 373)?
2. The order of Table 3 and Table 4 appears inverted.

[1] Fang, Xiaolin, et al. "KALM: Keypoint Abstraction Using Large Models for Object-Relative Imitation Learning." 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Point Bridge, a novel framework designed to address the sim-to-real transfer bottleneck in robotic manipulation. The core innovation is the use of a unified, domain-agnostic point-based representation to bridge the visual domain gap. By leveraging VLMs for automated extraction of task-relevant 3D keypoints, Point Bridge distills observations into a compact point cloud, upon which transformer-based policies are trained. The paper demonstrates that this approach enables effective zero-shot sim-to-real transfer using purely synthetic data from tools like MimicGen. Furthermore, it shows that performance can be substantially enhanced through co-training with small amounts of real-world data and that the framework naturally facilitates multitask learning. Extensive real-world experiments on several manipulation tasks report significant improvements over prior methods.

### Strengths
1.The central idea of a point-based representation is both powerful and elegant. It directly attacks the problem of visual domain gap by moving away from pixel-level inputs to a more geometric abstraction. This is a more scalable approach than striving for photorealistic simulation.

2.The framework presents a complete and highly automated pipeline. It intelligently integrates synthetic data generation, VLM-guided scene filtering, and modern policy learning into a cohesive system. The automation of point extraction, removing the need for manual annotation, is a particularly crucial contribution for practical adoption.

3.The experimental validation is thorough and compelling. The evaluation covers not only zero-shot transfer but also co-training and multitask scenarios, showcasing the flexibility of POINT BRIDGE. The large number of real-world evaluations lends significant credibility to the results. The ablation studies on depth estimation strategies and the importance of camera-aligned point sampling in simulation provide valuable insights into the system’s engineering nuances.

### Weaknesses
1.POINT BRIDGE exhibits a strong dependence on external pre-trained vision models. The entire pipeline’s entry point relies on models like Gemini and SAM2. Consequently, the robustness of POINT BRIDGE is inherently tied to the performance of these components, and any failures in perception cannot be easily corrected within the framework itself.

2.The framework relies on assumptions about a calibrated scene with known camera intrinsics and extrinsics. This requirement for a consistent reference frame might limit deployment in more dynamic setups where camera poses are not fixed or precisely known.

3.The abstraction into point clouds, while beneficial for generalization, can lead to a loss of critical scene context. The paper itself notes that this can limit performance in cluttered environments, as fine-grained visual details or contextual cues necessary for disambiguation might be discarded.

### Questions
1.Could you provide more detail on the failure cases and robustness of the VLM-guided pipeline? For instance, how often did the initial object identification or the subsequent segmentation with SAM2 fail or produce inaccurate results in your real-world trials? A discussion of common failure modes and whether the system has any inherent mechanisms to detect or mitigate them would be very helpful.

2.The choice of 128 points per object is noted. Was this parameter systematically ablated? It would be interesting to know if there is a point of diminishing returns or if certain tasks benefit from a different number of points. Furthermore, was any sampling strategy beyond uniform sampling explored that might better capture object geometry for dexterous manipulation?

3.The paper repeatedly emphasizes minimal visual and object-level alignment. To better understand the boundaries of this claim, could you clarify what minimal object alignment entails? Does it allow for transferring policies between objects of different categories with entirely different geometries, or does it assume functional and rough geometric similarity? Showcasing results on a task with extreme object shape variation between sim and real would powerfully reinforce this point.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The manuscript Point Bridge proposes to use a 3D point cloud representation as the basis of a robotic policy. By utilizing this approach the authors are able to bride training on simulation data with deployment on real robots. The advantages of simulation data are utilized by the use of MimicGen, increasing the size of the training dataset substantially. On real data, object- and robot-centric point clouds are generated using multi-view stereo and SAM2 for object segmentation. Performance is evaluated in simulation and on real data by comparing to an image-only baseline, showing substantial improvements on the same amount of training data.

### Strengths
- The method shows good transfer from simulation training to real-world deployment.
- The proposed pipeline is well-engineered, utilizing powerful open-vocabulary models likely capable of generalization to broader scenarios.

### Weaknesses
- The work has very limited novelty. Point cloud and point track representations have been used in numerous previous works (as cited by the authors), for specialist policies as well as of generalist VLA models. While these works do not explicitly target the sim2real problem, they show capable policies on simulation data, human demonstrations and real robot demonstrations. In this context, especially human demonstration data also represents a domain transfer problem.
- The work does not compare to any point cloud and point track-based method. Such a comparison could demonstrate the potential advantage of the proposed pipeline over the baseline methods on the sim2real problem.
- The method requires a calibrated depth/stereo camera setup, with no change between training and inference. The authors do not evaluate the sensitivity of calibration change between training and deployment.
- Minor: Tables 1/2 have incorrect captions, with both being labeled as single task.

### Questions
- A list of differences w.r.t. the most important point tracking works would help the reader to position the work's contributions.
- A benchmark against these methods in the domain transfer case would show the advantages of the proposed method.
- It is unclear how strongly the method depends on a match of camera calibration between training and inference.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes POINT BRIDGE, a cross-domain policy learning framework that centers on “task-relevant points” (keypoints / sparse point clouds) as a unified 3D representation for sim-to-real transfer. The system uses a VLM-guided pipeline (segmentation + depth/reconstruction) to extract task-relevant 3D points (via FoundationStereo / RGB-D / stereo triangulation / tracking), then feeds them to a PointNet + Transformer (BAKU) policy. The goal is zero-shot sim-to-real from large-scale synthetic training, with further gains from limited real co-training.

### Strengths
1.Unified point representation: Mapping both sim and real to task-relevant points is pragmatic and deployment-friendly.

2.Empirical gains: The approach improves over image-based baselines in both zero-shot and limited co-train regimes.

3.Systematic ablations: The paper compares multiple depth/reconstruction sources and discusses viewpoint alignment, offering evidence for deployment trade-offs (success vs. frequency).

4,Implementation clarity: The synthetic-to-3D-to-policy pipeline is clearly described and appears reproducible.

### Weaknesses
1.Limited novelty: Most components (data generation, object filtering, depth estimation, policy learning) are existing modules strung together; the main contribution is a well-engineered integration and representation choice rather than a new learning principle.

2.Baseline coverage (3D/depth): Comparisons are primarily against image-based policies. Missing are baselines that take dense point clouds/depth directly (e.g., point-cloud based, depth-only Diffusion/BC variants) under matched data—making it hard to claim that POINT BRIDGE universally outperform other input modalities.

3.Task simplicity: Core evaluations focus on pick-and-place / stacking, leaving uncertainty about performance on high-contact, non-rigid, assembly, or constrained tasks. I would like to see more results on more complex tasks.

4.Latency/robustness under-analyzed: FoundationStereo yields ~5 Hz, additionally the paper applys many foundation models, The latency and robustness of the pipeline should be more seriously analyzed.

### Questions
Please see 2.3.4 in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
