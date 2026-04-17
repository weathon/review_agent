# Articulation in Motion: Prior-free Part Mobility Analysis for Articulated Objects By Dynamic-Static Disentanglement

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Articulated objects are ubiquitous in daily life. Our goal is to achieve a high-quality reconstruction, segmentation of independent moving parts, and analysis of articulation. Recent methods analyse two different articulation states and perform per-point part segmentation, optimising per-part articulation using cross-state correspondences, given a priori knowledge of the number of parts. Such assumptions greatly limit their applications and performance. Their robustness is reduced when objects cannot be clearly visible in both states. To address these issues, in this paper, we present a new framework, *Articulation in Motion (AiM)*. We infer part-level decomposition, articulation kinematics, and reconstruct an interactive 3D digital replica from a user–object interaction video and a start-state scan. We propose a dual-Gaussian scene representation that is learned from an initial 3DGS scan of the object and a video that shows the movement of separate parts. It uses motion cues to segment the object into parts and assign articulation joints. Subsequently, a robust, sequential RANSAC is employed to achieve part mobility analysis \textit{without any part-level structural priors}, which clusters moving primitives into rigid parts and estimates kinematics while automatically determining the number of parts. The proposed approach separates the object into parts, each represented as a 3D Gaussian set, enabling high-quality rendering. Our approach yields higher quality part segmentation than previous methods, without prior knowledge.
Extensive experimental analysis on both simple and complex objects validates the effectiveness and strong generalisation ability of our approach. Project page: https://haoai-1997.github.io/AiM/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Articulation in Motion (AIM), which reconstructs articulated objects from motion videos instead of two-state scans. It uses a dual-Gaussian representation (static + deformable) to separate moving/static parts, then applies Sequential RANSAC on motion trajectories to segment parts and estimate joint parameters without knowing part count.

### Strengths
1. Novel Problem Formulation: The shift from two-state (start-end) analysis to continuous motion video is well-motivated and more practical. The paper correctly identifies that many articulated objects cannot be well-represented by just two discrete states, especially when interior parts are gradually revealed.

2. Technical Contribution - Dual-Gaussian Representation and Static-During-Motion Detection (SDMD): The proposed dual-Gaussian approach ({G^S} for static, {G^M,t} for moving) addresses a real limitation of prior work. The joint optimization strategy with selective pruning is useful.

3. Prior-Free Framework: Eliminating the need to know the number of parts is a significant practical advantage. The sequential RANSAC approach for automatic part discovery is well-suited to this problem.

### Weaknesses
1. Unrealistic Trajectory Assumption (T=2 Sampling): The paper's core technical assumption is critically flawed for real-world videos. The method samples T=2 timesteps: {0 → 0.5, 0 → 1}, which assumes that all the parts of an articulated object should have different state at t=0, 0.5, 1.0. However, in fact, when we capture videos of interacting with objects in real scenes, we usually can only interact with different parts one by one. This means that in a video with hundreds of frames, only a few dozen frames of a certain part may be in motion.  For example:
- t=0.0: Door closed (0°)
- t=0.5: Door closed (0°)      ← No motion detected
- t=0.7: Door opens and then closes
- t=1.0: Door closed (0°)      ← No motion detected!

2. Questionable Baseline Results: The reported performance of ArtGS and DTA is suspiciously poor and inconsistent with the original paper and other current works [1, 2, 3]. For example, Table 2 shows ArtGS achieving 473.72mm CD-m on Blade and 155.65mm on Washer—these are catastrophically bad results. Similarly, Table 3 reports axis angular errors of 89.91° and 89.92° on Fridge and Oven, meaning the estimated joint axes are essentially perpendicular to ground truth, suggesting complete method failure rather than mere performance degradation. These results are orders of magnitude worse than what ArtGS reports in their original paper and what subsequent works[1, 2, 3] have achieved when reproducing ArtGS. If the method truly failed this catastrophically, the authors should explain why, but no such analysis is provided. Other recent papers successfully reproduce ArtGS with reasonable performance, making these results highly suspicious. Additionally, the performance of ArtGS on complex objects is even better than that on two-part objects. This abnormal phenomenon is also an unreasonable reproduction resultWithout credible baseline implementations, the claimed improvements cannot be validated—this is a fundamental flaw that invalidates the entire experimental section.

3. Lack of results in real-world scenario: despite emphasizing practical advantages over two-state methods, the paper provides zero real-world experiments. The authors never demonstrate that their system works on actual captured videos of people opening doors or manipulating objects. Without real-world validation, it's unclear whether the method's restrictive assumptions (fixed T={0,0.5,1.0} timesteps) actually provide any practical advantage over existing approaches. 

[1] Self-Supervised Multi-Part Articulated Objects Modeling via Deformable Gaussian Splatting and Progressive Primitive Segmentation
[2] Part2GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting
[3] REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the task of reconstructing articulated objects from a user-object interaction video and a start-state scan. Previous methods mainly focus on reconstruction from two different articulation states, which the authors think limits their applications and performance. The authors propose a dual-Gaussian scene representation learned from an initial 3DGS scan. It then applies RANSAC on motion trajectories to segment parts and estimate joint parameters.

### Strengths
1. The paper explores a new setting that copes with start-state scans and a human-object interaction video instead of two-state multi-view observations in previous papers. This sounds more reasonable and practical than previous papers.
2. The proposed method use RANSAC to perform part discovery, eliminating the necessity of knowing part numbers ahead.
3. The idea of dual-Gaussian representation that differentiate static and moving Gaussians sound reasonable to me.

### Weaknesses
1. One of the claim of this paper is that it copes with a human-object interaction video and a start-state scan instead of two-state multi-view observations. But the dataset used in the paper seems to be rendered from PartNet-Mobility. Some real-world examples should be helpful.
2. The baseline choices seem weird. The chosen baselines are mainly PARIS, DTA, and ArtGS, which mainly focuses on two-state observations. Some more plausible baseline may be Video2Articulation[1]. 
3. The authors should also elaborate on how they perform these baselines as the setting are very much different.

[1] Generalizable Articulated Object Reconstruction from Casually Captured RGBD Videos

I slightly lean towards borderline reject, and I will change my rating if my concerns are addressed and consider other reviewers' comments.

### Questions
See above

### Soundness
3

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
3

### Summary
The paper introduces a novel framework called ”Articulation in Motion” (AIM), which performs reconstruction, segmentation, and articulation analysis of articulated objects. The authors propose a practical solution that requires only a user-object interaction video and a start-state scan to generate a part-level decomposition, an abstraction of the articulation kinematics, and a 3D digital replica. The framework is developed over three stages: the first consists of a 3D Gaussian splatting reconstruction of the object in an initial static state; in the second step, the static 3D Gaussian splatting is optimised, and jointly another 3D Gaussian splatting representation of the moving parts is generated from the video. At this stage, the novel static parts revealed by the motion of the articulated parts are detected using a static-during-motion (SDMD) module and added to the static representation. In the third stage, with a sequential RANSAC, the trajectories of each moving primitive are grouped to perform motion-based part segmentation. The proposed solution has been tested over a dataset provided by the authors, and it has been compared with state-of-the-art methods by overcoming them, especially on the
segmentation of the articulated parts.

### Strengths
• The internal organisation of the paper is a significant strength, ensuring that each component of the proposed solution is introduced and explained coherently and sequentially.
• The proposed framework introduces a strong methodological contribution, integrating several state-of-the-art algorithms in a novel manner to solve an important problem, such as the analysis and reconstruction of articulated objects without any prior knowledge of the object's moving parts.
• The quantitative and qualitative evaluations reported in the experiment section confirm the validity of the proposed solution. In particular, the proposed method outperforms all state-of-the-art methods across metrics related to part segmentation, while achieving above-average results in the other metrics.

### Weaknesses
- The dual Gaussian representation is introduced as novel, but it seems the same as the dense static Gaussians and sparse dynamic Gaussians introduced in ArtGS.
- Apart from the novelty of the video interaction, it is not quite clear where the novelty here is w.r.t. ArtGS.

### Questions
1. Given the interaction video, is it possible that two parts are in motion at the same time? Or is the interaction constrained to a single specific part motion?
2.  The number of states is not clear.
3.  How is the freezing and unfreezing of the static Gaussian managed?
4. I would appreciate some clarification on the dual optimisation.

### Soundness
3

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
3

### Summary
Paper addresses the problem of piecewise-rigid reconstruction via combining old-school computer vision (sensible priors in a dual Gaussian formulation and sequential RANSAC) with modern Gaussian-based 3d rendering.

This is a solid piece of engineering that shows that the use of established CV methods can substantially contribute to the performance of modern systems.

+ Comfortably state-of-the-art 
+ Good ablation studies showing the contribution of each component.
+ Thoughtful qualitative results/figures making it clear how each component is useful.

It's worth mentioning that this paper opens the door to more sophisticated forms of model fitting than sequential RANSAC. However, the proposed benchmark is saturated by this approach. More challenging data would be needed before anything better than greedy optimization is needed.

### Strengths
This is an extremely well-motivated approach where the design decisions have a clear and obvious contribution and that substantially improves over multiple recent existing approaches.

The writing is clear, and the design decisions are well supported by experimental evaluation.

+ Comfortably state-of-the-art 
+ Good ablation studies showing the contribution of each component.
+ Thoughtful qualitative results/figures making it clear how each component is useful.

This paper opens the door to more sophisticated forms of model fitting than sequential RANSAC. However, the proposed benchmark is saturated by this approach. More challenging data would be needed before anything better than greedy optimization is needed.

### Weaknesses
The clarity of the approach probably works against this paper (at least at review time, this clarity is a benefit when published), and I suspect that some of the other reviews may complain about lack of novelty as it looks obvious in hindsight. Of course, if it was obvious, the existing approaches would already be doing something similar, and the performance improvement would be less noticeable. 

I think describing the dual Gaussian approach as prior free is somewhat misleading. The reason it works well is because it injects a healthy prior that most of the world is static, which makes the actual task much easier.

The dataset is small by modern ML dataset standards, consisting of few enough sequences that they can be named and discussed individually. However, this is fairly common in 3d reconstruction, where if you want high-quality ground truth, you typically have to pay the cost in terms of a small number of evaluation sequences.

### Questions
-

### Soundness
4

### Presentation
3

### Contribution
4
