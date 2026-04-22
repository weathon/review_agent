# RACE: Real-Time Adaptive Camera-Intrinsics Estimation via Control Theory

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Modern embodied AI systems, from mobile robots to AR devices, rely on accurate camera intrinsics to ensure reliable perception. Yet in real-world operation, intrinsics drift due to heating, zoom events, mechanical shocks, a single hard landing, or simply incorrect factory calibration, violating the fixed-parameter assumption that underpins most vision and learning pipelines. This induces a distribution shift in the visual input, which in turn degrades the performance of downstream models and tasks that rely on stable camera geometry. We introduce RACE (Real-time Adaptive Camera-intrinsic Estimation), a provably stable online learning algorithm that continually estimates camera intrinsics directly from continuous monocular image stream. RACE updates parameters through a Lyapunov-stable adaptive law, guaranteeing global asymptotic convergence of the reprojection error dynamics and recovery of the true intrinsics under persistent excitation. Unlike prior batch optimization, heuristic self-calibration or learning-based approaches, RACE requires no training data, bundle adjustment, or retraining, and provides the first theoretical bridge between adaptive control and online learning for camera models. Empirically, we evaluate RACE across public benchmarks (EuRoC, TUM, and TartanAir), showing that it matches or surpasses state-of-the-art learning-based calibration while adapting in real time with negligible computational overhead. Our results highlight RACE as a new class of theoretically grounded continual learners for camera intrinsics, enabling robust long-term perception in embodied agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel and highly original approach to online camera intrinsic calibration by framing it as an adaptive control problem. The core strength of the work lies in its rigorous theoretical foundation, providing formal stability guarantees (global asymptotic convergence and GUUB) under ideal conditions. The method is impressively efficient, achieving real-time performance on a single CPU core, and is training-free, offering strong potential for long-term autonomy.

### Strengths
RACE (Real-time Adaptive Camera-intrinsic Estimation) frames intrinsic calibration as an adaptive control problem. Treating intrinsics as dynamic states, RACE applies a lightweight Lyapunov-based update driven by reprojection errors. Under standard persistent excitation(PE), it proves global Lyapunov stability
of the error dynamics. Practically, RACE performs intrinsic calibration online, which is training-free, requires no bundle adjustment, and runs in real-time on a single CPU.

### Weaknesses
Despite its theoretical elegance, the paper suffers from several critical weaknesses that significantly undermine its claims of practicality and generalizability:

1.Overly Idealized Assumptions: The theoretical guarantees are predicated on the assumption of "known accurate poses," which creates a fundamental circular dependency in any real SLAM system. The analysis lacks any robustness guarantees under pose estimation noise, which is unavoidable in practice.

2.Inadequate Experimental Validation:
Unrealistic Perturbation Model: The use of a "global scaling" perturbation for initial offsets is a severe oversimplification. It avoids the core challenge of coupled parameter identifiability and likely overstates the method's robustness. The absence of tests with independent parameter perturbations is a major omission.
Unfair Benchmarking: The comparisons against state-of-the-art methods are not conducted on a level playing field. A direct comparison against classical online estimators (e.g., Recursive Least Squares, online Gauss-Newton) under the same simplified setting (known poses, same perturbations) is required to isolate the benefit of the proposed control law.
Lack of System Integration Clarity: The paper fails to clearly explain how the RACE module would be integrated into a full SLAM pipeline to break the circular dependency, leaving its practical implementation in doubt.

3.Incomplete Analysis of Limitations:
The extension to lens distortion relies on a heuristic "continual linearization" approach without providing stability proofs for the resulting nonlinear system.
The analysis of failure cases (e.g., on TartanAir) is superficial. The role and effectiveness of the proposed PE-gating mechanism are stated but not quantitatively demonstrated with ablation studies or diagnostic plots.

The paper presents a promising direction but in its current form, it reads more as a proof-of-concept under idealized conditions than a thoroughly validated practical solution. Addressing these points is crucial for transitioning the work from a theoretically interesting idea to a impactful contribution with clearly demonstrated real-world applicability.

### Questions
The following major revisions are essential for establishing the paper's credibility and practical value:
1.	Conduct a fair ablation study comparing RACE directly against Recursive Least Squares and an online Gauss-Newton optimizer under identical conditions (known poses, same perturbation models).
2.	Evaluate robustness using independent parameter perturbations to test performance in a more realistic and challenging scenario.
3.	Provide a clear description and analysis of how RACE can be integrated into a SLAM system without the "known pose" assumption, addressing the circular dependency problem.
4.	Include a thorough diagnostic analysis of failure modes (e.g., on TartanAir), providing quantitative evidence linking performance drops to PE condition violations and demonstrating the efficacy of the PE-gating mechanism.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
1. The paper proposes that intrinsics will drift in real-world operation, due to heating, zoom events, mechanical shocks.

2. This work presents RACE, a control-theory-based online camera intrinsic estimator that runs in real time on a single CPU (no GPU/pre-training/batch processing). It provides theoretical guarantees (global stability, convergence) and outperforms baselines (e.g., COLMAP, DroidCalib) on EuRoC/TUM RGB-D with sub-pixel reprojection error, addressing real-world calibration pain points.

3.  Grounded in control theory, RACE treats intrinsic parameters as dynamic states and employs a lightweight Lyapunov-based update law. RACE requires no training data, bundle adjustment, or retraining, and provides the first theoretical bridge between adaptive control and online learning for camera models. RACE is a new class of theoretically grounded continual learners for camera intrinsics, enabling robust long-term perception in embodied agents.

### Strengths
1. Novel insight: camera intrinsics will drift in real-world operation, due to heating, zoom events, mechanical shocks in real-world application. 
2. Real-time and online calibration
3. Less projection errors by allowing for online update of camera intrinsics

### Weaknesses
1. Require precise camera extrinsics such as Rotation and Translation matrix.
2. Rely on persistent excitation to guarantee convergence and stability.

### Questions
1. Suppose that the colmap could update its predicted camera intrinsics, could RACE still surpass colmap in precision?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces RACE, a new method for estimating camera intrinsic parameters (like focal length and principal point) online and in real-time. The core idea is to frame this as an adaptive control problem. The authors use a Lyapunov-based update law to prove that their estimator is stable and that the intrinsic parameter error will asymptotically converge to zero, provided the camera's motion and scene are sufficiently varied (a condition known as persistent excitation).

The method does not require any pre-training or large datasets. The paper's experiments on benchmarks (EuRoC, TUM) show that RACE is fast (running on a CPU), robust to large initial errors, and achieves accuracy that matches or surpasses other state-of-the-art learning-based methods.

### Strengths
1. Theoretical Novelty: The primary strength is the use of adaptive control theory to tackle this problem. Applying a Lyapunov-based analysis provides provable guarantees of stability and convergence, which is a significant advantage over many end-to-end deep learning methods that may lack such guarantees.

2. Efficiency and Accessibility: The method is very lightweight. It runs in real-time on a single CPU core, requires no GPU, and is training-free. This makes it highly practical from a computational standpoint.

3. Strong Experimental Results (Given Assumptions): The empirical results are impressive. The method demonstrates high accuracy on the EuRoC and TUM datasets, and the ablation studies show that it is robust to significant noise and very large initial parameter errors (e.g., 100-200% offsets), which is a strong validation of its stability.

### Weaknesses
1. Strong Assumption of Known Poses: The most significant weakness is the assumption that the algorithm has access to accurate camera poses and 2D-3D correspondences. The authors state this in Section 3.2 and the limitations. This assumption seems to create a 'chicken-and-egg' problem for the scenarios the paper motivates, like autonomous driving or robotics. In a real-world setting, a drift in camera intrinsics would almost certainly degrade the performance of the pose estimation system (e.g., SLAM or localization). One cannot assume a perfect pose to fix the intrinsics, because the imperfect intrinsics are needed to find the pose. This assumption, while also made by some other works, severely limits the practical applicability of the method as a standalone solution.
2. Reliance on Persistent Excitation (PE): The method's convergence guarantee depends on the PE condition, meaning the camera must be moving in a way that provides rich visual information. The paper's own results on the TartanAir dataset show that performance degrades in challenging segments (fog, low light) where this condition is likely not met. While the authors propose 'gating' (pausing) the update, this remains a practical limitation for real-world use where long, non-informative sequences can occur (e.g., driving on a straight highway).

### Questions
1. Given the reliance on known camera poses, could the authors elaborate on the intended practical deployment scenario? Is this method intended to be integrated into a larger SLAM or visual odometry system?

2. Following up on the first question, Appendix E shows a test with ORB-SLAM3. Can you clarify the setup? Was the RACE algorithm still fed ground-truth poses to estimate intrinsics, which were then passed to ORB-SLAM3? Or was there a joint estimation where RACE used the (potentially incorrect) estimated poses from ORB-SLAM3? How would the stability guarantees be affected if the input poses are noisy or drifting?

3. The paper proposes 'gating' the update when the PE condition is weak. Was this technique used in the TartanAir experiments? How well does the system perform if it encounters a long period of weak PE and then must re-converge when informative motion resumes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors propose a real-time adaptive camera-intrinsic estimation (RACE) method. 

Different from previous offline approaches, self-calibration methods with global bundle adjustment (e.g. colmap), and pretrained models, the proposed method updates the intrinsic parameters adaptively based on control theory with a strong assumption that correct 2D-3D correspondences and camera poses are known.

Experiments on public datasets including EuRoc, TUM RGB-D, and TartanAir datasets demonstrate the proposed approach gives smaller reprojection errors than previous methods such as Colmap, DroidCalib which don’t have any assumptions.

### Strengths
1.	Good motivation. As mentioned in  the paper, the camera intrinsics parameters change with time (maybe not too much) and require careful calibration before being use for data collection. Previous works including Zhang’s approach, global BA, and end-to-end models do have their disadvantages. Therefore, a simple, effective, and fast online calibration approach is very useful. 

2.	Experiments on three public datasets are good. Maybe one or two outdoor datasets would be better.

3.	The metric of reprojection error makes sense especially with the strong assumption that correct 2D-3D correspondences and camera poses are known.

### Weaknesses
1.	Very strong and impractical assumption. The proposed method assumes that correct 2D-3D correspondences and camera poses are known. This is a extremely strong assumption and is impractical in real applications. How can we get correct 2D-3D correspondences and camera poses without knowing the correct intrinsic parameters?

2.	If correct 2D-3D correspondences and camera poses are given as assumed in the paper, the easiest way to obtain the intrinsic parameters is solving a least square problem (LSP). What is difference between the proposed method and the least square problem? Besides, the LSP should be one baseline for comparison in the experiments.

3.	The methods chosen for comparison including colmap, DroidCalib, etc  never have assumptions of 2D-3D correspondences or camera poses. The comparison between the proposed approach with these methods is not fair.

4.	The reprojection error with ground-truth camera intrinsic parameters should be included in experiments as reference as their results should be the upbound.

### Questions
Overall, I don’t think this paper should be accepted. Please see the weaknesses section for more details.

### Soundness
1

### Presentation
2

### Contribution
1
