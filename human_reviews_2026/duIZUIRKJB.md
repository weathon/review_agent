# MAV-SLAM: Multi-LLM-Agent Crew for Visual SLAM with 3D Gaussian Splatting

- Decision: Reject
- Scores: 4, 6, 2, 0

## Abstract
Visual Simultaneous Localization and Mapping (SLAM) reconstructs the metric structure of the physical world from sensor imagery, enabling precise robotic pose estimation. However, environmentally induced image degradation and varied image processing strategie significantly compromise localization accuracy. Intelligent SLAM systems address this challenge by autonomously perceiving dynamic perturbations and formulating adaptive processing strategies, further identifying and deploying optimal methodologies to achieve target localization objectives with enhanced metric precision. This paper introduces MAV-SLAM, a novel Multi-LLM-Agent-Orchestrated visual SLAM framework that proactively identifies and compensates for suboptimal image quality while autonomously selecting optimal depth estimation models. Specifically, we integrate a visual-language model that performs autonomous image restoration guided by image quality assessment, significantly enhancing SLAM localization performance. Furthermore, we implement a routing large language model for adaptive depth estimation, which consequently elevates the quality of 3D reconstruction via 3D Gaussian Splatting (3DGS). Rigorous evaluation across multiple benchmarks demonstrates that MAV-SLAM exhibits superior performance in both localization accuracy and 3DGS-based reconstruction fidelity, validating its effectiveness in real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MAV-SLAM, a novel Visual-Inertial (VI) SLAM framework designed to improve localization accuracy and 3D reconstruction quality, especially in challenging real-world conditions. The authors argue that SLAM performance is often compromised by environmental factors (e.g., low light, haze, rain) and camera artifacts (e.g., motion blur), and that no single depth estimation model is optimal for all scenarios.

### Strengths
1. Instead of a monolithic model, the paper presents a "crew" of specialized agents that collaborate, mimicking a human-like reasoning process (assess, then act). This modularity (Assessor, Augmenter, Estimator) makes the system interpretable and extensible.

2. The experiments in Table 1 and Figure 4 effectively demonstrate that this restoration frontend improves downstream localization accuracy and feature matching.

3. The performance graphs in Figure 5 are very compelling, showing "Ours" consistently achieving the lowest, most stable error, effectively creating a "best-of-all-worlds" expert. Table 5 shows that MAV-SLAM, which is the only method in the comparison that does not use ground-truth depth, achieves reconstruction quality (PSNR/SSIM/LPIPS) that is competitive with other state-of-the-art SLAM frameworks that do

### Weaknesses
1. The proposed system is a cascade of multiple large, heavyweight models (Orchestrator LLM -> Co-Instruct VLM -> InstructIR -> PolicyNet -> one of three large depth models). This entire pipeline must be executed per frame for a SLAM system. This introduces a massive, unaddressed latency problem. The "Time" metric in Table 3 is ambiguous (units are not specified) and only covers the routing module, not the full end-to-end pipeline. The paper makes claims about "real-time" SLAM  but provides no evidence that this complex, sequential-agent framework can operate at the framerates required.

2. The paper's two main contributions (restoration and routing) are never evaluated together. The restoration experiments are run on EuRoC. The depth routing experiments are run on Replica and TUM-RGBD. There is no experiment that feeds a degraded image (e.g., "blurry" or "low-light" from EuRoC) into the full MAV-SLAM pipeline to see how the restoration and routing modules interact. This is a significant missed opportunity to validate the complete system.

### Questions
1. What is the end-to-end, wall-clock latency (in milliseconds) for processing a single frame through the entire MAV-SLAM pipeline, from image capture to pose estimation? Can this system truly run in real-time on the NVIDIA 3090 GPU mentioned?

2. The experiments evaluate restoration (Sec 5.1) and routing (Sec 5.2) in isolation. How do these modules interact? For example, does the image restoration from the Augmenter ever introduce artifacts that confuse the PolicyNet or the downstream depth models, leading to a worse depth estimate than using the original, degraded image?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a novel visual SLAM framework that integrates multi-agent large language models (LLMs) to improve robustness under image degradation (e.g., blur, low light, rain) and enhance 3D reconstruction quality using 3D Gaussian Splatting (3DGS).

### Strengths
The work has broad potential impact on both SLAM and embodied AI communities. Rather than viewing SLAM as a fixed pipeline, this work positions it as an adaptive, reasoning system that perceives environmental conditions and self-modifies—a step toward truly autonomous robotic perception. By using LLMs not just for language but for orchestrating geometric processing, the paper offers a blueprint for hybrid neuro-symbolic systems in robotics. Image degradation (blur, low light, rain) is a real-world bottleneck in drone, automotive, and AR/VR applications. A system that autonomously detects and compensates for these issues without manual tuning is of clear industrial value. The modular, agent-based design invites extensions—e.g., adding failure recovery, incorporating inertial data, or integrating with world models. It also sets a precedent for using LLMs as controllers rather than just annotators in vision pipelines.

### Weaknesses
1. The paper attributes performance gains to the multi-agent orchestration framework, but it never isolates the impact of the agent structure versus the underlying components (e.g., InstructIR for restoration, RouteLLM for depth selection). For instance, Is the Orchestrator truly necessary, or could a monolithic LLM with function calling achieve the same result? Does the ReAct loop provide measurable benefits over static pipelines?

2. The Assessor uses Co-Instruct for IQA, but the paper does not clarify how degradation types (blur, rain, etc.) are quantitatively identified. Is this a classification head? Zero-shot VLM prompting? If the Assessor relies on textual prompts like “Is this image blurry?”, then its reliability hinges on VLM robustness—yet no failure analysis or confidence calibration is provided. Table 2 reports counts of blur/low-light frames, but it’s unclear if these labels come from ground truth or the Assessor itself. If the latter, the evaluation may be circular: the system is evaluated on its own diagnostic output.

3. The RouteLLM is trained on Replica (room2) and TUM-RGBD (freiburg2) and tested on other sequences from the same datasets. However, both datasets are indoor, synthetic or controlled, with limited visual diversity (e.g., no outdoor scenes, weather, dynamic objects). The paper claims robustness to “real-world scenarios” (Abstract), but provides no evaluation on truly unstructured environments (e.g., KITTI, TartanAir, or custom drone footage with motion blur + rain).

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents MAV-SLAM, a multi-LLM-agent orchestration for visual SLAM: a VLM-based IQA and instruction-driven all-in-one restoration frontend, plus Route-LLMs that select among monocular metric depth experts and predict a global scale. The enhanced images and depth are fed to VO and 3DGS to achieve robust localization and high-quality reconstruction.

### Strengths
1.	The use of LLM/VLM-based autonomous image quality assessment and degradation recognition at the SLAM frontend enables condition-aware preprocessing and parameterization, improving robustness under low light, blur, weather, and other visual degradations.
2.	A Route-LLM dynamically selects among metric monocular depth experts and performs scale prediction/calibration, mitigating the cross-domain generalization gap and scale drift inherent to any single depth model and stabilizing downstream VO/reconstruction.

### Weaknesses
1.	Limited technical novelty and contribution. Although the paper is selling some fancy “terms”, the system largely stitches together existing components. There are no new framework or solution proposed. The title “MAV SLAM with 3DGS” suggests new advances at the SLAM/3DGS level, but the paper mainly plugs into and evaluates existing components; the naming/positioning is somewhat misleading.
2.	Insufficient experimental evidence to validate the effectiveness of the LLM-based routing policy. Improvements are modest and inconsistent across datasets/scenes, while the proposed system even degrades the performance in some senarios. In addition, the author should report the running speed and computational cost. 
3.	The manuscript suffers from several inconsistencies in presentation that hinder clarity. Multiple tables including Table 1 fail to consistently bold the best-per-sequence results, making it difficult to identify top-performing methods at a glance. Moreover, the labeling in Table 1 is ambiguous: it is unclear which columns correspond to results before versus after image restoration. Figure 4 also presents interpretability challenges. While the authors state that they “quantify the variation in feature point density and matching efficiency,” no actual quantitative metrics, such as the number of detected feature points or the feature matching ratio, are provided. This omission undermines the claim of quantification and limits the figure’s usefulness. Additionally, the sub-captions across figure 4 lack terminological consistency. For instance, terms like “deblur” and “enhance” are positive-action verbs, whereas “low light” is a descriptive noun phrase with negative connotation

### Questions
1.	On image enhancement comparisons: Does the authors include fair comparisons against state-of-the-art enhancement/restoration methods? Comparing only “raw vs. enhanced” inputs is insufficient to demonstrate superiority over other enhancement techniques.
2.	On dataset breadth: Have the authors validated on additional SLAM/reconstruction datasets? Results on only five EuRoC sequences are too limited to support claims of cross-domain robustness. Please add multi-dataset evaluations with variance/statistical significance.
3.	On the mechanism of Route-LLM: The ablation in Table 3 suggests that gains mainly stem from scale calibration, while the benefit of “intelligent model selection” is unclear. Are there any experiments to prove the effectiveness of the proposed method?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper uses a VLM to attempt to overcome image degradations due to environmental conditions, and correspondingly improve VSLAM performance in settings with poor observations

### Strengths
It is hard to find a genuine strength among the review criteria set of originality, quality, clarity, or significance.

### Weaknesses
The main weakness with this approach is that its validity is highly questionable. Accurate VSLAM requires accurate feature tracking from the environment. An input image manipulated by a VLM is will have no guarantees on consistency or accuracy wrt real world geometry. At best, some simple image processing may help, such as gamma correction, but in such cases, the use of a VLM is overkill - autoexposure algorithms are a better bet. For anything non-trivial, such as distortion due to rain or fog, I would not trust the manipulated image from a VLM for performing VSLAM downstream.

The RouteLLMs claim is that per-frame choice of depth estimation algorithm should improve consistency, which does not make sense - if the algorithm changes per frame, the output depth would vary significantly between frames too.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1
