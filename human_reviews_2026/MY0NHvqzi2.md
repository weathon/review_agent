# DriveMamba: Task-Centric Scalable State Space Model for Efficient End-to-End Autonomous Driving

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Recent advances towards End-to-End Autonomous Driving (E2E-AD) focus on integrating modular designs into a unified framework for joint optimization. Most of these advances follow a sequential paradigm (i.e., perception-prediction-planning) based on separable Transformer decoders and rely on dense BEV features to encode scene representations. However, such manual ordering design can inevitably cause information loss and cumulative errors, lacking flexible and diverse relation modeling among different modules and sensors. Meanwhile, insufficient training of image backbone and quadratic-complexity of attention mechanism also hinder the scalability and efficiency of E2E-AD system to handle spatiotemporal input. To this end, we propose DriveMamba, a Task-Centric Scalable paradigm for efficient E2E-AD, which integrates dynamic task relation modeling, implicit view correspondence learning and long-term temporal fusion into a single-stage Unified Mamba decoder. Specifically, both extracted image features and expected task outputs are converted into token-level sparse representations in advance, which are then sorted by their instantiated positions in 3D space. The linear-complexity operator enables efficient long-context sequential token modeling to capture task-related inter-dependencies simultaneously. Additionally, a bidirectional trajectory-guided "local-to-global" scan method is designed to preserve spatial locality from ego-perspective, thus facilitating the ego-planning. Extensive experiments conducted on nuScenes and Bench2Drive datasets demonstrate the superiority, generalizability and great efficiency of DriveMamba.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes DriveMamba, a novel task-centric and scalable paradigm for end-to-end autonomous driving that replaces traditional sequential or Transformer-based modules with a unified Mamba decoder built on selective state-space models for linear-time complexity. Its key contributions include a sparse token representation that integrates image features and task queries (for perception, prediction, and planning) with 3D positional encoding, a hybrid spatiotemporal scan method—featuring a trajectory-guided "local-to-global" scan—that preserves spatial locality and enhances ego-centric planning, and a unified architecture that simultaneously learns view correspondence, dynamic task relations, and long-term temporal fusion. Extensive experiments on nuScenes and Bench2Drive show that DriveMamba achieves state-of-the-art planning accuracy and efficiency.

### Strengths
1. Brief framework: The paper presents a scalable and unified framework for end-to-end autonomous driving, characterized by its simplicity, effectiveness, and elegant design.

2. Clear presentation and comprehensive experiments: The paper is clearly written and well-structured, with extensive experiments that convincingly demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. **Novelty**: This paper largely follows the framework of DriveTransformer, sharing a similar architecture, experimental setup, and overall structure. The main modification lies in replacing the Transformer module with Mamba, which leads to only marginal improvements in performance and efficiency. Therefore, while the paper does not exhibit major flaws, its contribution is relatively incremental and may receive only a moderate level of interest.

### Questions
See weaknesses.

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
>This paper proposes DriveMamba, a novel Task-Centric and scalable State Space Model paradigm designed for efficient End-to-End Autonomous Driving. DriveMamba aims to address the limitations of conventional Transformer-based E2E-AD systems, which suffer from quadratic complexity and sequential (Perception-Prediction-Planning) design-induced cumulative errors. The core innovation is the Unified Mamba Decoder, which leverages the linear-complexity Mamba architecture to concurrently integrate dynamic Task Relation Modeling, View Correspondence Learning, and Long-term Temporal Fusion in a single stage. Crucially, DriveMamba utilizes sparse, token-level representations instead of dense BEV features and introduces a Hybrid Spatiotemporal Scan guided by the ego-vehicle's trajectory. This scanning mechanism enables efficient long-range context modeling and Ego-planning. Experimental results on the Bench2Drive and nuScenes datasets confirm that the DriveMamba-Tiny model achieves both superior performance and high efficiency, demonstrating the model's scalability and efficacy.

### Strengths
>S1. By replacing the quadratic-complexity Transformer with a Mamba-based decoder (SSM), the method effectively solves the major bottleneck of E2E-AD systems. This design drastically reduces memory consumption and makes the decoder easily scalable through simple layer stacking, which is a critical contribution to the exploration of scalable E2E-AD systems.

>S2. The ablation study rigorously confirms that simply stacking the decoder layers monotonically improves CIPO (Closest In-Path Objects) perception performance. This quantitatively validates that DriveMamba effectively learns perception specifically optimized for planning, rather than general scene perception.

>S3. The experiments are very dense and well-constructed. The proposed method demonstrates superior performance compared to existing baselines across the majority of in-domain scenarios.

### Weaknesses
>The work is well-executed, and I have only one significant concern regarding the robustness properties of the proposed architecture.

>W1. As shown in Table 10, the performance of the trajectory-guided scan appears to be highly dependent on the accuracy of the predicted trajectory. This suggests that the model might perform poorly and lack robustness in Out-of-Distribution (OOD) or extreme scenes with significant domain gaps, potentially causing planning failures. Given that robust operation in diverse and challenging deployment environments is a critical requirement for autonomous driving, the authors should have conducted more comprehensive generalization experiments (e.g., cross-dataset validation) to address this concern, which seems to be missing.

### Questions
>Could the authors clarify the reason for omitting the performance results of the DriveMamba-Base model in Table 2? Including this would allow for a clearer understanding of the model's scalability and the performance trend across different model sizes (Tiny, Base, Large).

### Soundness
3

### Presentation
3

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
This paper proposes DriveMamba, a Task-Centric Scalable State Space Model for efficient end-to-end autonomous driving. The core innovation lies in replacing the traditional attention-based Transformer architecture with a Unified Mamba Decoder, which jointly models perception, prediction, and planning in a single-stage pipeline with linear complexity. The authors further introduce several technical components — Hybrid Spatiotemporal Scan (HSS), task-centric tokenization, 3D sensor token localization, and long-term memory fusion — aiming to enhance scalability, efficiency, and task-level relational modeling. Experiments on nuScenes and Bench2Drive demonstrate consistent performance improvements in L2 and collision metrics, with notable inference speed.

### Strengths
1. Unified Mamba Decoder: Achieves linear complexity while jointly processing perception, map, and planning queries, showing clear scalability advantages on high-resolution multi-camera inputs.
2. Hybrid Spatiotemporal Scan (HSS): Cleverly alternates between spatial and ego-centric scanning to balance locality preservation and long-range temporal consistency.
3. Task-centric tokenization: Structured query design (ego/map/agent) improves modular interpretability and relational learning.
4. 3D sensor token localization: Replacing uniform ray sampling with depth-predicted projection enhances geometric accuracy and spatial reasoning.
5. Comprehensive experiments: Covers multiple benchmarks and provides results with and without ego-status, supporting generalizability and robustness claims.

### Weaknesses
1. Insufficient HSS details: The paper lacks explicit layer-wise configurations or stability studies when varying scan order; the contribution of each H/V-first and L2G layer is not isolated.
2. FPS reporting: Experimental FPS comparisons are unclear due to missing details on resolution, camera count, and hardware setup.
3. Depth branch robustness: No analysis of depth estimation noise, calibration error, or trade-offs between uniform-ray and learned-depth methods.
4. Trajectory prior ambiguity: The source of trajectory guidance (e.g., ego-pose history vs. future leakage) is not clarified, raising potential fairness concerns.
5. Limited interpretability of task relations: The “shared Task Query B-Mamba” strategy lacks ablation or visualization, leaving its contribution unclear.
6. Efficiency gap: The Large model exhibits a steep FPS drop, suggesting scalability bottlenecks at high model capacity.

### Questions
1. How is the trajectory prior in Trajectory-L2G obtained? Any risk of future information leakage?
2. How sensitive is the method to depth noise or calibration errors?
3. Could the authors clarify the HSS layer configuration and its effect on stability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies key challenges in current End-to-End Autonomous Driving (E2E-AD) systems. It argues that dominant methods, which often use sequential Transformer decoders (e.g., perception-prediction-planning), suffer from information loss, cumulative errors, and inflexible task modeling. Furthermore, the reliance on dense BEV features and the quadratic complexity of Transformer-based attention mechanisms create bottlenecks in efficiency and scalability. To address these issues, the paper proposes DriveMamba, a "Task-Centric Scalable paradigm" for E2E-AD. The central idea is the replacement of Transformer decoders with a Unified Mamba decoder. This decoder is based on State Space Models (SSMs), which have linear complexity, to improve efficiency and scalability.

DriveMamba is a single-stage, parallel framework. It tokenizes multi-view images and task-specific queries (Agent, Map, Ego) into sparse representations. These tokens, along with positional embeddings derived in part from a predicted depth map, are fed into the unified decoder. This decoder is designed to simultaneously learn view correspondence, dynamic task relations, and long-term temporal fusion.
A key component is the hybrid spatiotemporal scan method, which is required to apply the 1D Mamba model to the 3D driving scene. The paper introduces a "bidirectional trajectory-guided 'local-to-global' scan". This method dynamically sorts tokens based on their proximity to an intermediate predicted ego-trajectory, aiming to preserve spatial locality from an ego-centric perspective.

Experiments are conducted on the nuScenes and Bench2Drive datasets for both open-loop and closed-loop evaluation. The results show that DriveMamba models achieve lower L2 displacement error and collision rates compared to previous methods, while also demonstrating significant improvements in inference speed (FPS) and reductions in GPU memory consumption. The paper includes ablation studies on the decoder's modular components , scan methods , and scalability.

### Strengths
- The paper clearly articulates two significant problems in E2E-AD: 1) The limitations of sequential, manually-ordered pipelines, such as information loss and error accumulation. 2) The efficiency and scalability constraints imposed by the quadratic complexity of attention in Transformer models.
-  The idea to replace the Transformer decoder with a Mamba-based (SSM) decoder  directly addresses the efficiency and scalability problem. The linear complexity of SSMs is a clear advantage for processing long spatiotemporal sequences. Figure 5 provides a clear comparison of this, showing a 3.2x speed increase and 68.8% less memory use at higher resolutions.
- The paper proposes a "Trajectory-Centric Local2Global" (TC-L2G) scan method. The idea of dynamically sorting tokens based on their relevance to the predicted future ego-path is an explicit attempt to inject an ego-centric bias, which is relevant for the planning task.
- The approach moves away from dense BEV feature maps and instead uses sparse tokenized representations for both sensor inputs and task outputs. These are processed in parallel by a unified decoder, which is designed to enable dynamic modeling of task relationships.
- The model is evaluated on both open-loop (nuScenes) and closed-loop (Bench2Drive)  benchmarks. This dual evaluation provides a more complete picture of the model's planning capabilities, as open-loop metrics do not always correlate with real-world driving performance.

### Weaknesses
I have the following concerns from this work

- The "Trajectory-Centric Local2Global" scan (L232) creates a potential circular dependency. The scan order, which is an input to the decoder layers, is determined by an importance weight $w_i$ calculated from an intermediate predicted ego-trajectory $\psi^{\prime}$. This means the decoder's output (the trajectory) is required to define its input (the scan order). The paper does not specify how this intermediate trajectory is generated or analyze the stability of this co-dependent design. Table 10  shows that using a Ground-Truth trajectory improves results, but this does not resolve the question of how the model functions in practice. 

- The scalability study in Table 6 presents conflicting data. As the decoder is scaled from 3 to 12 layers, closed-loop planning performance improves (51.1 to 66.5). However, open-loop perception performance (Detection mAP, NDS, and Mapping mAP) decreases (e.g., Detection mAP drops from 34.8 to 33.1, Mapping mAP drops from 50.3 to 46.6). This suggests a performance trade-off between tasks, not uniform scalability. The paper's explanation that the model learns "planning-oriented perception" is an interpretation that requires more evidence, as it implies general perception is being sacrificed.
- To support the "planning-oriented perception" claim, the paper provides Table 5 showing that perception of "Closest In-Path Objects (CIPO)" improves with more decoder layers. However, this table only shows CIPO metrics. It does not show the general perception metrics for the same models. To make a convincing argument, the paper should present both CIPO and general perception metrics side-by-side to demonstrate that while general perception degrades (as suggested by Table 6), CIPO-specific perception improves. The current presentation disconnects these two key results.
- The 3D position of sensor tokens, $P_{sensor}$, is a critical component for token sorting and view correspondence. This position is entirely dependent on a predicted depth value $d_{i,k}$, which comes from an auxiliary depth prediction branch. This introduces a significant potential failure point. If the predicted depth is inaccurate, the 3D positions of sensor tokens will be incorrect, leading to flawed token sorting and feature extraction. An ablation study on the accuracy of this depth branch or the sensitivity of the overall model to depth prediction errors is needed.

### Questions
Please see weakness above

### Soundness
3

### Presentation
3

### Contribution
3
