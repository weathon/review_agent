# UIT-Pred: Universal Intermittent Trajectory Predictor for Autonomous Driving

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Trajectory prediction is a fundamental component of autonomous driving, requiring models that can handle intermittent observation patterns such as variable-length histories and missing data. Existing state-of-the-art methods, however, often assume fixed-length trajectories and complete input, which challenges their applicability in real-world scenarios where sensor occlusions, communication delays, and temporal sparsity are common. Moreover, conventional approaches typically address tasks such as trajectory prediction, variable-length modeling, or missing data handling in isolation, making them less effective in multi-task settings that naturally arise in practice. To address these challenges, we propose Universal Intermittent Trajectory Predictor (UIT-Pred) that processes inputs with the time index features, which capture temporal variations to effectively adapt to diverse input patterns within the domain. Particularly, We extend recent State Space Models (SSMs) by introducing the Bidirectional Time Decay Mamba (BTD-Mamba), designed to capture dependencies both forward and backward along the sequence. By integrating a decay process, BTD-Mamba effectively analyzes trajectories while maintaining relationships under intermittent observation. Furthermore, the proposed prediction module employs state encoding to capture the underlying motion patterns in the input data and models a multimodal trajectory distribution to account for uncertainty in future predictions. These components are fused through a unified fusion module, enabling the model to jointly reason over observed dynamics and potential future behaviors. Extensive experiments on Argoverse 1 and Argoverse 2 datasets validate the effectiveness of the proposed model. By simultaneously handling prediction, variable-length observations, and missing inputs within a universal architecture, the framework proposes to meet the challenges of real-world autonomous driving systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes UIT-Pred, a unified trajectory prediction framework for autonomous driving that effectively handles intermittent and variable-length input observations, such as those caused by occlusions, sensor limitations, or agents entering/exiting the scene. Its key contributions include: (1) a time-aware input representation using scaled timestamps and inter-observation intervals to model motion dynamics without fixed-length assumptions; (2) the Bidirectional Time-Decay Mamba (BTD-Mamba) module, which enhances state-space models with bidirectional processing and a temporal decay mechanism to maintain continuity across irregular observations; and (3) a novel prediction module with learnable state embeddings and cross-attention mechanisms to fuse agent history, multi-modal intentions, and map context. Extensive experiments on Argoverse 1 and 2 show that UIT-Pred achieves state-of-the-art performance across diverse and challenging input conditions.

### Strengths
1. Address the core points: The paper addresses two key challenges in real-world trajectory prediction - variable-length observations and data loss - which mainstream methods typically handle only one of or rely on complete data.
2. The structure of the paper is complete, including methods, experiments and visualization results.

### Weaknesses
1. **Novelty**: The overall architecture is highly similar to DeMo [1], with only minor modifications. Such incremental changes are insufficient to establish strong novelty for the paper.

2. **Experiment**: As shown in Table 1, the proposed method does not demonstrate significant improvements over DeMo across most metrics. Moreover, Table 2 does not specify whether the reported results on the Argoverse 1 and Argoverse 2 datasets are based on the validation split or the test split, which reduces the clarity and credibility of the experimental comparison.

3. **Figure**: The figures are not provided in vector format, leading to limited visual clarity and reduced readability in the manuscript.


**References**:

[1] DeMo: Decoupling Motion Forecasting into Directional Intentions and Dynamic States, NeurIPS 2024.

### Questions
See weaknesses.

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
3

### Summary
This paper proposes Universal Intermittent Trajectory Predictor, a universal trajectory prediction framework for autonomous driving that simultaneously handles variable-length historical trajectories and missing observations—common real-world challenges caused by sensor occlusions, communication delays, and temporal sparsity. The authors introduce three key components: (1) a time-aware input representation using timestamps and inter-observation intervals; (2) a Bidirectional Time Decay Mamba module that extends the Mamba architecture to capture forward and backward temporal dependencies while mitigating information gaps from missing data; and (3) a unified prediction module that encodes motion states and models multimodal future distributions. Extensive experiments on Argoverse 1 and Argoverse 2 demonstrate consistent performance gains across diverse observation conditions.

### Strengths
1. The joint modeling of variable-length inputs and missing data aligns closely with practical autonomous driving scenarios, addressing a notable gap in existing methods that assume fixed-length, complete trajectories.
2. Comprehensive experiments: The evaluation covers four realistic input conditions and compares against strong baselines (e.g., LaKD, RSD, DTO).
3. The paper includes detailed component-wise analyses (TAIR, BTD-Mamba, prediction module) and auxiliary loss ablations, validating the contribution of each design choice.

### Weaknesses
1. The model integrates BTD-Mamba, Transformers, GRUs, cross-attention, and multiple auxiliary tasks, raising concerns about computational cost and real-time deployability, critical for autonomous systems, but these aspects are not discussed.
2. While the method outperforms variants of HiVT and QCNet, it lacks comparison with cutting-edge approaches such as diffusion-based predictors or hybrid SSM-Transformer models under the same intermittent observation settings.
3. The use of a negative exponential decay (Eq. 5) is intuitive but not rigorously motivated or compared against alternatives (e.g., learnable gates, linear decay), leaving room for doubt about its optimality.
4. Missing observations are generated via random drop (RSD), which may not reflect real-world structured missingness (e.g., prolonged occlusions behind large vehicles), limiting the practical validation of robustness.

### Questions
1. Can the authors provide inference speed (FPS) or model size comparisons? Is UIT-Pred feasible for real-time deployment on embedded automotive platforms?
2. Does the bidirectional processing in BTD-Mamba violate causality during inference? If used online, how is the backward pass implemented without future observations?
3. Have the authors considered modeling missingness as a function of scene context (e.g., occlusion by nearby agents or map geometry), rather than relying solely on time intervals?
4. Is the model trained on mixed data (full + missing + variable-length), or can it generalize to missing scenarios when trained only on complete trajectories?

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
3

### Summary
This paper introduces UIT-Pred, a unified architecture for trajectory prediction in autonomous driving that robustly handles variable-length input histories and missing data. The method combines a time-aware input representation, a Bidirectional Time Decay Mamba (BTD-Mamba) module (extending state space models with bidirectional and decay-aware processing), and a multimodal prediction module using integrated state encoding and cross-attention fusion. The approach is validated empirically on the Argoverse 1 and Argoverse 2 benchmarks, demonstrating state-of-the-art performance under several observation regimes and scenarios. Extensive ablations and qualitative results further support the claims.

### Strengths
+ The paper tackles an important and under-addressed real-world challenge: predicting agent trajectories under intermittent (i.e., variable-length and missing) observation sequences, directly motivated by realistic issues such as sensor occlusion and communication drops in autonomous driving.
+ The method brings together several principled ideas—temporal normalization, bidirectional SSMs with time decay, and unified fusion—to handle variable and incomplete data without reliance on input masking, explicit validity masks, or duplicative augmentation. The proposed Bidirectional Time Decay Mamba module is theoretically motivated and technically well-integrated, extending recent progress in SSMs in sequence modeling to the irregular sampling domain.

### Weaknesses
1. While the decay process in BTD-Mamba is heuristically motivated (see Equations 4–5), the formulation and parameterization of the decay via an MLP, as well as the use of a negative exponential, seem ad-hoc. There is no theoretical analysis on how this decay mechanism affects information retention or gradient propagation through highly irregular gaps. A more principled or formal treatment would be valuable.
2. Ambiguity in Loss Weighting and Training Objective: The main loss function (Equation 23 and detailed appendix losses) weights all terms equally without analysis or justification. No tuning or sensitivity study is reported regarding these hyperparameters. In multi-loss systems, uniform weighting is rarely optimal. This could materially affect empirical results (see Equations 24–33, Table 8 in the appendix for auxiliary losses – but no broader tuning study).
3. Certain key steps gloss over important mathematical details: the cross-attention decoders used extensively for graph, lane, and multimodal fusion are described abstractly ($\mathcal{D}(·)$ in Equations 14–20) but lack explicit architectural detail (are positional encodings used? What are the dimensions of queries/keys/values in multi-modal context?).
4.  There are a few spots where notational clarity is lacking, particularly in not distinguishing between predicted, reconstructed, and observed trajectories within tensors in the main methodology (Equations 15–17). Indexing switches between $j$, $t$, and $l$ within equations can be confusing.

### Questions
1. How robust is UIT-Pred to adversarial missing patterns? Any chance for synthetic stress-testing to be added or analyzed?
2. In the fusion module, are positional encodings used within the cross-attention layers (cf. Equations 14-20)? If so, how are these adapted for variable-length or irregularly timed data? 
3. Why did the authors choose to weight all loss terms equally in $L_\text{total}$? (cf. equation 23)

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
This paper proposes UIT-Pred, a trajectory prediction framework for autonomous driving that is designed to work under intermittent observation, e.g. variable-length histories and missing inputs. It uses time-aware input representation so the model doesn’t need to rely on fixed windows or masks. It uses a Bidirectional Time Decay Mamba (BTD-Mamba) to capture sequential dependencies in both forward and backward directions across input observations. Experiments on Argoverse 1 and 2 Datasets show consistent gains over baselines (HiVT, QCNet, DeMo, Forecast-MAE).

### Strengths
1. The time-aware input representation allows the model reason about intermittency without validity masks.
2. It extends Mamba to BTD-Mamba to preserve temporal coherence under missing observations.
3. On Argoverse 1 and 2 Datasets, the model outperforms mask-based baselines.

### Weaknesses
1. The notations in this paper are very confusing. Usually we use $x_t$ for position at time $t$, but this paper uses $t_x$ for position and $t_{vel}$ for velocity. The paper also uses $t_i$ for timestamps, $\Delta t_i$ for gaps, and $t’, t’’$ for temporal embeddings. 
2. The core ideas of time-aware input features (scaled timestamps, inter-observation gaps) and a Bidirectional Time-Decay Mamba are reasonable, but read as modest adaptations of known ingredients (temporal features + bidirectional/decay in SSMs) rather than a clearly new idea.
3. The baselines this paper is comparing against are old. For example, on the current Argoverse 2 leaderboard, top models can achieve MR$_6$ of 0.11 or 0.12, and minADE$_1$ of 1.42.
4. No complexity/runtime is given.

### Questions
There are many clear typos and small grammar issues. Here is a list:
1. Line 21: “Particularly, We extend” → “Particularly, we extend”
2. Line 40: “are are designed” → “are designed”
3. Line 41: “observations are often intermitten” → intermittent
4. Line 61: “variale-length” → “variable-length”
5. Line 72-73: “intervals feature” → “interval features”
6. Line 85: “to effectively captures” → “to effectively capture”
7. Line 107: “pedestrians, or cyclist” → “pedestrians, or cyclists”
8. Line 420: “Observations Lengths” → “Observation Lengths”

### Soundness
3

### Presentation
2

### Contribution
2
