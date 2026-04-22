# Social Interaction Modeling for Group Re-identification

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Group Re-identification (G-ReID) focuses on associating group images that contain the same members across different camera views. The key challenge is that identity and position differentiation in group topology structure changes are difficult to capture. Drawing on principles from social psychology, we observe that the core members are more likely to remain in the group under different camera views with smaller position changes, while peripheral members are more likely to have significant position changes or even fade out of the group. To this end, we propose a novel social interaction modeling (SIM) method, which treats each group as a social interaction field to explore more authentic and robust group features through dealing with the member differentiation: identity and position differentiation. Our method constructs the social interaction calculation module (SICM) to capture the member differentiation in the fields, and implements identity differentiation and position differentiation by the social prior attention mechanism (SPAM) and social layout variation module (SLVM), respectively. Extensive experiments on three available datasets show that the proposed method SIM is effective, and outperforms all previous state-of-the-art methods, surpassing the baseline on Rank1/mAP by up to 8.6\%/9.6\% on DukeGroup, 3.7\%/2.7\% on RoadGroup and 2.5\%/2.9\% on CSG. The code will be available on Github.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the core challenge in Group Re-identification (G-ReID), the difficulty of capturing identity differentiation and position differentiation amid changes in group topological structures. Drawing on social psychology principles (i.e., core members tend to have minimal position changes while peripheral members are prone to significant displacements or even exiting the group), the authors propose a Social Interaction Modeling (SIM) framework. SIM treats each group image as a "social interaction field" and implements differentiated modeling through a three-stage module pipeline: first, the Social Interaction Calculation Module (SICM) extracts interaction features (distance d, orientation \(\theta\), openness O) between members, quantifies interaction probabilities, and generates a normalized variable \(\hat{p}\) that reflects member differentiation; second, the Social Prior Attention Mechanism (SPAM) assigns higher attention weights to core members to achieve identity-differentiated learning; finally, the Social Layout Variation Module (SLVM) constructs a learnable position variation matrix \(\triangle D\) to simulate realistic dynamic layouts for position-differentiated learning. Experimental results show that SIM outperforms existing state-of-the-art (SOTA) methods in Rank-1/mAP metrics across three datasets (DukeGroup, RoadGroup, CSG).

### Strengths
1. The designed SICM-SPAM-SLVM module chain forms a logical closed loop: SICM provides a quantitative foundation for differentiated modeling, while SPAM and SLVM target identity and position differentiation respectively. Ablation experiments further demonstrate that "the performance gain of combining the two modules exceeds the sum of their individual gains", fully justifying the coordination and necessity of the modules.

2. The study covers three mainstream G-ReID datasets and compares two types of methods (handcrafted features such as CRRRO-BRO and deep learning-based methods such as UMSOT, PBSOT). Additionally, it uses parameter analysis (impact of hyperparameter \(\alpha\)), feature visualization (t-SNE dimensionality reduction), and attention visualization (GradCAM heatmaps) to validate the model’s effectiveness, enhancing the credibility of the results.

### Weaknesses
1. The modeling of "member differentiation" relies solely on Latane’s (1980) Social Impact Theory and Lewin’s (1943) Field Theory, without comparing more mature alternative theories in social science (e.g., "degree centrality/betweenness centrality" in social network analysis, "group belonging quantification methods" in social identity theory). The authors neither explain "why the selected classical theories are more suitable for the G-ReID scenario than modern theories" nor demonstrate "whether the core logic of the model remains valid if alternative theories are adopted", leading to inadequate justification for the rationality of theoretical selection.

2. The calculation of interaction probabilities in SICM is highly dependent on skeleton extraction tools (e.g., Mediapipe, AlphaPose). While the paper mentions "errors in skeleton extraction", it does not quantify the impact of such errors on performance (e.g., performance degradation curves when simulating "10%/20% skeleton key-point errors"), making it impossible to assess the model’s practical value for low-quality skeleton data.

3. Extreme scenarios commonly encountered in practical security applications (e.g., low light at night, member occlusion rates exceeding 50%, small-sample groups) are not covered, making it impossible to verify the model’s performance in complex real-world environments.

4. Ablation experiments only verify the roles of SPAM and SLVM, without a control experiment for "the independent effect of SICM". This makes it impossible to determine the independent contribution of SICM as the "quantitative foundation for differentiation modeling" (e.g., the extent of performance decline when SICM is removed).

5. The update mechanism of the learnable position variation matrix \(\triangle D\) in SLVM is not explained (e.g., how gradients are backpropagated, how \(\hat{p}\) constrains "minimal displacement for core members and maximal displacement for peripheral members"), resulting in vague mathematical logic.

### Questions
1. In social network analysis, "degree centrality" (measuring the number of connections between a member and others) and "betweenness centrality" (measuring a member’s role as a hub in group connections) are widely used to quantify member importance, and their calculation does not rely on skeleton data. Why did the authors not attempt to replace the classical theories of Latane and Lewin with such theories? If "degree centrality" is used to replace the interaction probability calculation in SICM, how would the Rank-1/mAP metrics of the model change across the three datasets? Please supplement a theoretical comparison analysis and validation experiments for alternative approaches.

2. Are the weights of the three interaction features (d, \(\theta\), O) in SICM determined through experimental tuning or theoretical derivation? If one of the features (\(\theta\) for orientation or O for openness) is removed, how much would the model’s performance decline? Can experiments prove that "the coexistence of all three features is optimal" rather than relying solely on distance d to meet the needs of differentiation quantification?

3. The paper mentions that "CSG contains 5K distractor samples", but it does not analyze how the retrieval efficiency of SIM changes with the increase in the number of distractor samples (e.g., 10K, 20K). Please supplement experiments on the relationship between "number of distractor samples, model inference time, and Rank-1 accuracy" to verify the practical scalability of SIM.

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
This paper proposes a social interaction modeling method for group re-identification. It integrates three key parts, including the social interaction calculation module to capture the member differentiation in fields, the social prior attention mechanism to accomplish identity differentiation, and the social layout variation module to address position differentiation. Experiments validate the effectiveness of the proposed method.

### Strengths
1.	The paper is clearly structured.
2.	The paper draws insights from social psychology principles and design novel methods for identity differentiation and position differentiation.

### Weaknesses
1.	In the methodology section, the authors mention they utilize multiple frameworks such as Mediapipe, AlphaPose, and HigherHRNet to extract skeletal keypoints. However, the paper does not clearly explain how these frameworks are merged or integrated. Providing details on the merging strategy would help improve the clarity and reproducibility of the proposed method.

2.	i-LIDS MCTS[1] dataset is a commonly used benchmark that has been widely adopted by methods such as MACG[2]. However, the authors did not report experimental results on this dataset. Including results on i-LIDS MCTS would make the evaluation more comprehensive and allow for a fairer comparison with prior work.

3.	Several grammatical, spelling, and mathematical notation errors were found throughout the paper. For example, “enote” should be corrected as “denote” in line 184. Parameter “m” has different meanings in line 247 and line 315, which may cause confusion. Careful proofreading is recommended to improve the overall clarity and presentation quality.

[1] W. Zheng, S. Gong, and T. Xiang. Associating groups of people. In British Machine Vision Conference, BMVC, pages 1–11, 2009.

[2] Yan, Y., Qin, J., Ni, B., Chen, J., Liu, L., Zhu, F., Zheng, W.-S., Yang, X., & Shao, L. (2023). Learning multi-attention context graph for group-based re-identification. IEEE TPAMI, 45(6), 7001–7018.

### Questions
Please refer to paper weaknesses.

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
This paper introduces Social Interaction Modeling (SIM) for Group Re-Identification (G-ReID), motivated by findings in social psychology that group members exhibit differentiated importance and positional stability within a group.
SIM models each group image as a social interaction field, capturing member differentiation through two core modules:
Social Interaction Calculation Module (SICM), which computes normalized interaction probabilities among members using distance, orientation, and openness cues derived from pose estimation.
Social Prior Attention Mechanism (SPAM), which applies these probabilities to weight member tokens in the attention layer, enhancing identity differentiation.

### Strengths
1. The paper provides extensive evaluations on three datasets, with consistent improvements over 10+ strong baselines, including recent transformer-based models such as UMSOT and PBSOT.

2. Each component (SICM, SPAM, SLVM) has a clear functional role. Ablation (Table 2) shows each module’s contribution, and the synergy when combined.

3. Introducing psychological insight into G-ReID provides an interesting interdisciplinary perspective, even if the connection is heuristic.

### Weaknesses
1. The method reuses known ideas (attention reweighting, layout augmentation) but describes them as new under social terminology. No mathematical or empirical evidence proves that “social interaction fields” outperform simpler geometric priors.

2. Experiments are restricted to small datasets (< 4 K images). The model’s scalability to larger or real-world scenes (e.g., street or crowd surveillance) is unclear.

3. The “core vs. peripheral members” metaphor lacks quantifiable validation. No user study or statistical correlation supports that interaction probabilities correspond to true social hierarchy.

### Questions
1. How do SPAM and SLVM differ from attention reweighting and layout-augmentation schemes already used in UMSOT or PBSOT? What unique mechanism validates the claim of “social interaction modeling”?

2. Have authors evaluated the accuracy of the computed interaction probabilities ​p_ij? For example, how consistent are they with human-annotated social centrality maps?

3. Since SICM relies on pose detectors, how does SIM perform when keypoint extraction fails or under heavy occlusion?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Social Interaction Modeling (SIM) for Group Re-Identification (G-ReID) to associate group images containing the same members captured by different cameras with non-overlapping views, as inspired by social psychology that treats each group as a social interaction field. A Social Interaction Calculation Module (SICM) is constructed to estimate member differentiation via interaction probabilities derived from spatial and pose-based features using two mechanisms, including the Social Prior Attention Mechanism (SPAM) for identity differentiation through adaptive attention weighting, and the Social Layout Variation Module (SLVM) for position differentiation via learnable layout perturbations. Experiments on three benchmark datasets (CSG, RoadGroup, and DukeGroup) demonstrate state-of-the-art results, demonstrating competitive performance.

### Strengths
1. This paper proposes a new idea of incorporating social psychology principles into G-ReID to model intra-group member differentiation through interaction probabilities.

2. The proposed SIM framework effectively achieves robust feature representations and consistent improvements across multiple benchmark datasets.

### Weaknesses
1. The technical innovations of the proposed model are vague, though the paper claims to be the first to introduce social interaction modeling. In fact, transformers have been widely adopted in modeling social interactions in applications like trajectory prediction, especially pedestrian trajectory prediction (for example [R1]-[R5] in recent years). The contributions of methodology to construct attention mechanisms and transformer architectures beyond using different inputs for group re-identification are not clarified and comparison to existing transformer architectures for social interaction modeling is also missing. 

2. The proposed model based on interaction probability is largely empirical. The formulation of the interaction probability and its claimed contribution to optimizing the attention matrix are not supported by adequate theoretical or mathematical analysis. The effect of the definition of interaction probability on member differentiation is not clear. Furthermore, it lacks of theoretical justification showing that directly applying the interaction probability to the attention matrix via Hadamard product enhances model performance.

3. The formulation lacks clarity and completeness, which limits the reproducibility of the proposed model. Several equations are difficult to interpret, and key algorithmic components are insufficiently explained. For example, the definition of Equation (5) is confusing, as the variable $S_{ij}$ appears on both sides of the equation, and the scalar is defined as the product of an Impulse function and an unspecified function $f(\cdot)$. In addition, $f(\cdot)$ seems essential for computing $p_{ij}$, but its actual form or implementation is not described in the paper.

4. Experimental evaluations are not sufficient to validate the proposed model. 

i) Performance comparison with prior works appears inconsistent with the results originally reported in those studies. Regarding the comparison with PBSOT, its original paper (Zhang et al., 2025) reports higher performance than the results shown in Table 1 of this submission (e.g., Rank-1 of 96.35 on CSG and 95.06 on RoadGroup), while several shared baselines (e.g., SOT) show consistent results across both papers. This raises the concerns on whether the results are convincing to support the effectiveness of the proposed model. 

ii) Moreover, maintaining a $p$ matrix matching the attention shape for every input is likely to incur significant computational overhead. However, related analysis and experimental evidence are not provided in the paper.

5. The writing of the paper should be improved. There are too many grammatical errors. Here is just an example of the first paragraph in Section 3.1: "In this paper, ... The key issue is extract social interactions features, and calculate interaction ... we treats it as ..., (missing conjunction) member i and j have ... determine whether pedestrians i and j (missing verb) in same social interaction field."

[R1] Liu, Yao, et al. "Social graph transformer networks for pedestrian trajectory prediction in complex social scenarios." Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022.

[R2] Wang, Zixu, et al. "SocialFormer: Social interaction modeling with edge-enhanced heterogeneous graph transformers for trajectory prediction." arXiv preprint arXiv:2405.03809 (2024).

[R3] Chen, Kai, et al. "SocialTrans: Transformer based social intentions interaction for pedestrian trajectory prediction." Physica A: Statistical Mechanics and its Applications 663 (2025): 130435.

[R4] Wang, Chengdong, et al. "SIAT: Pedestrian trajectory prediction via social interaction-aware transformer: C. Wang et al." Complex & Intelligent Systems 11.8 (2025): 335.

[R5] Liu, Yao, et al. "Attention-aware social graph transformer networks for stochastic trajectory prediction." IEEE Transactions on Knowledge and Data Engineering 36.11 (2024): 5633-5646.

### Questions
Please refer to the section of Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
