# Learning to Anticipate: A Conditional Representation Fusion Network for Pre-Stroke Prediction

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 0, 6, 4

## Abstract
Predicting the future in dynamic environments requires reasoning about the in-
tentions of agents from rich, multi-modal data. We introduce a novel machine
learning problem: pre-intervention anticipation—forecasting outcomes before an
action is completed by fusing contextual cues with ongoing sensor data. To ad-
dress this, we propose ConFu, a general neural architecture featuring two key
innovations: (1) a conditional gating mechanism that dynamically modulates pri-
mary features (e.g., trajectory) based on secondary context (e.g., intention cues),
and (2) a cross-fusion strategy for systematic multi-stage integration of heteroge-
neous modalities. Our model achieves a prediction accuracy of 92.6% with
a mean absolute error of 0.20 meters, significantly outperforming existing
methods by 7.8-10.5% in accuracy. Experimental validation on a real-world
badminton dataset comprising 13,582 strokes demonstrates that ConFu provides
immediate tactical feedback, saving 85% decision time compared to trajectory-
based approaches. This time advantage is particularly valuable for practical appli-
cations such as enabling badminton robots to compute interception strategies.
Our work establishes a foundation for intention-aware prediction, with broader
implications for robotics, autonomous systems, and human-AI interaction. Code
will be released for reproducibility (https://anonymous.4open.science/r/AI-
Sport18-BFE9/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the problem of shuttlecock landing-point prediction in badminton, focusing on pre-stroke anticipation rather than post-stroke trajectory analysis. It proposes an architecture that integrates four feature streams extracted from monocular video: (1) 3D shuttlecock trajectory before impact, (2) player dynamic positions, (3) arm gesture features, and (4) stroke type labels. The model employs conditional gating and cross-fusion mechanisms to combine these multimodal cues and is trained using an L1 loss to predict the 2D drop point. Experimental results on real-world datasets show that the method outperforms prior methods, achieving higher accuracy and significantly lower latency by relying solely on pre-stroke information.

### Strengths
- The paper is among the first to address pre-stroke shuttlecock landing prediction, shifting the focus from post-stroke trajectory analysis to genuine anticipation.
- It leverages intuitive multimodal cues such as pre-stroke shuttlecock trajectory, player positions, arm gesture, and stroke type, to capture both motion dynamics and contextual intent.

### Weaknesses
- The method assumes access to stroke-type information during inference. It is unclear how these cues can be obtained in real time or how much latency or noise would be introduced if they must be inferred automatically.

- It remains uncertain how reliably stroke type can be determined within the 21 pre-stroke frames, given that similar early motions can lead to different shot outcomes ?

- It is not clearly stated whether baseline models were retrained to directly optimize drop-point regression or merely trained using their original training objectives then evaluated using the drop-point.

- Since the paper’s novelty centers on the fusion and gating mechanisms, a missing baseline is a naïve concatenation model using the same four modalities; such an ablation would clarify whether the gating actually contributes beyond simple feature aggregation.

### Questions
- I am not sure how it works in badminton, but do the majority of strokes cluster around standard regions (e.g., back corners or mid-court)? Could the authors provide more analysis on the distribution of landing points to clarify whether the metric might be influenced by spatial bias?

- How feasible is it to infer stroke type given the short input window? Rather than retraining a stroke-type classifier, could the authors simulate potential misclassification by introducing small errors in the stroke-type input (e.g., some labeled as smash instead of drop) and report how this affects performance?

- Since the paper’s novelty lies in the fusion and gating mechanisms, could the authors include an ablation or comparison against a naïve concatenation baseline using the same four modalities to better quantify the contribution of gating ?

- Were the baseline methods retrained specifically for the landing-point regression objective, or were they evaluated using their original task heads and loss functions? Clarifying this would help assess the fairness of the comparisons.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes ConFu (Conditional Gated Cross-Fusion Network) for multimodal pre-stroke anticipation in badminton, which achieves effective prediction quality on real-world badminton datasets while preserving efficiency.

### Strengths
1. This paper compares their proposed method with the state-of-the-art methods, demonstrating effectiveness on two real-world datasets.
2. The experiments are extensive and well studied, verifying the proposed claims and design choice for their method.
3. The paper is well writtent and easy to understand.

### Weaknesses
1. From my perspective, this paper should be submitted to application-oriented conferences (e.g., AAAI, KDD, etc.) instead of general-purpose conferences such as ICLR, which may have more interest on the broad applicability and modalities. Therefore, the ICLR community may share less interests to this work since it is more like an application paper.
2. While the authors claim unified multimodal integration as one of the contributions, existing badminton-related works have explored at least the unified representation for stroke trajectory, player position, and stroke types [1, 2]. Similarly, the dynamic and hierarchical fusion strategies was also proposed in [2]. Though the inner design may be different, the ideas are similar to some extent.

[1] ShuttleNet: Position-Aware Fusion of Rally Progress and Player Styles for Stroke Forecasting in Badminton. AAAI 2022.

[2] Where will players move next? dynamic graphs and hierarchical fusion for movement forecasting in badminton. AAAI 2023.

3. The authors simplify the stroke type to 4, which may be too simplified for evaluation that can be observed from the >80\% accuracy. Since the authors do not use datasets from BadOL as mentioned in L340, it remains unclear why the authors convert them to only 4 types.

4. Some experiment selections are unclear as described in the Questions section.

5. The model framework does not introduce much novelty. For instance, the module design seems to be a bit heuristic selection, e.g., using Transformer+CNN for 3D trajectory and conditional LSTM for gestures. It would be better if the authors could propose a unified and simplified structure so that it can be served as a fundamental architecture for future use. Additionally, it remains unclear about why not fusing all features together in the end, but performing "multi-stage" fusion.

### Questions
1. ShuttleSet consists of larger amount of data; why do the authors opt for ShuttleSet22, which focuses on a small portion matches, instead of ShuttleSet?
2. Why do the authors choose accuracy instead of measuring uncertainty for shot types (i.e., cross entropy from the ShuttleSet paper)?
3. Is there any reason that ShuttleNet is not included in Table 2 for comparison?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The article present a novel method to determine the landing point of shuttlecocks in badmington matches. Using different datasets and methods, the paper provide a multimodal fusion architecture that increase accuracy and inference time prediction using pre-shot data rather than entire trajectories. The modalities taken in consideration are the following: 3D trajectories, player positions, stroke type and keypoint tracking during gesture. Through the usage of transformer networks and LSTM, the work provide a methodology to compute fundamental fusions over widely distant modes. Surpassing current SOTA techniques the article improve accuracy and interpretability for this task.

### Strengths
1. Usage of LSTM and gating mechanism enhance interpretability, as stated and discussed in Figure 3 and section 3.4
2. Dataset collection is well explained. The paper even rewrites motion dynamics (Eq. 11) or the shuttlecock, addressing for miscalculated 3D points extracted with different physical settings  
3. Experiments are complete: inference time evaluation and comparison against other methods with different thresholds for distance

### Weaknesses
1. Claims over inference time speed do not address off-line preprocessing.
2. Choice of LSTM over better sequence model could be experimentally motivated.

### Questions
1. The paper claims that inference time is faster with respect to other networks such MonoTrack. Specifically, in Section 4.1 the article says "We employed the MonoTrack pipeline to extract features, including the reconstructed 3D trajectories, ...". Later, in Section 4.1.4 the paper says that MonoTrack and ShuttleNet primarily rely on reconstructing the shuttlecock's 3D trajectory. Given that the trajectories are precomputed, this cost does not appear at inference time for ConFu, but in a real settings this informations should be included and computed on the fly. Can the authors discuss this choice and address this concern including offline computation of different modalities?
2. The paper justify the usage of LSTM over Transformers saying that the latter require a lot of training data and have less interpretability. While that is true, the claim should be supported by small ablations or at least by a citation. Can the authors provide very few, small and demonstrative experiments using Transformers?

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
This submission proposed a new future-prediction method and its application to badminton shuttlecock's landing point prediction task.
The proposed method is termed Conditional Gate-Based Cross-Fusion Network (ConFu).
It exhibits a sensor-fusion approach using monocular video, players' location, gesture, and stroke types. 3D trajectories provided by a trajectory-reconstruction method MonoTrack is used for training, and physically plausible recalibration is also discussed.
Experiments are conducted using TrackNetV2 and ShuttleSet22, existing badminton-video datasets. CongFu performed better in drop point prediction accuracy.

### Strengths
- Analyses of the predictions, for example, distribution of 2D prediction points and gating, are well done and they are useful to understand the models' behavior beyond the accuracy.

- Application of the machine learning based prediction models for sports understanding is an interesting and promising research area. The domain knowledge, such as importance of stroke type and player position is nicely exploited.

- The system enables real-time inference, which is useful for deployment in industry.

### Weaknesses
-The design of the method is heavily on Although it is nice for an application paper, for a broader machine-learning audience, the insights from the method may be limited.

- The technical contributions in the neural architecture is not very large. The gating mechanisms have been studied well in single-modal settings (e.g., LSTMs and GRUs) and multimodal settings [a].
 [a] GATED MULTIMODAL UNITS FOR INFORMATION FUSION, ICLR2017 workshop

- Reliance on the aerodynamic calibration shown in Table 1 is a concern for real applications and the method's generalizability beyond badminton analyses, although it is well considered. If this part is performed in a learning-based manner, it would be a great contribution in machine learning venues.

### Questions
Did incorporating stroke types as a feature contribute to the overall performance? It might be redundant after inputting the gesture.

### Soundness
4

### Presentation
3

### Contribution
3
