# Reading the Room: Learning Group States Beyond Pooled Individual Signals

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
The fundamental challenge in modeling group dynamics is that collective states arise from interdependent processes that violate the standard assumption of independent observations. We study this in a triadic Collaborative Problem-Solving (CPS) task where five constructs (group synchrony, group confidence, group interaction phase, individual engagement, individual leadership) are annotated as directional trends (increase / stable / decrease) over short windows. We formulate a hypothesis that group-level states are not reliably recovered from pooled individual features due to the aggregation fallacy. To test this, we introduce Syntality, a benchmark with participant-indexed multimodal streams and paired individual+group trend labels, and SyntalNet, an architecture that satisfies three minimal requirements: (i) permutation-equivariant cross-participant fusion, (ii) mask-aware intra-modality fusion, and (iii) low-rank cross-modal interactions. On Syntality, SyntalNet consistently outperforms additive baselines, improving group-level macro-F1 from 0.37–0.49 to 0.62 while also achieving strong individual-level performance (e.g., 0.64 balanced accuracy / 0.85 AUROC / 0.63 F1 on engagement; 0.60 / 0.78 / 0.58 on leadership). Under 5-fold cross-validation we show that gains are statistically significant. In a leave-one-group-out (LOGO) setting, zero-shot macro-F1 on held-out teams recovers to 0.47 when fine-tuning on 25\% of the target group, and to 0.58 when freezing the encoder and adapting only classifier heads. Ablation studies confirm that explicit cross-participant fusion, intra-modality fusion, and low-rank cross-modal fusion each contribute to robustness under various corruption scenarios. Critically, our results provide empirical evidence that pooled individual signals yield performance statistically equivalent to constant predictors on group states in this triadic CPS setting, highlighting the necessity of explicit cross-participant modeling. We will release our dataset with processed features, code repository, and models upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper addresses the challenge of modeling group dynamics, where collective states emerge from interdependent interactions, making standard aggregation of individual features insufficient. The authors formalize this as a prediction problem for five constructs: group synchrony, group confidence, interaction phase, individual engagement, and leadership. To tackle this, they propose SyntalNet, a neural architecture with three main components:
a) Mask-aware encoders to handle missing modalities,
b) Permutation-equivariant cross-participant context gating (SoSE-X) to model interdependencies,
c) Low-rank cross-modal fusion (GLRX) to efficiently capture interactions across modalities.
They introduce Syntality, a dataset derived from triadic interactions with temporally smoothed crowd annotations, formulated as directional trend prediction (increase/stable/decrease). Using 10-fold Leave-One-Group-Out cross-validation, SyntalNet outperforms baselines (e.g., Temporal CNN) across all constructs and evaluation metrics. Notably, pooled individual features perform no better than constant predictors, underscoring the necessity of explicitly modeling cross-participant interactions.

### Strengths
- They introduce Syntality, derived from triadic interactions with temporally-smoothed annotations for five constructs.
- They report that SyntalNet outperforms baselines, including Temporal CNN, across several metrics.

### Weaknesses
1)	The paper overlooks several important earlier works that are essential for connecting established practices with this paper. For instance, it should reference the foundational leadership studies by Daniel Gatica-Perez, as well as the cohesion-related works by Giovanna Varni. Relevant contributions by Maja Pantic and Alessandro Vinciarelli also cover aspects closely related to this study. The absence of such background literature leaves many sentences insufficiently supported by citations.

2)	The Related Work section focuses primarily on affective computing (e.g., stress, trust, emotions, measures of work load (tireness, stress), arousal…), where the analyzed signals are typically much shorter than those encountered in social signal processing phenomena. Many of the papers cited in this section appear to be only loosely related or even irrelevant to the current study. To better align this section with the paper’s scope—particularly regarding group dynamics—the authors could benefit from incorporating insights from a recent survey paper such as: [A] Beyan, Cigdem, Alessandro Vinciarelli, and Alessio Del Bue. "Co-located human–human interaction analysis using nonverbal cues: A survey." ACM Computing Surveys 56.5 (2023): 1-41.

3)	While the proposed framework is well-structured and integrates several established components, I find limited technical novelty in the design described. The key elements: mask-aware encoders, Squeeze-and-Excitation style modules for participant interactions, low-rank cross-modal fusion, and DeepSets compatible prediction heads are all based on existing and well-studied concepts. The paper appears to combine these techniques in a coherent manner rather than introducing a fundamentally new mechanism or theoretical contribution. 

4)	The evaluation setup appears quite limited in scope. The experiments are conducted on the Syntality dataset, which includes only 10 triads (approximately 170 minutes of data). This is a very small sample size for drawing strong or generalizable conclusions, especially for a method claiming to model complex group-level or social dynamics. Moreover, it is unclear why existing, larger datasets (see [A]) many of which cover similar social or multimodal interaction scenarios, were not used or discussed. The authors should justify this choice and explain how such a limited dataset can meaningfully support the claims of the study.

5)	The listed contributions are clearly structured, but they appear to reflect a combination of existing methodological components rather than introducing a fundamentally new formulation or architectural mechanism. The formulation point restates well-known challenges in modeling group interactions, and the architectural modules seem to adapt established concepts (e.g., SE blocks, low-rank fusion, DeepSets) with minor variations. The empirical evidence, although positive, is based on a very limited dataset, making it challenging to assess the approach's generality. Overall, the contributions section would benefit from clarifying what is genuinely novel, conceptually, methodologically, or empirically, beyond the integration of existing techniques.

6)	The text contains a broken reference (“see ??”), which prevents readers from accessing important implementation details. Further, I am not able to spot implementation details; the given ones belong to the dataset's cues.

7)	The claim regarding dataset availability is inaccurate. The paper states that there is “no public dataset of collaborative problem solving,” which is not correct. Several publicly available datasets exist that capture collaborative or group problem-solving scenarios. The authors should revise this statement to accurately reflect prior resources and position their dataset accordingly.

8)	The description of inter-rater agreement is unclear. It is not specified which metric was used to assess agreement, whether it is percent agreement, Cohen’s kappa, Fleiss’ kappa, or another statistical measure. Additionally, the meaning of “66.7% threshold for strong labels” and how it relates to the reported percentages for the five classes is ambiguous. Clarification is needed to understand the reliability of the annotations.

9)	The reported weak correlation between individual and group states, alongside strong links between group states and synchronized behaviors (e.g., mutual attention, shared gaze), suggests that modeling individual features alone is insufficient for predicting group-level outcomes. This emphasizes the need for methods that explicitly capture inter-participant interactions and temporal synchronization. Given this, the paper should clarify how the proposed model effectively leverages these synchrony signals and discuss potential limitations due to the small dataset size.

10)	The baseline comparisons are somewhat limited and raise concerns about fairness. Logistic Regression uses hand-crafted group features, while SyntalNet leverages learned multimodal representations, potentially biasing results. It is also unclear whether the LSTM, Temporal CNN, and VLM baselines were fully optimized for the small dataset (no implementation details supplied), which could exaggerate SyntalNet’s performance advantage. The authors should justify baseline selection and tuning, and consider including alternatives like DeepSets or cross-modal attention models that are more directly relevant to group-level modeling.

11)	The paper presents results in a single table without any ablation studies. While some components (e.g., SoSE-X, GLRX, DeepSets heads) may not be directly removable, the authors could still assess their contribution by replacing them with reasonable alternatives or simpler versions. Ablation experiments of this kind are necessary to understand which parts of SyntalNet drive the reported improvements and to evaluate the necessity of each module.

### Questions
Please see the weaknesses. Importantly:

Could you clarify which inter-rater agreement metric was used and how the “66.7% threshold for strong labels” relates to the reported class percentages?

Can you explain the intended contribution of each SyntalNet component (SoSE-X, GLRX, DeepSets heads) and why these design choices were made, given that no ablation studies are included?

How does the model leverage synchronized behaviors to predict group states, and how does this relate to the weak correlation between individual and group features?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a machine learning model to infer five constructs related to human interaction: two individual-level constructs and three group-level constructs. The authors aim to demonstrate that group-level constructs cannot be inferred from individual observations alone, but rather through collective states that emerge from interactions between partners. In addition to their model, they manually annotate a publicly available dataset to create grounded labels for model training. They compare their results against various baseline models and find that their proposed model outperforms them.

### Strengths
The proposed architecture is able to encompass data from different modalities, each with its own time scale, dimensions, and preprocessing steps. There are also distinct architectural components designed to fuse information across different dimensions, both across individuals and across modalities, which may serve as inspiration for future work on group interaction. Additionally, the annotated dataset, created by multiple annotators through an online platform, is a valuable contribution of this paper and could benefit future work if made publicly available.

### Weaknesses
There are no ablation studies included in the paper. The authors claim that their findings show emergent team states cannot be reduced to pooled individual signals, but this is not demonstrated, as all reported results are based on their full-fledged model. For this claim to be supported, I would expect to see lower group-level prediction performance when cross-module components are removed. The architecture is also quite complex, making it difficult to assess the individual contribution of each component without ablation. Additionally, while the abstract mentions using 10-fold Leave-One-Group-Out cross-validation, this detail is missing from the evaluation section. The authors also omit reporting variance in their results, which makes it hard to assess statistical significance. Finally, the paper does not cite relevant prior work on multimodal dynamic systems for group interaction and group-level construct inference, such as [1] and [2].

[1] Paulo Soares, Adarsh Pyarelal, Meghavarshini Krishnaswamy, Emily Butler, and Kobus Barnard. 2024. Probabilistic modeling of interpersonal coordination processes. In Proceedings of the 41st International Conference on Machine Learning (ICML'24), Vol. 235. JMLR.org, Article 1867, 45906–45921.

[2] Moulder, R. G., Duran, N. D., & D'Mello, S. K. (2022). Assessing Multimodal Dynamics in Multi-Party Collaborative Interactions with Multi-Level Vector Autoregression. In ICMI 2022 - Proceedings of the 2022 International Conference on Multimodal Interaction (pp. 615-625). (ACM International Conference Proceeding Series). Association for Computing Machinery. https://doi.org/10.1145/3536221.3556595

### Questions
# Questions
1. Why predicting trends on 10-s windows, 3-s stride and 19-s observation? I just want to understand how those numbers were chosen.
2. In L157,  the authors say "...observe 10 clips, and each clip was annotated by at least 3 participants". What's the duration of each clip and how did you guarantee that all clips have some relevant information to be extracted? Or were some clips discarded after labeling?
3. What do the increase, flat and decrease actually mean for the labels? Does increase here mean that label score in clip t < label score in clip t+1?

# Typos
L106. In paralle -> In parallel

L146. Broken link

L189. Distribution of the label 4. -> The distribution of labels is in Figure 4

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
The paper seems to propose a new video dataset capturing a problem-solving task between a group of people to analyze the group synchrony, confidence, interaction phase, and individual engagement and leadership. A baseline is proposed to solve the task and compare against temporal CNN, LSTM, and VLM.

### Strengths
The paper proposes a new video dataset capturing a problem-solving task between a group of people, which seems new to me, and many efforts have been put into building the data.

### Weaknesses
The paper seems hard to follow for new readers, especially since it is proposing a new dataset and (seemingly) a new task. From my view as a computer vision researcher, the paper does not well formulate the technical problem (not the participant's task). I do not know what the group dynamic is or what the expected visualization of the output is.

It can be fixed if the author can include some better visualization of prediction, annotation, eg, a confusion matrix, or distribute the dataset at the submission time, but unfortunately, it was not. I have a feeling that classification categories are LEFT/MIDDLE/RIGHT associated with participant social roles are just a handful number of classes, and that is limited.

Figure 1 is not informative in showing the features it aggregates or the mathematical operations.

There is no clear comparison with previous datasets to help position the proposed task and method within existing research. From the perspective of established computer vision and temporal modeling tasks, numerous state-of-the-art approaches already exist in time-series estimation and scene-level categorical reasoning (e.g., Scene Graph Generation). In this context, the experiment and method appear limited in novelty and technical innovation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SyntalNet, a multimodal architecture designed to model emergent group-level (synchrony/confidence/interaction phase) and individual-level (leadership/engagement) social trends in small-group interactions. SyntalNet addresses the limitations of existing approaches, which fail capture inter-participant dependencies or handle modality and permutation challenges, through specialized modules for cross-person (SoSE-X), intra-modality (GSX), and cross-modal (GLRX) fusion.
To support evaluation, the authors construct Syntality, a benchmark dataset derived from the Weights Task Dataset, augmented with crowdsourced annotations for five constructs and formulated as a directional trend prediction problem (increase/stable/decrease).
Experiments show that SyntalNet significantly outperforms LSTM, Temporal CNN, and VLM baselines.

### Strengths
1.	Well-Motivated Architecture: The design of SoSE-X and GLRX is well motivated by the proposed need for permutation invariance and mask-aware multimodal fusion.
2.	Good Empirical Results: SyntalNet outperforms compared baselines across multiple constructs and metrics, with especially large gains for group-level constructs.
3.	Interdisciplinary Impact: The paper bridges social psychology, multimodal learning, and group cognition, an emerging and impactful direction for socially intelligent AI.
4.	Reproducibility: The authors commit to releasing code, features, and annotations, aligning well with the openness and reproducibility ethos.

### Weaknesses
1.Problem Formulation: Segmenting interactions into fixed 19-second clips for modeling group dynamic trends over one activity that lasts 9–34 minutes lacks validity. It is unclear whether such a short timescale meaningfully captures emergent group phenomena or introduces noise from overly limited temporal context.
2.Architectural Novelty: The proposed components appear to be adaptations of existing designs, and the paper lacks clear ablation studies demonstrating how each module contributes to performance or uniquely supports the modeling of group dynamics. The contribution appears engineering-heavy but conceptually shallow.
3.Evaluation Limitations:
a)The experimental baselines are relatively weak (logistic regression, LSTM, CNN). There are no comparisons against recent multimodal video understanding frameworks that can be far more relevant.
b)The Syntality dataset covers only a single collaborative task (block-weight inference), leaving the generalizability of SyntalNet to other forms of social interaction or team contexts uncertain.

### Questions
1.	Given that original sessions last up to 34 minutes, how do 19-second clips sufficiently capture higher-level constructs like group confidence or leadership, which often manifest at slower temporal scales?
2.	Can the authors provide ablation results or qualitative analyses to illustrate the impact of SoSE-X /GLRX and GSX? Also, how do these modules interact, are improvements additive or synergistic?
3.	While the appendix discusses statical associations with behavioral patterns, how do the proposed group trends relate to the existing behavior measurements on group collaboration, such as the Collaborative Problem Solving (CPS) Facets originally included in the Weights Task Dataset?
4.	Missing reference at line 146
5.	All figures in appendix are low-resolution.

### Soundness
3

### Presentation
2

### Contribution
3
