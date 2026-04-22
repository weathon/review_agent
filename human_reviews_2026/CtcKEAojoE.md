# Leveraging Shared Prototypes for a Multimodal Pulse Motion Foundation Model

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Modeling multi-modal time-series data is critical for capturing system-level dynamics, particularly in biosignals where modalities such as ECG, PPG, EDA, and accelerometry provide complementary perspectives on interconnected physiological processes. While recent self-supervised learning (SSL) advances have improved unimodal representation learning, existing multi-modal approaches often rely on CLIP-style contrastive objectives that overfit to easily aligned features and misclassify valid cross-modal relationships as negatives, resulting in fragmented and non-generalizable embeddings. To overcome these limitations, we propose ProtoMM, a novel SSL framework that introduces a shared prototype dictionary to anchor heterogeneous modalities in a common embedding space. By clustering representations around shared prototypes rather than explicit negative sampling, our method captures complementary information across modalities and provides a coherent “common language” for physiological signals. In this work, we focus on developing a Pulse Motion foundation model with ProtoMM and demonstrate that our approach outperforms contrastive-only and prior multimodal SSL methods, achieving state-of-the-art performance while offering improved interpretability of learned features.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents ProtoMM, a prototype-based self-supervised learning method for biosignals (ECG, PPG). ProtoMM aligns views of one sample of the same and different modalities, exploiting within- and between-modality information.

### Strengths
- The paper is well-written and easy to follow
- The experimental setup is clear showing the advantages of the method compared to the baselines

### Weaknesses
1/ Outdated baselines.

1a/ Foundation models: The main weakness of the work is the fact that the baselines are outdated. There exist several modern works that use foundation models for biosignals; in all cases the procedure is different to the one proposed in this work, but we cannot understand how well the proposed method works without modern baselines. 

[a] Xu et. al.,  RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data, ICLR 2025
[b] Abbaspourazad, et. al., Large-scale Training of Foundation Models for Wearable Biosignals, ICLR 2024
[c] Narayanswamy et. al., Scaling wearable foundation models, ICLR 2025 
[d] Saha et. al., Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications across Lab and Field Settings, ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 

1b/ Self supervised methods: 
There exist several more recent SSL methods operating on biosignals that could be explored or adapted. See some examples below: 

[e] Tag et. al., Electrocardiogram Report Generation and Question Answering via Retrieval-Augmented Self-Supervised Modeling, ICASSP 2025
[f] Mordacq et. al., ADAPT: Multimodal Learning for Detecting Physiological Changes under Missing Modalities, MIDL 2024 
[g] Shen et. al., CIMSleepNet: Robust Sleep Staging over Incomplete Multimodal Physiological Signals via Contrastive Imagination, NeurIPS 2024 

Given the 1a/ and 1b/ categories, right now the proposed method seems standalone without proper bibliographic discussion and without convincing baselines and comparison of other methods.

2/ Modalities. 
The paper has evaluated only two modalities, which makes claims such as “general multimodal framework” not convincing. It would be useful to have at least one higher-level modality (e.g., EDA, ECG, or text). 

3/ Datasets. 
The datasets seem outdated and it seems that there exist newer and more within scope datasets: 
[i] EEVR: A Dataset of Paired Physiological Signals and Textual Descriptions for Joint Emotion Representation Learning, NeurIPS 2024
[ii] WildPPG (A Real-World PPG Dataset of Long Continuous Recordings), NeurIPS 2024 
[iii] Stressid: a multimodal dataset for stress identification, NeurIPS 2023 
 
4/ Methodology
While intuitively well-motivated, the paper lacks a deeper theoretical justification of why prototype consistency across modalities results in better disentanglement or regularization. Furthermore, there is no discussion on the convergence or stability of the Sinkhorn-based assignment. 

5/ Missing ablations
- Several ablations are missing (hyperparam robustnes, number of prototypes, temperature, choice of \alpha, etc). These could really help understand what works and why in the proposed method. 
- It remains unclear whether ProtoMM embeddings are better than contrastive baselines so a baseline with contrastive learning would help.. 

6/ Frozen encoders. 
All downstream evaluations use frozen encoders with linear probes thus making the method dependent on the quality of the encoder. 

7/ Visualizations and analysis. 
Although there exist some visualizations, we cannot understand the importance of training with both losses neither the importance of prototypes. It would have been helpful to have these. For example, we cannot understand if the shared prototype space is truly modality-agnostic, or if some modality-specific subclusters have emerged. 

8/ Prototypes.
Prototypes are the main contribution of the work but we cannot understand how they operate. Besides visualization, we would now what happens for instance when we have one dominant modality or how many we need to have or if we need additional latents to represent them or what would happen if we have longer sequences. Analysing them and having findings transferable to larger datasets and other modalities would strengthen the work.  

9/ Failure cases are not discussed.

### Questions
It would have been useful to have answers to the weaknesses above. 
Some additional questions are: 

Q1: it would be interesting to have the computational overhead of the Sinkhorn-based prototype updates compared to contrastive methods

Q2 (W6) How does ProtoMM perform when fine-tuned end-to-end on downstream tasks, compared to linear probing?

Q3 Are the learned prototypes stable across training runs, or do they vary significantly with initialization?

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
4

### Summary
This work introduces ProtoMM, a self-supervised multimodal model for time-series data that uses a shared prototype dictionary to align embeddings across modalities and capture both shared and modality-specific information. The model consistently outperforms unimodal and multimodal baselines in experiments across three datasets and downstream tasks, demonstrating strong interpretability and generalization.

### Strengths
1. The shared prototype dictionary provides an innovative alternative to contrastive learning methods, effectively addressing the limitations in negative pair construction in multimodal settings.

2. The model considers both shared and modality-specific information, deriving a shared cross-modal representation while preserving unique modality information.

### Weaknesses
1. Although the proposed method outperforms most baselines, the improvements are relatively small. Reporting standard deviations would help assess the statistical significance.

2. In Table 2, results are shown only for the alpha=0.5 setting and two extreme cases (0 and 1). Additional intermediate values should be included to verify whether the performance trend is consistent rather than random.

3. Same for Table 3. It would be great if the authors could show the results across a wider range of choices of alpha.

### Questions
Within-modality samples and cross-modality pairs are different. The within-modality samples are derived from augmentations, while each modality is from distinct sources. Therefore, the semantic meaning of distance (or similarity) differs. 
Has the model design considered this discrepancy?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present ProtoMM, a prototype-based self-supervised framework for multimodal time-series learning, designed specifically for pulse motion foundation models using PPG (photoplethysmography) and accelerometry data. The key idea is to replace traditional contrastive alignment (which requires explicit positive/negative pair mining) with a shared prototype dictionary that serves as semantic anchors across modalities. Each signal modality is encoded separately, projected into a shared prototype space, and trained via a Multimodal Prototype Prediction Loss, combining within-modality and between-modality consistency objectives. Empirical results across three datasets and six downstream tasks show that ProtoMM often outperforms unimodal and multimodal baselines. The authors also provide qualitative evidence of interpretability, showing that learned prototypes correspond to meaningful physiological and behavioral states.

### Strengths
- The paper presents a novel adaptation of SwAV’s prototype mechanism to a multimodal, multi-view learning setting.
- The study is highly relevant, addressing a practical and timely challenge in wearable sensing and multimodal foundation modeling.
- The proposed objective is generalizable, featuring a tunable parameter $\alpha$ that effectively controls the trade-off between within- and between-modality learning, with strong empirical support.
- The experimental evaluation is comprehensive, spanning multiple datasets and benchmarks under a consistent architecture with fair and transparent comparisons.
- The learned prototypes shows correspondence with semantically meaningful physiological patterns that enhance understanding of model behavior.
- The paper is clearly written, well-structured, and effectively motivates the proposed approach.

### Weaknesses
- The theoretical grounding of the approach is limited, as the benefits of prototypes over contrastive losses are supported mainly by empirical evidence. A formal analysis of how prototypes mitigate false negatives or enhance cross-modal alignment would substantially strengthen the work.
- The mathematical and biological motivation for the proposed Multimodal Prototype Prediction Loss is underdeveloped, with only a brief intuition provided in lines 194-195. A more detailed justification would better support the design and relevance of this loss function.
- Several closely related baseline methods (e.g., SLIP, FOCAL) are introduced only in the experimental section, rather than in the related work discussion. Including them earlier would clarify the paper’s positioning and help readers assess the novelty of the proposed approach more fairly.
- The ablation study is somewhat shallow, as it explores only three α values (0, 0.5, and 1). Testing a finer range would yield more informative results regarding the model’s sensitivity and stability.
- The interpretability evaluation is purely qualitative, lacking quantitative metrics such as clustering purity or correlations with physiological labels. Moreover, interpretability is not compared against baseline models, making it unclear whether these insights are unique to the proposed method.
- Minor typos in lines 058, 073.

### Questions
- How were the hyperparameters, such as $\alpha$ and the encoder architecture parameters, selected? Were they tuned empirically or determined through heuristic choices?
- In equations (5)-(7), there appear to be $MA^2$ individual within-modality losses and $M^2A^2$ between-modality losses. Would this not still lead to an imbalance between the two loss types, particularly when $\alpha = 0.5$?
- Could the authors clarify the definitions of $\mathbf{z}_t$ and $\mathbf{q}_s$ in equation (4)?
- In Lines 361–364, why would the hypothesized advantages of the prototype-based approach not extend to unimodal settings, especially given that empirical results show contrastive methods outperforming the prototype-based variant?
- In the interpretability experiment, could the authors provide quantitative validation showing that the learned prototypes correspond to specific physiological states?

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
3

### Summary
This paper introduces ProtoMM, a self-supervised learning framework for multimodal time-series
data that addresses the limitations of traditional methods in modality alignment by using a shared
prototype space. ProtoMM outperforms existing methods in several tasks, particularly in stress
detection and activity recognition. Additionally, prototype visualization enhances the model's
interpretability. This approach offers an innovative solution for multimodal self-supervised
learning

### Strengths
The ProtoMM framework addresses the issue of negative sample sampling in
multimodal self-supervised learning by introducing a shared prototype space. Particularly in the
application of biosignals, ProtoMM effectively captures complementary information both within
and between modalities, providing a novel solution.


The model framework is highly versatile and can seamlessly be applied to different types
of time-series modalities.

### Weaknesses
Although ProtoMM outperforms the existing baseline models on some metrics, its
performance improvement is very limited (about 0.01-0.02), and it is difficult to determine
whether the improvement is due to the effect of the method itself or the experimental
randomness;


 The paper claims that ProtoMM can simultaneously capture within-modality (unique)
and between-modality (shared) information, but does not design a direct and objective
experiment to verify it. The authors only infer indirectly that the model learns on both types
of information based on the optimal model performance when α=0.5, which lacks support.

### Questions
How can you prove that ProtoMM can effectively capture intra and inter-modality information?

How do you demonstrate that prototyping is more effective on biological time series data?

### Soundness
2

### Presentation
3

### Contribution
2
