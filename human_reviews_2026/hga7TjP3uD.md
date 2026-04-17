# A Unified Framework for EEG–Video Emotion Recognition with Brain Anatomy Guidance

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Recent studies in video- and EEG-based emotion recognition have shown notable progress. However, multi-modal emotion recognition remains largely unexplored, particularly the integration of physiological signals with video. This integration is crucial, as EEG–video fusion combines observable behavioral cues with internal neural dynamics and enables a more comprehensive and robust characterization of human emotion. To this end, we propose EVER, a novel EEG–Video Emotion Recognition framework that effectively integrates complementary information from both modalities. Specifically, EVER employs a Brain anatomy-aware Inter-modal Hierarchical Graph Convolution Network (BIH-GCN), which aggregates EEG channel features into region-level representations guided by anatomical priors. These region-level features are combined with global EEG and video embeddings to form a unified representation for emotion classification. Furthermore, we introduce a correlation-based distribution alignment loss to reconcile modality-specific embeddings and reduce cross-modal discrepancies. To provide a comprehensive evaluation, we conduct comprehensive benchmark across three public EEG-video paired datasets---Emognition, MDMER, and EAV. We evaluate 12 representative models, consisting of 5 EEG-only, 5 video-only, and 2 audio-video models, and report their performance under EEG, video, and EEG–video settings. Our benchmark highlights the strengths and limitations of both unimodal and multi-modal approaches across diverse environments. Extensive experiments demonstrate that the proposed EVER achieves state-of-the-art performance by jointly modeling behavioral cues from video and physiological responses from EEG, thereby enabling the recognition of emotional patterns unattainable by either modality alone.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propose a multi-modal emotion recognition framework EVER, which integrates EEG and video modalities to recognize emotional patterns. EVER employs a BIH-GCN module to combine region-level features with global embeddings for emotion classification. The framework is evaluated on three public EEG-video datasets, reporting improvements over previous works in emotion recognition task.

### Strengths
1. The authors introduce a multimodal framework for emotion classification, which, through extensive experiments, is shown to outperform existing unimodal approaches.
2. To mitigate the modality gap, this paper propose an alignment module which transforms covariance into correlations to reconcile feature distributions.
3. The paper presents its methodology with exceptional clarity. The hierarchical design of EVER is described in a structured, step-by-step manner.

### Weaknesses
1. The manuscript lacks sufficient detail and justification for certain hyperparameters. For example, the parameters $w_e,w_f, w_{align}$ are presented without any theoretical or empirical justification.
2. The brain region partitioning in BIH-GCN relies on a fixed anatomical mapping based on the standard 10–20 system, without accounting for individual anatomical variability or handling missing or sparse EEG channels. This rigid grouping may degrade performance in real-world scenarios where electrode placement varies across subjects or devices
3. The video branch employs AdaMAE pretrained on VoxCeleb—a dataset designed for speaker identification—which may bias the model toward identity-related facial attributes (e.g., hairstyle, skin tone) rather than affective cues such as subtle facial muscle movements, potentially limiting its sensitivity to genuine emotional expressions.

### Questions
1. In this paper the authors adpot AudioMamba, which is designed for audio, as the EEG encoder. Why not direct empoly an encoder specifically designed for EEG signal? How does the model perform when using raw EEG signals instead of the spectrogram employed in this paper?
2. Why are only the refined global embeddings used for final classification, while the region-level embeddings are excluded from the prediction head?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes **EVER**, a unified EEG–video emotion recognition framework that integrates behavioral cues from video with neurophysiological information from EEG. The core contribution lies in a **brain anatomy-aware multimodal hierarchical graph convolutional network (BIH-GCN)**, which aggregates EEG channels into anatomically defined regions, combines them with global EEG and video embeddings, and performs structured cross-modal message passing. Furthermore, a **correlation-based distribution alignment loss** is introduced to reduce cross-modal discrepancies. Experimental results demonstrate that EVER achieves state-of-the-art performance.

### Strengths
The method is well-designed, integrating anatomical priors with multimodal fusion and alignment.
A comprehensive benchmark covering three datasets and twelve baselines is provided, with consistent evaluation metrics leading to the best results.

### Weaknesses
1. Novelty: The proposed method shows limited novelty. The BHI-GCN mentioned in the paper is a commonly used approach. Please clarify how Stage 1 of BIH-GCN differs from existing methods; and demonstrate the superiority of Stage 2 compared to fully connected or other GCNs that also consider both global and local connections. (e.g., [1], [2], [3])

2. Related Work:
   - The paper lacks references to existing EEG+video fusion methods, such as [4] and [5].
   - The GCN section contains some ambiguities: the proposed framework also relies on fixed brain region definitions and does not resolve cross-dataset applicability. For datasets with different EEG device layouts, the model still requires manual adjustment.

3. Experimental Section: The results only report mean values without standard deviations, making it difficult to assess the model’s stability under data distribution shifts.

4. The paper lacks comparison with dedicated EEG-based emotion recognition models.

5. It is recommended to include recent baselines from 2025 for fair comparison.

6. In the ablation study, it is suggested to add comparisons with different alignment methods.

[1] Wang J, Ning X, Xu W, et al. Multi-source Selective Graph Domain Adaptation Network for cross-subject EEG emotion recognition[J]. Neural Networks, 2024, 180: 106742.

[2] Jin M, Du C, He H, et al. PGCN: Pyramidal graph convolutional network for EEG emotion recognition[J]. IEEE Transactions on Multimedia, 2024, 26: 9070-9082.

[3] Ye M, Chen C L P, Zhang T. Hierarchical dynamic graph convolutional network with interpretability for EEG-based emotion recognition[J]. IEEE transactions on neural networks and learning systems, 2022.

[4] Jin X, Xiao J, Jin L, et al. Residual multimodal Transformer for expression‐EEG fusion continuous emotion recognition[J]. CAAI Transactions on Intelligence Technology, 2024, 9(5): 1290-1304.

[5]Lin N, Gao W, Li L, et al. vEpiNet: A multimodal interictal epileptiform discharge detection method based on video and electroencephalogram data[J]. Neural Networks, 2024, 175: 106319.

### Questions
1. Novelty: The proposed method shows limited novelty. The BHI-GCN mentioned in the paper is a commonly used approach. Please clarify how Stage 1 of BIH-GCN differs from existing methods ; and demonstrate the superiority of Stage 2 compared to fully connected or other GCNs that also consider both global and local connections.(e.g., [1], [2], [3])

2. Related Work:
   - The paper lacks references to existing EEG+video fusion methods, such as [4] and [5].
   - The GCN section contains some ambiguities: the proposed framework also relies on fixed brain region definitions and does not resolve cross-dataset applicability. For datasets with different EEG device layouts, the model still requires manual adjustment.

3. The experimental results present only mean values without standard deviations, making it impossible to assess the model’s stability under variations in data distribution.

4. Lack of appropriate baselines:
   - Add EEG+video fusion baselines for fair comparison.
   - Include dedicated EEG-based emotion recognition models as additional baselines.
5. In Table 2, EVER introduces a large increase in the number of parameters compared to the EEG-only models, yet the performance improvement is limited on some datasets. Please explain the rationale and significance behind this design choice.

6. Please explain why in Table 3 the combination of AdaMAE and AudioMamba results in lower performance, and why adding the alignment module yields a notable improvement only on Emognition (W-F1), while results on other metrics and datasets remain unchanged or even decline.

7. Ablation study:
   - Please include ablation results on the MDMER dataset in Table 4 to further verify the effectiveness of BIH-GCN.
   - To demonstrate the contribution of Stage 2 in BIH-GCN, it is recommended to add an additional ablation where the current three graphs are replaced by a single fully connected graph (treating the global EEG node, global video node, and all brain-region nodes as connected).

[1] Wang J, Ning X, Xu W, et al. Multi-source Selective Graph Domain Adaptation Network for cross-subject EEG emotion recognition[J]. Neural Networks, 2024, 180: 106742.

[2] Jin M, Du C, He H, et al. PGCN: Pyramidal graph convolutional network for EEG emotion recognition[J]. IEEE Transactions on Multimedia, 2024, 26: 9070-9082.

[3] Ye M, Chen C L P, Zhang T. Hierarchical dynamic graph convolutional network with interpretability for EEG-based emotion recognition[J]. IEEE transactions on neural networks and learning systems, 2022.

[4] Jin X, Xiao J, Jin L, et al. Residual multimodal Transformer for expression‐EEG fusion continuous emotion recognition[J]. CAAI Transactions on Intelligence Technology, 2024, 9(5): 1290-1304.

[5]Lin N, Gao W, Li L, et al. vEpiNet: A multimodal interictal epileptiform discharge detection method based on video and electroencephalogram data[J]. Neural Networks, 2024, 175: 106319.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes EVER, a novel EEG–Video Emotion
Recognition framework that effectively integrates complementary information
from both modalities. Specifically, EVER employs a Brain anatomy-aware
Inter-modal Hierarchical Graph Convolution Network (BIH-GCN), which aggregates
EEG channel features into region-level representations guided by anatomical
priors. These region-level features are combined with global high-level EEG and
video backbone embeddings to form a unified representation for emotion classification.
Furthermore, the paper introduces a correlation-based distribution alignment loss
to reconcile modality-specific embeddings and reduce cross-modal discrepancies.
Extensive
experiments demonstrate that the proposed EVER achieves state-of-theart
performance by jointly modeling behavioral cues from video and physiological
responses from EEG, thereby enabling the recognition of emotional patterns
unattainable by either modality alone.

### Strengths
The paper introduces a novel EEG–Video Emotion Recognition (EVER) framework that explicitly
unifies video and EEG representations to overcome the limitations of existing multi-modal
architectures across heterogeneous modalities and diverse emotion recognition objectives. Specifically,
The paper proposes a Brain anatomy-aware Inter-modal Hierarchical Graph Convolutional Network
(BIH-GCN) with two key stages: (i) a local stage that aggregates EEG channels into anatomically
defined cortical regions to capture region-specific dynamics, and (ii) a global stage that integrates
these region-level representations with video and EEG embeddings through structured inter-modal
message passing. In parallel, the paper introduces a correlation-based distribution alignment that normalizes
covariance into correlations, reducing scale discrepancies between modalities while preserving
their complementary variations.

This paper has well realized multimodal sentiment recognition and demonstrates considerable innovation. Meanwhile, its experimental design is relatively comprehensive.

### Weaknesses
While the experiments in this paper are relatively comprehensive, there is a lack of discussion on certain comparative aspects. For instance, the AudioMamba algorithm, whose parameter count is less than one-thousandth of that of the algorithm proposed in this paper, still achieves performance superior to most other algorithms. The paper fails to provide an effective discussion on the balance between performance and computational complexity. It seems unreasonable to improve performance to a certain extent at the cost of computational efficiency.

Meanwhile, this paper integrates two existing methods to construct the baseline model, which leads to a decline in performance and makes it difficult to effectively demonstrate the advantages of the algorithm design.

### Questions
Please refer to the issues mentioned in the weaknesses section.

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
The paper introduces EVER, an EEG and video emotion recognition framework that aims to make physiological signals and visual cues work in a single structured model. The authors argue that emotion recognition with only video or only EEG gives a partial view and is brittle across subjects and recording conditions, so they build a multimodal pipeline that extracts high level features from video with AdaMAE and from EEG spectrograms with AudioMamba, then fuses them through a two stage Brain anatomy aware Inter modal Hierarchical GCN. The first stage aggregates channel features into anatomically grounded regions through masked graph propagation and attention pooling. The second stage places both global modality embeddings and the region nodes in one graph and performs structured message passing so that video and EEG can exchange information while remaining connected to brain regions. A correlation based distribution alignment loss is added to reduce statistical mismatch between the two modalities. The authors further provide what is essentially a benchmark for EEG and video pair emotion recognition on three public datasets, namely MDMER, Emognition and EAV, and compare twelve representative unimodal and multimodal baselines. EVER gives the best overall numbers on the three datasets and the ablations show that both the hierarchical graph and the correlation loss are necessary.

### Strengths
The work targets a setting that is clearly under serviced. Existing affective computing studies often focus on video alone or on EEG alone, while audio and video fusion is better studied. Positioning the model around paired EEG and video recordings is therefore meaningful for the ICLR audience that is interested in multimodal learning and in models that can work with physiological signals. The model design is careful. The first graph stage does not simply connect electrodes according to similarity. It enforces a mask derived from standard scalp regions and then performs attention pooling inside each region, which makes the subsequent fusion less sensitive to the specific channel layout of a dataset. This is a sound way to bring domain priors into a modern backbone. The second graph stage is the part that actually unifies the modalities. By treating the video embedding, the EEG embedding and the region representations as nodes in the same graph, the model creates an explicit route for inter modality reasoning instead of the usual concatenation that appears in audio and video models. This is consistent with the motivation stated in the introduction. The distribution alignment objective is a small but useful addition. It operates on correlations rather than raw covariance and therefore avoids scale problems. The ablations confirm that alignment alone does not solve the task but in combination with the anatomy aware graph it brings the model over both unimodal baselines and over naive fusion. The experimental section is richer than what is typical for a first paper on a new multimodal pairing. The authors collect three public datasets that have time aligned EEG and video, they impose subject independent splits, and they report accuracy, unweighted average recall and weighted F1, so the reader can interpret results under both class balanced and class imbalanced regimes. On all three datasets EVER improves over the strongest single modality model, which is the right standard to use in multimodal work. The paper is technically self contained. The construction of the two adjacency masks, the attention pooling, the fusion of the two global nodes into logits and the formulation of the alignment loss are all given in sufficient detail so that an experienced reader can implement the model. The writing is clear and connects the graph construction to actual neuroscience practice on five scalp regions.

### Weaknesses
Although the paper claims a unified framework, in practice the current system is anchored to a very specific choice of backbones. Video features come from AdaMAE and EEG features come from a spectrogram that is processed by AudioMamba. The paper does not test the framework with lighter or more conventional EEG encoders such as shallow CNNs or temporal transformers trained from scratch, nor does it show whether the hierarchical graph can compensate when one modality is considerably weaker. This limits the generality of the claim. The benchmark is valuable but the size of the paired data is still modest. Emognition has slightly more than four hundred pairs. MDMER has about two point three thousand pairs. EAV is larger but its labels are discrete and balanced. Generalization to truly large scale recordings, to unlabeled synchronized streams, or to field recordings with missing frames is not studied. The model relies on discrete brain regions derived from the 10 20 system and maps channels to these regions through a fixed mask. This is reasonable for the three datasets at hand, yet cross dataset variation in channel count is already visible. The paper partially alleviates this through attention pooling, yet it does not quantify how sensitive the model is to errors in the region assignment or to setups with very few channels such as six or fewer electrodes which are common in wearable EEG devices. The correlation based alignment is motivated as a way to reduce modality discrepancy, yet there is no comparison with other simple alignment objectives such as maximum mean discrepancy or contrastive matching of the global embeddings. The current ablation only shows with and without. A stronger analysis would include parameter free alternatives to confirm that choosing correlations is important. The interpretation part could go further. Figure A1 shows that different emotions activate different regions, which is interesting, but the main paper does not link these observations to classification outcomes or to failure cases. A reviewer would expect at least one analysis of wrong predictions where the video backbone is correct and the EEG branch is not, and the other way around, to justify the structured fusion. Finally, the paper is long and dense and some implementation choices appear late in the appendix. A more compact main text would help readers who want to re implement the method quickly.

### Questions
1-The model forms the second stage graph with two global nodes and K region nodes and then directly projects every node to the label space. Why was this direct projection chosen over first producing a unified feature and then applying a task head. 
2-It would be helpful to know whether the gains come from graph reasoning or from letting regions vote directly for the label. The correlation alignment is applied to batches of unimodal embeddings. What is the effect of the batch size on this loss and did the authors try instance wise alignment or moving average statistics to make it less sensitive to the composition of a batch. 
3-In the Emognition dataset there are only four EEG channels. In this case the first stage graph is almost trivial. Can the authors report how much of the improvement on Emognition comes from the alignment loss alone. 
4-In the benchmark comparison the existing multimodal models are audio and video systems that were re purposed because audio is structurally similar to EEG. Have the authors tried to re implement these models with the same EEG spectrogram encoder used in EVER in order to separate architectural gains from encoder gains. For subject independent evaluation it is important to consider domain shift across subjects. Did the authors observe subject specific clustering of the embeddings before and after alignment.
5-Figure 2 shows a nicer shared space, yet a quantitative measure such as class conditional Fréchet distance across subjects would make the claim stronger.

### Soundness
3

### Presentation
2

### Contribution
2
