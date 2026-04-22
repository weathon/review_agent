# Brain Signal Rendering: Unifying EEG Video Representations for Subject-level Few-shot Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 0, 6, 4, 4

## Abstract
EEG modeling faces two core challenges: nonlinear, non-stationary dynamics and severe channel mismatch across datasets. We introduce Brain Signal Rendering (BSR), a new paradigm that reframes EEG representation learning as a rendering problem. BSR transforms EEG spectrograms into spatialized dynamic 'EEG videos', making representations invariant to electrode layouts and sampling protocols while preserving neural topology. Building on this, we propose EEG Consolidation — a unified multi-task training paradigm that integrates heterogeneous EEG-video data to adapt models to EEG-specific dynamics, improve data efficiency, reduce overfitting, and boost cross-task generalization. Crucially, BSR with EEG Consolidation enables subject-level few-shot learning, where each subject is treated as a distinct task requiring adaptation from minimal data. We validate this setting as a realistic benchmark and demonstrate substantial performance gains, establishing a scalable and interpretable framework toward foundation models for brain signals.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces Brain Signal Rendering (BSR), a framework designed to address key challenges in EEG signal representation learning, specifically the non-linearity and non-stationarity of EEG signals, as well as channel mismatches across datasets. By transforming EEG spectrograms into spatialized dynamic "EEG videos," BSR ensures that the resulting representations are invariant to electrode layouts and sampling protocols while preserving the neural topology. This paper propose EEG Consolidation, a multi-task training paradigm that integrates heterogeneous EEG-video data, improving data efficiency, reducing overfitting, and boosting generalization across tasks. This framework enables subject-level few-shot learning, where each subject is treated as a unique task requiring adaptation from minimal data. This work highlights the potential for using video-based models, such as VideoMAE, as backbones for EEG tasks.

### Strengths
The strengths of this work lie in its clear writing, which makes it easy to understand and enhances readability. The idea of transforming EEG signals into video representations is promising; however, the actual method has several shortcomings and lacks novelty. For further details, please refer to the "Weakness" section.

### Weaknesses
This work lacks novelty, and the authors show a limited understanding of the current state of research and standard methods in the EEG field. The overall approach is rather simplistic, and both the main text and the appendix contain very limited experimental work, which is insufficient and fails to meet the standards expected at ICLR. This is evident in the following aspects:

1. The core method of this work is the proposal to transform EEG signals into video representations, specifically by first extracting frequency-domain features from the EEG, then arranging the electrode positions into a 2D format, and integrating the frequency bands and time to create a 4D data structure. However, this method has already been extensively explored and repeatedly applied in the EEG field, with countless studies adopting similar approaches [1-5]. It is clear that the authors lack awareness and understanding of the current research landscape.

2. Another core method mentioned by the authors is subject-level few-shot learning. However, I am not sure if I misunderstood, but based on the author's description, this seems equivalent to the commonly used experimental setup in EEG, known as "subject-dependent," where the test and validation sets come from the same subject. This is a standard paradigm. Moreover, as EEG modeling has evolved, there has been a strong emphasis on generalization, with current work focusing on improving performance across subjects and tasks. In fact, research is increasingly seeking better generalization, and work that targets a subject-dependent scenario seems like a step backward in history. The authors, however, present this as a novel approach, which, in my view, is somewhat amusing.

3. The motivation of this work is not clearly articulated. Although EEG signals can be transformed into a format resembling that of a video, there are significant semantic differences between EEG data and natural images. The rationale for using the pre-trained VideoMAE as the core component of the model is not well justified, as these two types of data differ substantially in their underlying structure and meaning. This lack of alignment raises questions about the appropriateness and effectiveness of applying a model designed for natural video data to EEG signals.

4. The experiments in this work are extremely limited, overly simplistic, and basic. Specific issues and suggestions will be addressed in the "Question" section.


[1] Bashivan, Pouya, et al. "Learning representations from EEG with deep recurrent-convolutional neural networks." arXiv preprint arXiv:1511.06448 (2015).

[2] Shen, Fangyao, et al. "EEG-based emotion recognition using 4D convolutional recurrent neural network." Cognitive Neurodynamics 14.6 (2020): 815-828.

[3] Xiao, Guowen, et al. "4D attention-based neural network for EEG emotion recognition." Cognitive Neurodynamics 16.4 (2022): 805-818.

[4] Liu, Jiyao, et al. "Positional-spectral-temporal attention in 3D convolutional neural networks for EEG emotion recognition." 2021 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2021.

[5] Jia, Ziyu, et al. "Sst-emotionnet: Spatial-spectral-temporal based attention 3d dense network for eeg emotion recognition." Proceedings of the 28th ACM international conference on multimedia. 2020.

### Questions
1. If I am not misunderstanding, this work may contain a significant flaw. The model trains the renderer and reconstructor during the pre-training phase using a reconstruction approach. However, the issue is that the input data is not masked, which would allow the model to simply learn an identity mapping. This is problematic in the context of reconstruction-based pre-training paradigms, as it undermines the purpose of such pre-training.

2. The representation gap between video and EEG, which are transformed into images, needs to be considered. It cannot simply be assumed that because the format is similar, the same approach can be applied. The rationale for why this method is suitable for EEG data requires careful analysis and justification. The authors should provide a clear explanation of why pretrained video model works for EEG, considering the significant differences in the nature of the data.

3. Why VideoMAE? Are there more recent methods that could be considered? In the video and visual domains, the ability for few-shot learning does not seem to be unique to VideoMAE. Furthermore, the authors emphasize that VideoMAE can be replaced by other models, but they fail to provide relevant experimental evidence to support this claim, making it difficult to be convincing.

4. Is there a specific reference for the user-specified channel spatialization map matrix defined by the authors? What is the rationale behind determining the arrangement of electrode positions? Moreover, since EEG electrodes are actually placed in a 3D configuration on the scalp, why map them into a 2D arrangement? A direct 3D mapping would likely be more realistic and align better with the actual electrode placement.

5. In multi-task learning, each task should have a task-specific head to complete the final classification (L255). However, the authors mention treating each subject as a separate task. How many heads are there in total? Does each subject have its own head, or are the heads for subjects within the same dataset shared, given that they have the same dimensionality? This needs to be clarified. Additionally, the performance shown in Table 5 of the appendix does not seem very promising. This suggests that the multi-task joint training in this work does not lead to collaborative improvement across tasks. Therefore, the proposed approach does not appear to deliver practical benefits, and as such, it may not be suitable as a significant contribution.

6. If the performance is evaluated for each subject and the final results are presented as an average, it would be helpful to see the standard deviation as well. This would provide a clearer understanding of the variability and reliability of the model's performance across different subjects.

7. It is recommended to include an ablation study comparing the experimental results between two settings: one that inherits the pre-trained VideoMAE parameters and one that does not. This would help demonstrate the benefit of leveraging a pre-trained video model for EEG representation learning, providing evidence that pre-training with video data contributes positively to EEG modeling.

8. The authors did not use the official pre-trained weights for the replication of LaBraM, nor did they compare the  pre-trained BIOT model. This seems unreasonable. In fact, using pre-trained parameters does not lead to data leakage, as the model does not have access to the labels of the data. It is recommended that the authors redo the comparisons, including using the pre-trained models for LaBraM and BIOT to ensure a fair and thorough evaluation.

9. The authors have not provided the structure, description, or parameter scale of VideoMAE. Including this information would help readers better assess the effectiveness of the model and understand how the underlying architecture contributes to the results. 

10. It is recommended to include an ablation study or parameter analysis for the Renderer and Reconstructor structures. This would help to understand the impact of different design choices or hyperparameters on the performance of the model, providing insights into which aspects of the architecture contribute most to the overall results.

11. The BSR Render-Reconstruct pipeline is essentially a reconstruction of 4D inputs, which doesn't seem to align well with video representations. This approach may not be entirely reasonable. Why not consider integrating this reconstruction process directly with VideoMAE and treat the entire pipeline as part of the pre-training? Fine-tuning with VideoMAE for downstream tasks could introduce a gap between pre-training and downstream tasks, as well as increase computational overhead. Of course, this is just a rough suggestion, and I am curious to understand why this approach wasn't considered.

12. In the current setup of this work, the performance on various datasets appears to be quite underwhelming, with results that seem to fall below the standard of existing work in the field. It would be beneficial for the authors to conduct further research and optimization to improve the model's performance.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper reframes EEG representation learning as a rendering problem via Brain Signal Rendering (BSR). It per-channel spectrograms are spatialized according to electrode geometry and rendered into dynamic “EEG videos,” yielding representations that are invariant to electrode layouts and sampling protocols while preserving neuro-spatial structure. Building on this, EEG Consolidation performs multi-task fine-tuning of a shared video encoder across heterogeneous EEG-video datasets, unifying learning across datasets, tasks, and montages and enabling robust subject-level few-shot adaptation.

### Strengths
a.    Physics-informed perspective: EEG is recast from a channel vector to a physical projection. Under this view, channel mismatch is a change of viewpoint rather than noise; the objective shifts from task-specific embeddings to inverting the projection to recover underlying spatiotemporal neural dynamics.

b.   BSR for cross-modal transfer: Treats EEG spectrograms as structured projections and renders them into dynamic images (“EEG videos”), allowing direct reuse of video foundation models (e.g., VideoMAE) whose spatiotemporal inductive biases align with neural dynamics.

c.    Consolidated multi-task training: Unifies datasets with different montages/protocols, improving invariance to electrode layout, data efficiency, robustness, and downstream performance in subject-level few-shot settings.

### Weaknesses
a.    Many EEG datasets follow the international 10–20 electrode placement system, but the datasets used for experiments in the paper have a large number of channels. For example, CHB-MIT (23 channels), Sleep-EDF (1 channel), and ISRUC (6 channels). I recommend systematically evaluating the model on low-channel datasets and conducting robustness tests.

b.   This setting is not introduced "subject-level few-shot" for the first time in this work. Prior studies have already evaluated cross-subject few-shot or subject-independent adaptation. For example, *Calibration-free meta-learning based approach for subject-independent EEG emotion recognition* demonstrates few-shot classification on unseen subjects, and *BrainWave: A Brain Signal Foundation Model for Clinical Applications* likewise adopts cross-subject few-shot in its downstream evaluations. Consequently, the incremental contribution here is limited.

c.    Please provide the basic preprocessing pipeline used in pretraining and downstream tasks (e.g., filtering, artifact removal, resampling, normalization), to ensure fair and reproducible comparisons.

d.   The BSR renderer maps continuous spectral information directly into discrete RGB channels, which is relatively black-box and lacks interpretability. It need more experience to clarify and empirically demonstrate, whether this information compression sufficiently preserves critical band features and does not harm downstream performance.

e.    EEG Consolidation is positioned as a multi task fine-tuning strategy for integrating heterogeneous EEG video data, but the actual training data used for fine-tuning is currently limited to the TUAB and TUEV datasets, with limited coverage and difficulty reflecting the benefits of multi task integration in a wider range of scenarios. In addition, the comparison presented in the paper mainly consists of two settings: "using only video features" and "EEG+video", without providing a comparison that can prove the benefits brought by multitasking.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Brain Signal Rendering (BSR), a framework that reformulates EEG representation learning as a rendering problem. The key idea is to spatialize EEG spectrograms based on electrode locations, converting them into “EEG videos” compatible with large-scale video foundation models such as VideoMAE. The authors propose EEG Consolidation, a multi-task training strategy that integrates multiple EEG-video datasets. The paper defines a Subject-Level Few-Shot Learning benchmark to evaluate model adaptability to new subjects with minimal fine-tuning data. Empirical results show that BSR-VideoMAE achieves improvements over prior EEG models on emotion recognition (SEED, SEED-VII) and motor imagery tasks (SHU-MI, BCICIV-2a).

### Strengths
1. The subject-level few-shot tasks are a valuable contribution, aligning better with realistic EEG deployment scenarios.
2. The methodology is described in detail.
3. Improvements across several EEG datasets show that the framework works in practice.
4. The framework demonstrates how pre-trained VideoMAE weights can be adapted to low-data EEG domains.

### Weaknesses
My overall impression is that while the idea of EEG-to-video rendering is interesting, several claims (invariance to electrode layouts, reduced overfitting, scalability, interpretability, and robustness) are not sufficiently supported by controlled experiments. Moreover, the experimental setup lacks comparisons that would isolate the contribution of BSR from the VideoMAE backbone and pretraining regime. While the technical contribution is somewhat incremental in practice, mainly a data transformation enabling the reuse of pretrained video models, the proposed BSR paradigm and subject-level few-shot setting are interesting. However, without more in-depth experiments and more baselines considered, it remains unclear whether BSR is a fundamentally better representation or a temporary workaround for EEG data scarcity that scale less well than other methods, such as LaBraM.

1. The paper does not convincingly separate the benefit of rendering from the effect of video pretraining. Without an untrained-video-model control or direct comparison to 4D spectrum baselines, the intrinsic merit of EEG-video transformation remains uncertain.
2. Rigid processing: The fixed 224×224×3 spatial format and 16-frame sequence length may discard signal-relevant information.
3. Assertions about invariance to electrode layouts, reduced overfitting, improved robustness, and interpretability lack quantitative backing.
4. Limited scalability analysis: The experiments pretrain only on TUAB+TUEV. No scaling results across additional EEG datasets (e.g., SEED, SEED-VII) are provided to demonstrate extensibility.
5. Standard deviations and per-subject performance are omitted, limiting the interpretability of performance differences.

### Questions
1. How does BSR perform when trained without any video pretraining (i.e., trained from scratch on EEG videos)?
2. Can the authors compare BSR’s EEG-video representation to other related baselines such as [1]?
3. To substantiate electrode layout invariance, could you report experiments where, for example, channels subsets (especially the 19 used in pretraining) are removed during evaluation? Since the model is  pretrained only on 19 channels, it is not clear how much of its performance is dependent on those channels.
4. How do the models (especially BSR-VideoMAE and LaBraM) performance evolve as EEG pretraining data scale (e.g., TUAB+TUEV+SEED+SEED-VII)?
5. Self-supervised training is well established for video and is generally preferred over supervised training for pretraining. Would self-supervised pretraining on EEG video outperform the current supervised pretraining strategy?

[1] https://arxiv.org/abs/1511.06448

### Soundness
2

### Presentation
4

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
This work addresses a fundamental challenge in large-scale EEG modeling: the severe channel mismatch and lack of standardization across datasets.

### Strengths
It directly addresses the channel mismatch and dataset heterogeneity, which are arguably among the biggest blockers to creating large, general-purpose EEG foundation models. 

The BSR concept, which converts EEG signals into spatialized EEG videos, is a creative and powerful paradigm. It standardizes the data format while preserving the brain's spatial topology, regardless of the original electrode layout.

### Weaknesses
The rendering process, which transforms sparse data collected from EEG sensors into a comprehensive video format, can inadvertently introduce various artifacts. This transformation may obscure crucial local details essential for accurate interpretation, or worse, lead to incorrect assumptions about the underlying brain activity. 

Processing videos is inherently more computationally demanding than analyzing the original one-dimensional time-series data derived from just a handful of channels. This increased complexity involves not only higher processing power but also greater resource allocation in terms of time and energy.

The overall success of this approach is heavily dependent on the sophistication of the video-processing model employed, such as VideoMAE, in its ability to accurately interpret the rendered EEG videos. Any limitations inherent in the video model will directly translate to constraints in the EEG model, leading to potential inaccuracies in understanding brain function based on the processed video data.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
