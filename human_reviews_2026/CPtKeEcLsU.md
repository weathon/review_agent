# SpEmoC: Large-Scale Multimodal Dataset for Speaking Segment Emotion Insights

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Understanding human emotions in spoken conversations is a key challenge in affective computing, with applications in empathetic AI, human-computer interaction, and mental health monitoring. Existing datasets lack scale, tightly aligned modalities, and balance in emotion diversity thereby limiting robust multimodal models. To address this, we propose \textbf{SpEmoC}, a large-scale \textbf{Sp}eaking segment \textbf{Emo}tion dataset for \textbf{C}onversations. SpEmoC comprises 306,544 clips from 3,100 English-language videos, featuring synchronized visual, audio, and textual modalities annotated for seven emotions, and yields a refined set of 30,000 high-quality clips. It focuses on speaking segments under diverse conditions like low lighting and resolution, with a threshold-based filtering and human annotation ensuring a balanced dataset. SpEmoC is class-balanced, which enables fair learning across all emotions and leads to comparably balanced performance across all classes. We introduce a lightweight CLIP-based baseline model with a fusion network and a novel multimodal contrastive loss to enhance emotion alignment. We conduct a series of experiments demonstrating strong results, establishing SpEmoC as a reliable benchmark for advancing multimodal emotion recognition research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces SpEmoC, a large-scale multimodal dataset for emotion recognition in spoken conversational segments. The dataset consists of 306,544 raw clips extracted from 3,100 English-language movies and TV series. These clips were refined to a high-quality, class-balanced subset of 30,000 clips that are annotated for seven Ekman-based emotion categories. The authors propose an automated annotation pipeline that leverages pretrained DistilRoBERTa (for text) and Wav2Vec 2.0 (for audio) models. Human validation is also used to ensure the accuracy of the annotations. Additionally, the authors introduce a lightweight baseline model based on CLIP-HuBERT-MLP and a novel Extended Reweighted Multimodal Contrastive (ERMC) loss to align cross-modal emotion embeddings. The model is evaluated on both SpEmoC and two other datasets: MELD and CAER.

### Strengths
1. SpEmoC significantly outperforms previous benchmarks (e.g., MELD, CAER) in terms of scale and emotional balance. The use of synchronized video, audio, and text from various cinematic sources in real-world settings (e.g., with low lighting and variable resolution) improves the ecological validity of multimodal emotion recognition.
2. The authors use a two-step annotation strategy: first, they use pseudo-labels generated from pretrained unimodal emotion classifiers (DistilRoBERTa and Wav2Vec 2.0) to create a large dataset of labeled clips. They then use the KL-divergence regularization to ensure consistency between different modalities. Second, they have 20 experts validate 50,000 clips, achieving a Fleiss' Kappa score of 0.62, which balances scalability and reliability.
3. The proposed Extended Reweighted Multimodal Contrastive Loss incorporates sentiment-based reweighting using KL divergence between unimodal emotion distributions. This aligns emotionally consistent embeddings across modalities, which significantly improves performance compared to using cross-entropy alone.
4. The lightweight model was trained and evaluated on not only SpEmoC, but also MELD and CAER. This allows for a direct comparison of the quality of the datasets through consistent modeling, strengthening the claim that the balanced design of SpEmoC yields more equitable performance for minority emotion classes, such as Fear and Disgust.

### Weaknesses
1. While the dataset includes 3,100 videos, 85% originate from scripted films/TV shows, limiting generalizability to spontaneous, real-world affect. Moreover, the paper provides no information on speaker-level demographics (e.g., gender, age), only coarse video-level ethnicity estimates. This omission hinders fairness and bias analysis, that critical in affective computing.
2. The pipeline uses YOLOv8 for face/human detection, but the paper does not explain how the target speaker is chosen when multiple individuals are present. Without explicit speaker diarization or face-voice alignment, the emotion label (based on text/audio) may not match the visual subject, especially in group conversations.
3. The proposed model combines frozen CLIP-ViT (with AIM adapters), HuBERT, and a simple MLP fusion head - a standard architecture in multimodal learning. Although efficient (~8.68M trainable parameters), it does not offer any architectural innovation, and serves primarily as a validation tool for the dataset, rather than a significant contribution to the field of modeling.
4. The authors report F1 scores for MELD and CAER, but they do not compare them with existing state-of-the-art models. Therefore, it is unclear whether the performance differences are due to the quality of the dataset or the capability of the model. Consequently, the claim of "strong results" is not supported by the current literature.
5. The ERMC loss is only compared to vanilla cross-entropy and not to other contrastive, focal, or rebalancing losses. Without these comparisons, the additional value of ERMC is uncertain.
6. Although the authors use a movie-level split, emotional expressions in acted content can be stereotypical or dependent on the genre (e.g., fear in horror). If certain genres dominate the splits, the model may learn correlations between genres and emotions rather than genuine affective cues, leading to inflated generalization metrics.

### Questions
1. In clips with multiple speakers, how is the subject of visual analysis aligned with the audio/text transcript? Has any form of facial recognition or voice identification been used?
2. Can the authors provide speaker-level metadata, such as gender, age, and ethnicity, for the 30,000 refined clips? If not, how do they ensure that their model is not biased against certain groups in terms of representation?
3. Why was the proposed model not compared to recent state-of-the-art (SoTA) systems on MELD and CAER datasets? The authors could have included such comparisons to determine whether performance improvements are due to improved data quality or the model design.
4. Have the authors explored the use of alternative loss functions to address class imbalance and modality alignment in their experiments? If so, how does the ERMC model compare to these other approaches?
5. Given that 85% of the clips used for training were acted, how confident can the authors be that models trained on SpEmoC will be able to generalize to real-life, spontaneous conversations (e.g., interviews, customer service calls)?

### Soundness
2

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
5

### Summary
The paper introduces SpEmoC: a large-scale, multimodal corpus for the recognition of emotions in spoken, conversational segments. It is derived from 3,100 English-language films and television series. The dataset comprises 306,544 raw clips that have been refined into 30,000 high-quality samples that are balanced across seven Ekman emotions. The authors propose an automated annotation pipeline that uses pre-trained DistilRoBERTa (for text) and Wav2Vec 2.0 (for audio) models. These are fused via a KL-divergence-regularized logit fusion strategy and then validated by humans. They also present a lightweight CLIP-based baseline model with an Extended Re-weighted Multimodal Contrastive (ERMC) loss function for aligning cross-modal emotion embeddings.

### Strengths
1) The SpEmoC is the largest publicly available multimodal emotion corpus featuring class balancing, which enables fair evaluation across all seven emotions.
2) Multi-stage refinement (thresholding and human validation) and a movie-level split ensure the creation of high-quality, generalizable benchmarks.

### Weaknesses
1) Around 85% of the data originates from feature films and TV series, in which emotions are typically acted out and often exaggerated.
2) 60% of participants are from the Western/white ethnic group. This calls into question whether models can be generalized to global populations, and it may exacerbate inequalities in affective systems.
3) Although it is critical for multimodal systems, it is unclear how cultural norms influence the expression of emotions in data.
4) Pseudo-labels are generated by DistilRoBERTa and Wav2Vec 2.0, which are trained using social media and actor speech corpora. However, these models can carry their own biases (e.g. associating anger with aggressive vocabulary), which can distort the 'true' labels.
5) The architecture is a standard combination of CLIP-ViT, HuBERT and MLP, with no new fundamental components.
6) In real-life scenarios, emotions are usually complex, but the corpus assumes only one dominant category, which makes the task easier but reduces its practical value.

### Questions
1) How did you verify that no actors were duplicated between splits?
2) Why wasn't ERMC compared with modern methods?
3) What measures have been taken to address the cultural bias of casting 60% white actors?
4) How were cases handled where the actors spoke without emotion, even though the scene was emotional?
5) How did you deal with mixed emotions?
6) Has an analysis of model errors been conducted by demographic group (gender and ethnicity)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SpEmoC, a relatively large-scale multimodal dataset designed for emotion recognition in conversational speech segments. Curated from 3,100 English-language films and television series, the dataset comprises 306,544 raw clips as well as 30,000 refined clips. Each of these refined clips integrates synchronized visual, audio, and textual modalities, and all clips are annotated with labels corresponding to 7 basic emotion categories. It is anticipated that this dataset will provide a valuable contribution to advancing research in the relevant field.

### Strengths
The paper makes valuable contributions to multimodal emotion recognition (MER) research:
1. SpEmoC directly addresses the problem of emotion imbalance in existing MER datasets, it achieves a relatively balanced distribution across 7 emotions, enabling robust recognition of minority classes .
2. The scale and source diversity (thousands of films/TV series) provide rich acoustic, visual, and linguistic variability, potentially improving generalization beyond lab-recorded datasets.

### Weaknesses
1. The empirical validation of the dataset’s effectiveness is too limited. In the main paper, only a single table (Table 4) reports results, and although some ablations are deferred to the appendix, there is no exploration of how SpEmoC benefits other downstream tasks. For a dataset contribution, readers expect broader evidence of utility, such as transfer to related tasks, pretraining gains for unimodal and multimodal backbones, or improvements in low-resource settings.

2. If I understand correctly, comparisons across datasets are conducted on each dataset’s own train/test split, without cross-dataset evaluation. This setup prevents a clear assessment of generalization. For a large-scale dataset claim, the community typically prioritizes evidence of out-of-domain robustness over in-domain performance on the new dataset. High in-domain scores on same-source data may simply indicate that the collection is not particularly challenging.

3. Results reported for the other two datasets in your tables are substantially below those in the literature. Stronger baselines should be selected and reproduced under comparable settings. 

4. The dataset is potentially useful for the community; however, the novelty appears limited considering the data processing pipeline is quite standard.

### Questions
1. Beyond Table 4 and the appendix ablations, what additional evidence can you provide to demonstrate SpEmoC’s utility? Have you evaluated pretraining/fine-tuning on SpEmoC and transferring to related downstream tasks?
2. Do models pretrained on SpEmoC yield consistent gains in low-resource regimes on external benchmarks? Have you conducted cross-dataset evaluations  to assess out-of-domain robustness?
3. How sensitive are results to clip segmentation, alignment errors (In the data processing pipeline)?

### Soundness
2

### Presentation
3

### Contribution
2
