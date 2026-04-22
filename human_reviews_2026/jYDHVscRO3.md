# BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for  Digital  Behavioural Change

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Ambivalence and hesitancy (A/H), closely related constructs, are the primary reasons why individuals delay, avoid, or abandon health behaviour changes. They are subtle and conflicting emotions that sets a person in a state between positive and negative orientations, or between acceptance and refusal to do something. They manifest as a discord in affect between multiple modalities or within a modality, such as facial and vocal expressions, and body language. 
Although experts can be trained to recognize A/H as done for in-person interactions, integrating them into digital health interventions is costly and less effective. Automatic A/H recognition is therefore critical for the personalization and cost-effectiveness of digital behaviour change interventions. However, no datasets currently exist for the design of machine learning models to recognize A/H. 
This paper introduces the Behavioural Ambivalence/Hesitancy (BAH) dataset collected for multimodal recognition of A/H in videos. It contains 1,427 videos with a total duration of 10.60 hours, captured from 300 participants across Canada, answering predefined questions to elicit A/H.  
It is intended to mirror real-world digital behaviour change interventions delivered online. BAH is annotated by three experts to provide timestamps that indicate where A/H occurs, and frame- and video-level annotations with A/H cues. Video transcripts,  
cropped and aligned faces, and participant metadata are also provided. Since A and H manifest similarly in practice, we provide a binary annotation indicating the presence or absence of A/H.
Additionally, this paper includes benchmarking results using baseline models on BAH for frame- and video-level recognition, zero-shot prediction, and personalization with source-free domain adaptation methods. The limited performance highlights the need for adapted multimodal and spatio-temporal models for A/H recognition. Results obtained with specialized fusion methods are shown to assess the presence of conflicts between modalities, additionally temporal modelling for within-modality conflicts are essential for more discriminant A/H recognition.
The data, code, and pretrained weights are publicly available: https://github.com/LIVIAETS/bah-dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new multimodal dataset for recognizing ambivalence and hesitancy. It includes recordings with expert annotations covering facial, vocal, verbal, and bodily cues, offering a valuable resource for studying complex emotional states relevant to behavior change. The authors provide baseline experiments across different modalities and fusion strategies, showing that temporal context and multimodal learning improve performance but that the task remains challenging, paving the way for future research on nuanced emotion understanding.

### Strengths
1. The first public dataset focused on recognizing ambivalence and hesitancy, emotional states that play an important role in behavior-change research but have received little attention in machine learning.
2.The authors conducted comprehensive experiments comparing different setups: single-modality vs. multi-modality, with or without temporal context, and various fusion methods. The results show that adding temporal context improves performance, simple concatenation often works surprisingly well, and combining all three modalities does not always outperform simpler setups, indicating that multimodal fusion for ambivalence recognition remains a challenging problem.

### Weaknesses
1. The dataset provides only binary (0/1) labels for Ambivalence/Hesitancy, which may be too simplistic. Incorporating continuous annotations such as valence–arousal or PAD scales could enable a more fine-grained analysis of emotional states and cues.
2. Although the dataset includes facial, linguistic, audio, and body cues, these are annotated only at the video level. While this adds interpretability, it limits temporal precision. Adding timestamps or ideally frame-level cue annotations would make the dataset far more useful for detailed temporal modeling.
3. The paper describes the annotation process and co-annotation procedures but does not report quantitative inter-annotator agreement. Without this metric, it is difficult to assess the reliability and consistency of the labels.
4. The paper does not provide metadata statistics for the training, validation, and test splits, particularly regarding demographic balance. Reporting and ensuring demographic diversity across splits would support fairness analysis and enhance the dataset’s research value.

### Questions
1. What are the inter-annotator agreement (IAA) scores at both the video and frame levels for Ambivalence/Hesitancy labeling, as well as for cue tagging? Additionally, how do the models perform on high-certainty versus low-certainty segments?
2. How did the annotators distinguish ambivalence from mixed emotions or uncertainty without intent conflict? Please provide annotated examples and an error typology to clarify how these cases were handled.
3. Please include metadata statistics for the training, validation, and test splits, especially regarding demographic distribution, to assess data balance and fairness.
4. It would strengthen the paper to clearly explain the conceptual and methodological differences between Ambivalence/Hesitancy recognition and temporal emotion recognition, highlighting why A/H detection requires distinct modeling strategies.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents the Behavioural Ambivalence/Hesitancy (BAH) dataset, a multimodal video corpus for recognizing ambivalence/hesitancy (A/H) states. It includes 1,118 videos from 224 participants across Canada, annotated at video- and frame-levels with onset–offset segments and multimodal cues (face, audio, text). The dataset aims to model complex, sustained emotional states rather than discrete peaks. Baseline experiments perform binary A/H vs. non-A/H classification, showing the task’s difficulty and establishing a benchmark for future affective computing research.

### Strengths
1. This paper introduces a novel multimodal dataset that integrates visual, audio, and textual modalities, focusing on the recognition of ambivalence and hesitancy (A/H), which are complex, sustained emotional states rather than discrete basic emotions. It thereby proposes a new affect recognition task that broadens the traditional emotion recognition paradigm.
2. The dataset is of moderate scale, containing videos from 224 participants across 9 Canadian provinces, ensuring demographic and ethnic diversity and supporting fairer modelling of human affect. This makes it a valuable contribution to the affective computing community.
3. The baseline experiments are well designed and relatively comprehensive, covering frame-level and video-level recognition in both unimodal and multimodal settings. The paper also reports zero-shot prediction results and unsupervised domain adaptation models for personalization, which further enhance the dataset’s utility and research relevance.

### Weaknesses
1. While the dataset itself is valuable, the title and task definition are misleading. The term “Ambivalence/Hesitancy Recognition” suggests separate classification of ambivalence and hesitancy, yet all experiments address a binary detection task (A/H vs. non-A/H) without distinguishing the two.
2. The inclusion of “for Behavioural Change” in the title appears overstated, as the study does not involve any behavioural intervention, longitudinal tracking, or pre– post change analysis. The link between recognizing A/H emotions and actual behavioural change is discussed only conceptually, not empirically demonstrated.
3. The scientific relevance of detecting A/H remains insufficiently justified. While ambivalence and hesitancy are theoretically relevant to behaviour regulation, the paper does not clarify how A/H recognition could inform or improve behavioural interventions. Strengthening this conceptual bridge would significantly enhance the contribution.

### Questions
1. As mentioned in weaknesses. We suggest distinct recognition of ambivalence and
hesitancy and a connection to behavioural change, while the paper only conducts a binary A/H vs. non-A/H classification without behavioural intervention. A clearer alignment between the task scope and the title would improve conceptual precision.
2. The abstract could be more concise and focused on the dataset’s conceptual contribution and key findings rather than listing detailed statistics. Streamlining this section would make the paper’s main message clearer.
3. In the Introduction, the motivation for detecting A/H remains vague. The paper would benefit from a clearer explanation of why distinguishing A/H from non-A/H emotions is important and in which practical contexts this task has value. At present, the classification objective seems somewhat detached from real-world applications. The connection to behavioural change also remains conceptual. Clarifying how A/H recognition could be used in adaptive feedback or intervention systems would make this link more convincing.
4. Finally, the explanation of the multimodal fusion results (Table 5 / Table 15) is not entirely satisfactory. The paper attributes the drop in performance with tri-modal fusion (visual + audio + text) to “modality conflicts.” However, given that ambivalence and hesitancy are themselves conflictive emotional states, such cross- modal inconsistency might in fact be an informative signal rather than noise. It is unclear why three-modality fusion causes conflict while two-modality setups do not, especially since the text modality is derived from speech. Further clarification and analysis of this phenomenon would strengthen the interpretation of the results.

### Soundness
2

### Presentation
3

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
This paper introduces the first Behavioural Ambivalence//Hesitancy (BAH) dataset collected for subject-based multimodal recognition of A/H in videos. It contains videos from 224 participants captured across nine provinces in Canada, with different age, and ethnicity. BAH contains 1,118 videos for a total duration of 8.26 hours with 1.5 hours of A/H. The paper also provides preliminary benchmarking results using baseline models trained on BAH for frame- and video-level recognition with mono- and multi-modal setups. It also includes results on models for zero-shot prediction, and for personalization using unsupervised domain adaptation. The limited performance of baseline models highlights the challenges of recognizing A/H in real-world videos.

### Strengths
1. It is interesting to have a dataset for ambivalence and hesitancy (A/H)  recognition which  involve subtle and conflicting emotions that are manifested by a discord between multiple modalities, such as facial and vocal expressions, and body language. 

2. The authors conducted extensive experiments to demonstrate the potential of this dataset for frame- and video-based emotion recognition.

### Weaknesses
1. In the experimental results, the authors only consider CNN- and ViT-based models which are originally designed for image classification. It would be more interesting to consider video-based models to demonstrate the difficulty of the proposed benchmark.

2. While a dedicated dataset for ambivalence and hesitancy (A/H)  recognition is interesting, it is not clear what is the major difference between this task compared with other video recognition task such as activity recognition. In other words, what aspect makes this task challenging or difficult. 

3. The dataset size is relatively small with only a total duration of 8.26 hours which can possibly limit the usefulness of the proposed dataset.

### Questions
1. In table 5, it is shown that with Visual + Audio + Text, the results are worse than other cases, can the authors explain the possible reason? 

2. What are main challenges of ambivalence and hesitancy (A/H)  recognition compared with other types of emotion recognition such as anger or happy?

### Soundness
3

### Presentation
3

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
The paper proposes BAH, a dataset for recognizing ambivalence and hesitancy (A/H) in short webcam-style videos. The dataset consists of 1118 videos (~8 h total) from 224 participants across 9 Canadian provinces, each responding to 7 prompts. The dataset includes both frame-level and video-level A/H annotations. Baseline models are provided for visual, audio, and text modalities, as well as simple fusion and contextual vs non-contextual comparisons.

### Strengths
1. The first publicly available dataset specifically focused on ambivalence/hesitancy detection.
2. Clearly described data collection and annotation framework, incorporating both global- and frame-level labels

### Weaknesses
1. All subjects are from a single country (Canada), which may limit cultural and linguistic generalization of ambivalence expressions.
2. Contextual and multimodal (tri-modal) results are unexpectedly similar to non-contextual or single-modality baselines, suggesting limited exploitation of temporal or cross-modal dependencies.

### Questions
1. The contextual and multimodal (especially tri-modal) results are very close to those of single-modality models. How do you explain this? Why did tri-modal fusion underperform pairwise combinations? Are these differences statistically significant across runs? 
2. Given that ambivalence is highly context-dependent, what temporal window or duration do you consider sufficient for a valid A/H judgment? How sensitive are the annotations or models to this choice? 
3. How was ground truth established when annotators disagreed? Was it based on majority vote, adjudication by a lead annotator, or consensus discussion? 
4. The codebook explicitly includes body language cues, yet the modeling pipeline omits body or pose features. Why didn’t you include these, and do you plan to in future work? 
5. Can you provide inter-annotator reliability metrics (e.g. Cohen’s κ) for A/H labels? Also, how were temporal boundaries defined and aligned between annotators?

### Soundness
3

### Presentation
3

### Contribution
3
