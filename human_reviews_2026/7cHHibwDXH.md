# CRSA: A Chinese Single-Domain Task-Oriented Dialogue Dataset with Contextual Rich Semantic Annotations

- Decision: Reject
- Scores: 2, 4, 6, 2, 6

## Abstract
Task-oriented dialogue (TOD) systems support users in achieving domain-specific goals via natural language interactions and critically depend on high-quality datasets. However, existing datasets often lack authenticity, fine-grained semantic annotations, and explicit process control, limiting effectiveness in complex business scenarios. To address these, we introduce CRSA, a Chinese TOD dataset that integrates diverse sources to construct semantically rich, structurally realistic dialogues, and adopts a multi-level annotation framework to model dialogue acts, user intents, and task flows more effectively. To evaluate the quality and application potential of CRSA, we conduct three sets of experiments spanning data quality, system training effectiveness, and task adaptability. Results demonstrate that CRSA provides strong support for process modeling, strategy learning, and response generation, establishing it as a robust and versatile resource for TOD research. The dataset is publicly available at https://anonymous.4open.science/r/CRSA-CBBB.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents CRSA, the first Chinese single-domain Task-Oriented Dialogue dataset explicitly designed for complex business process modeling and controllable response generation. Focusing on airline ticket booking, CRSA integrates real call-center logs, crowdsourced role-play, and GPT-4o augmented dialogs through a cleaning, normalization and stage-structuring pipeline. A three-layer hierarchical annotation scheme (Context / Dialogue / Slots) labels 26 k turns across 1.5 k dialogs with fine-grained dialogue acts, 63 system behavior triplets, six types of user anomalies, candidate options, and stage-aware slot updates. An automatic Baichuan2-7B multi-task annotator is trained to scale labeling while preserving quality. Extensive experiments demonstrate that models trained on CRSA outperform strong baselines on intent detection, action F1, task-completion and controllability metrics, and supply more diverse, realistic user simulators.

### Strengths
1. A new TOD dataset combines real logs + crowdsourced + LLM data.
2. Rich, layered annotation: stage, anomaly, option, behavior triplets & subjective-slot normalization enable process-aware training.

### Weaknesses
1. The data is single domain (airline) oriented, it is not clear if the pipeline can be generalized to other verticals settings.
2. The quality of the data is not fully illustrated, e.g. the diversity of the dialogue flows and pattern, as well as the word.
3. The data building aims to complex business process modeling. But it is not clear that how complex the dialogs in the data are?

### Questions
1. How does annotation quality degrade when the auto-annotator is applied to radically different domains?
2. What is the human-interaction cost of the proposed control tokens in live deployments?
3. Can the stage-transition prior be learned or relaxed for tasks without clear sequential phases?
4. How will CRSA perform with larger LLMs (e.g., > 70 B) or in reinforcement-learning dialogue pipelines?

### Soundness
2

### Presentation
3

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
This paper introduces a new Chinese task-oriented dialogue dataset for flight booking. 

It combines real, crowdsourced, and LLM-generated dialogues and includes multi-level annotations:
- Context: dialogue history, task progress
- Dialogue: system intent of current round, user response
- Slot: global slot value, status update

It highlights process control, exception handling, and system proactive guidance capabilities in dialogue content. This includes two key features: 
1) modeling and annotating six types of "user abnormal behavior" (e.g., unclear, vague, irrelevant, etc.); 
2) designing slot value normalization and subjective slot mapping strategies for unclear and subjective user expressions.

Experiments show that the proposed dataset provides support for process modeling, strategy learning, and response generation

### Strengths
- This is a well-written paper with a well-organized structure and clear expressions.

- The dataset construction process is reasonable, integrating real, crowdsourced, and LLM-generated data, with good authenticity and diversity.

- Data annotations are rich, covering context, conversational behavior, slot states, supporting process modeling and controllable generation.

- The experimental design is comprehensive, with validation ranging from data quality to system training and multi-task adaptability.

### Weaknesses
- **Limited novelty**: The main contribution is the provision of a task-oriented dialogue dataset. Although it introduces annotations for flow control, compared to historical datasets, it covers fewer linguistic phenomena (such as anaphora and ellipsis) and is restricted to a single domain—booking flights. In terms of innovation, it does not propose any new models or methodologies.

- **Unexplained low performance**: The low metrics on CRSA need more validation to prove they signify higher quality/complexity. Is it possible that there are inconsistencies in labeling, noise, or differences in task definitions? There is a lack of human evaluation or error analysis cases here.

- **Universality and transferability are questionable**: The paper only validates the approach in the domain of flight ticket booking, and whether the solution can be generalized and transferred to other domains remains unknown.

### Questions
1. What are the contributions of different data sources to the dataset, and how do they respectively impact the improvement of task performance?

2. The paper provides an empty project repository, making it impossible to assess the authenticity of the dataset's scenarios and the quality of annotations. Can it truly be open-sourced or offer more data examples?

3. Could you quantify the human cost of the initial “seed” dataset required to train the automatic annotation model (Section 3.3), for example, the number of person-hours or annotated dialogues needed? This is crucial for assessing the reusability of this annotation framework in other domains.

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
2

### Summary
The paper proposes a new dataset for task oriented dialogue with multiple semantic annotations.  The quality of the dataset is demonstrated by comparing with existing TOD datasets and the potential applications of the dataset are shown on four canonical TOD subtasks.

### Strengths
A new dataset with rich semantics is proposed for Chinese airline booking scenario.

Extensive experiments are conducted to verify the effectiveness or advantages of the dataset compared with the previous datasets.

Ablations of the rich semantic annotations and the automated compared with manual annotation are provided to show the effectiveness of the data annotation pipeline.

### Weaknesses
**Not very self-contained presentations.** The dataset proposed in the paper is about the air line booking scenario and one of the major contribution is about rich semantic annotation. Although the paper provides many statistics about the dataset compared with existing ones, no examples to show the typical booking dialogue and the claimed ''rich semantic annotation''. In the main paper, no example dialogue is demonstrated and only very limited examples are mentioned in the supplemental material.

**Lack of clarification on the comparisons with existing TOD dataset.** The datasets are collected for different tasks, for example, the proposed dataset for airline booking while crossWOZ for transportation, tourism, etc. **How to compare these dataset in a fair way? Does the questions for the comparison set for the airline booking? This seems unfair comparisons with existing datasets?**

### Questions
see the weakness above

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a Chinese single-domain task-oriented dialogue dataset, focusing on flight booking. The dataset is composed of three different sources, including real dialogues, crowd-sourced dialogues, and LLM-assisted generated dialogues.

### Strengths
* A single-domain TOD dataset is proposed.

### Weaknesses
* The definition of the question this work aimed to tackle should be clarified. At the end of Section 2, Related Work, the authors mentioned that previous works cannot capture dynamic scenarios, model anomalous user expression and support system-driven dialogue guidance. However, it is unclear how the proposed dataset captures dynamic scenarios, especially since it is a single-domain dataset without domain switching or coreference. In addition, why are the unclear, vague, or irrelevant user behaviour defined as "anomaly" behaviour and filtered out during data cleaning (see Section 3.2), when they are common in real-world interactions? Furthermore, what is the definition of "system-driven" dialogue? Does it refer to the proactive dialogue system or something else? 
* Following the previous point, what is the definition of "limited semantic diversity" and "insufficient process control"? Without a clear definition of these points, it is difficult to estimate the contribution of this work, especially since there are plenty of flight booking datasets which is more complex than this proposed dataset [1,2]. 
* In addition, the novelty of multisource data integration, structural standardisation, and hierarchical annotation framework should be elaborated further. Various TOD datasets also include dialogues from different sources, e.g. AirDialogue [2] and EmoWOZ [3], in order to enrich the variety of the dataset and to bridge the gap between human-to-human and human-to-machine conversations. Unified format across corpora is also proposed previously [4], and the hierarchical annotation framework, context-dialogue-slot, is a standard annotation schema, which is also presented in MultiWOZ, SGD, crossWoZ, etc. 
* The effectiveness of the experiments should be clarified as well. For example, the authors claimed that their proposed dataset exhibits greater semantic richness and diversity, as evidenced by the lower performance of mBART for DST compared to the other datasets in Table 3. However, it is unclear why lower performance on DST means the dataset is more diverse, which should be supported by other studies or more experiments. 


[1] Frames: a corpus for adding memory to goal-oriented dialogue systems (El Asri et al., SIGDIAL 2017)

[2] AirDialogue: An Environment for Goal-Oriented Dialogue Research (Wei et al., EMNLP 2018)

[3] EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems (Feng et al., LREC 2022)

[4] ConvLab-3: A Flexible Dialogue System Toolkit Based on a Unified Data Format (Zhu et al., EMNLP 2023)

### Questions
* The citation of MultiWOZ is incorrect (L081). It should be [1], instead of [2], which is MultiWOZ 2.4.
* In the related work, the authors mentioned that pipeline systems are facing joint optimisation issues, which is not always true [3]. 
* What is the meaning of "complex business scenarios" (L054)?
* Statistically significant test is missing, making it difficult to assess the results, especially Table 3, 4, 6, 7, and 9.


[1] MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling (Budzianowski et al., EMNLP 2018)

[2] MultiWOZ 2.4: A Multi-Domain Task-Oriented Dialogue Dataset with Essential Annotation Corrections to Improve State Tracking Evaluation (Ye et al., SIGDIAL 2022)

[3] A Generative Model for Joint Natural Language Understanding and Generation (Tseng et al., ACL 2020)

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CRSA, a Chinese TOD dataset in the airline booking domain that contains diverse sources (real, crowd-sourced, and LLM-generated) and rich annotations. 

The paper conducts extensive experiments to evaluate the quality and usefulness of CRSA. Results show that transforming other datasets into CRSA schema significantly improves the effectiveness of those datasets. Ablation studies also show that removing any of the components will harm the metric scores, validating the effectiveness of each component.

### Strengths
1. CRSA contains >1400 dialogs and >26000 turns, with more system-led control and diverse user behaviors, making it larger than existing baselines, and more diverse and representative of real-world scenarios.

2. CRSA also contains rich and process-aware annotation. It incorporates three-tier schema, user anomaly modeling, system behavior triplets, fuzzy expression handling, etc. Ablation studies show that removing any component hurts the scores, validating the effectiveness of each piece.

3. Re-annotation gains on other datasets demonstrate the strong utility of the annotation schema.

### Weaknesses
1. Focusing on only one domain (airline booking) constrains cross-domain generalization and limits application to broader scenarios. 

2. CRSA scores the lowest on DST/DA/BLUE/ROUGE-L after being converted to a common schema. While interpreted as "harder", it also signals that current models struggle to learn and generalize on CRSA effectively. It doesn't seem a fair comparison and does not reflect the advantage of CRSA.

### Questions
1. What are the portions of real, crowdsourced, and LLM-generated dialogues in the final corpus?

2. For LLM-generated dialogues, could you share more details on data quality judgment and filtering?

### Soundness
3

### Presentation
3

### Contribution
3
