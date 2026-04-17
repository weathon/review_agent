# Empowering Memory Assistance: An Episodic Memory-Based Framework for Personalized Recommendations

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Artificial agents operating in dynamic environments require the ability to recall and contextualize past experiences to inform future behavior. Drawing inspiration from human episodic memory, we propose a cognitively grounded recommendation framework that models time-evolving personal experiences using a dynamic, multimodal memory architecture. Our system encodes temporally structured actions, places, and interactions into a hierarchical temporal graph network (TGN), enabling agents to disambiguate overlapping behavior patterns and anticipate future actions based on long-term experience. Unlike traditional recommendation or forecasting models that rely on static, task-specific patterns, our approach supports continual memory updates without retraining, and generalizes across varied activity sequences. Evaluated on a structured dataset derived from three years of egocentric recordings, our model significantly outperforms state-of-the-art baselines (e.g., AntGPT, DyRep, Palm) on next-activity prediction and sequence alignment metrics. This work introduces a scalable, cognitively inspired memory architecture with broad applications in lifelong learning, assistive robotics, and human-AI collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an episodic memory-based framework for personalized activity recommendations using temporal graph networks (TGNs). The system organizes multimodal experiences (actions, places, times) into day-wise memory structures and uses dynamic memory updates to predict future activities. The authors claim state-of-the-art performance on a dataset derived from three years of egocentric video, significantly outperforming existing baselines like AntGPT and Palm.

### Strengths
1. Clear motivation and important application: The paper addresses a meaningful problem (memory assistance for individuals with memory impairments) with strong societal impact. The connection to human episodic memory provides good cognitive grounding.

2. Comprehensive multimodal integration: The framework effectively combines visual, audio, and textual modalities through VideoRecap and Whisper, which is appropriate for real-world assistive applications.

3. Thorough ablation studies: Tables 3-7 systematically evaluate different design choices (time encoding, backbone architecture, loss functions, sampling strategies), demonstrating that wavelet encoding and DyRep backbone work best for this task.

### Weaknesses
1. Dataset construction lacks transparency: The paper claims evaluation on "three years of egocentric recordings" but provides insufficient detail about how this timeline was constructed from Ego4D. Ego4D consists of short clips (typically 8 minutes) from 931 different people—how were these transformed into a continuous 3-year personal diary? This is fundamental to evaluating the validity of the approach but is glossed over in the main text.

2. Questionable experimental setup and fairness: The comparison with baselines (Palm, AntGPT, etc.) may not be meaningful if those models were trained on standard Ego4D benchmarks while your model operates on a differently organized dataset. Without results on standard Ego4D Long-Term Anticipation (LTA) benchmarks, it's impossible to assess whether the improvements come from your method or from the specific data organization. The 18% improvement over AntGPT (0.7220 vs 0.8853 ED) seems too good without proper controls.

3. Limited technical novelty: The core contribution appears to be organizing TGN memory by day rather than by event sequence. While this is sensible for lifelog data, it's more of an engineering choice than a fundamental algorithmic innovation. The paper doesn't clearly articulate what makes this approach fundamentally different from applying existing temporal graph methods to long-horizon forecasting.

### Questions
1. Data construction specifics: Can you clarify exactly how the "three years" of data were constructed? Since Ego4D has 931 different camera wearers with 1-10 hour sessions each, did you select videos from a single person over 3 years, or did you concatenate clips from multiple people to simulate a 3-year timeline? How do you ensure temporal consistency and prevent data leakage between train/test splits?

2. Baseline comparisons: Did you retrain Palm, AntGPT, and other baselines on your specific 3-year dataset organization, or are you comparing against their published results on standard Ego4D? Can you provide results for your method on the standard Ego4D LTA benchmark to enable fair comparison?

3. Method differentiation: How does your approach fundamentally differ from applying DyRep (which you use as backbone) with appropriate temporal features to the Ego4D LTA task? What specific advantages does "day-wise memory organization" provide beyond standard temporal graph modeling, and does this advantage hold on non-repeated, truly continuous egocentric data?

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
3

### Summary
This paper presents a cognitive grounded recommendation system to model the episodic memory. The method extends temporal graph network to encode multimodal signals (audio, visual, and linguistic cues) in the short- / long-horizon settings. On a reorganized Ego4D, authors show that the proposed framework achieves higher performance on the sequence level action prediction task, compared to AntGPT and Palm. The authors also conduct extensive ablation on the choice of time encoding, backbone, and sampling strategies.

### Strengths
1.	The need for episodic memory is well-motivated in the context of activity prediction. Embedding-based approach with no need for retraining is a promising direction.

2.	The authors consider multi-modal signals when constructing the memory, e.g., with both vision and audio signals processed.

3.	The authors present extensive ablation on the time encoding, backbone architecture, loss function, and (negative) sampling strategies.

### Weaknesses
1.	Although multiple modalities are considered, it is unclear how the interactions between the time, activity, and place are learned or used in the task context. For example, how would a simple baseline of node embedding (line 157) work: separately encode time, activity, and place, and use rank-fusion or weighted summarization to conduct the recommendation. On the other hand, the authors can include some quantitative or qualitative analysis on what patterns in the interactions are captured through the training, e.g., the model gets more sensitive with the co-occurrence of certain action-time pairs.

2.	Experiments are only done on Ego4D. AntGPT, the most relevant work mentioned in the paper, considers Epic-Kitchen-55 (Damen et al., 2020) and EGTEA Gaze+ (Li et al., 2018) in the year of 2024. More datasets should be considered to show the effectiveness and generalizability of the proposed framework.

3.	The framework is shown to work when trained and tested on an in-domain action anticipation task. However, there is limited insight into its usefulness in other areas with the simple framework. Besides the current ablation study, a potential pilot study on downstream tasks can further strengthen the claims on the application to real-world scenarios

### Questions
1. Citation on line 32 should be (Tulving, 2022)

2. Is the citation format correct? All the names are presented in-line.

3. Table captions should be above the tables

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a model capable of performing temporal reasoning with episodic memory for action anticipation.

### Strengths
This paper leverages temporal graph neural networks with multimodal data, which is promising for many tasks that require memorization and reasoning in practice.

### Weaknesses
- The writing should be largely improved: (a) It is unclear what exactly the authors are proposing. In Section 3, it seems that the proposed method is essentially a temporal graph network–based prediction model with episodic memory. However, this is not clearly described. The authors mention, “Our episodic memory–based … with a primary focus on action recommendation.” So, is the output a recommendation, a classification, or a regression model? Is it designed for action anticipation? Such concrete descriptions should be included. (b) In Section 3, the authors merely describe the components—node embedding, memory representation, message function, etc.—which are necessary, but the current version lacks motivation. Why did the authors design it this way? What differentiates it from prior works? What is the novelty in terms of architecture? 


- The figures are not informative. Figure 1 does not clearly illustrate the proposed method. While an introductory figure is optional, it should be highly informative. This one appears to be just a generic AI-generated image. The authors should be more careful about what figure they include in the Introduction. In addition, Figure 2 is also unclear. It should correspond to Section 3, as it is supposed to show how the proposed method works, but it is difficult to tell that they are related.

- The reported results do not include variance. In addition, I suggest that the authors provide some examples of how the proposed method works and its outputs on the considered datasets. Only presenting numerical results is not sufficient to claim that the model provides recommendations.

### Questions
See Weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method for predicting video-based action sequences on a dataset adapted from Ego4d. It is a semi-synthetic dataset because the videos are repeated and rebalanced to obtain continuous multi-year activity logs. The method involves training a graph neural network that stores memories (comprising features like event, time of day, time of year, activity, and place).

### Strengths
The use of episodic memory seems like a fundamental challenge to overcome, and the authors show progress on their particular dataset.

### Weaknesses
* the dataset is synthetic, and I am concerned that its statistics have been manipulated in such a way that it is particularly amenable to the method (e.g. repetition of exact sequences across days would make episodic memory particularly helpful)
* using episodic memory for sequence prediction is common and in fact commoditized -- chatgpt, gemini, anthropic all have memory based on rag. maybe you should narrow the focus to something more precise, e.g. multimodal generation instead?

### Questions
Could you evaluate on some kind of unmanipulated dataset?

### Soundness
2

### Presentation
3

### Contribution
2
