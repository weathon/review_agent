# LogSTOP: Temporal Scores over Prediction Sequences for Matching and Retrieval

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Neural models such as YOLO and HuBERT can be used to detect local properties such as objects ("car") and emotions ("angry") in individual frames of videos and audio clips respectively. The likelihood of these detections is indicated by scores in [0, 1]. Lifting these scores to temporal properties over sequences can be useful for several downstream applications such as query matching (e.g., "does the speaker eventually sound happy in this audio clip?"), and ranked retrieval (e.g., "retrieve top 5 videos with a 10 second scene where a car is detected until a pedestrian is detected"). In this work, we formalize this problem of assigning Scores for TempOral Properties (STOPs) over sequences, given potentially noisy score predictors for local properties. We then propose a scoring function called LogSTOP that can efficiently compute these scores for temporal properties represented in Linear Temporal Logic. Empirically, LogSTOP with YOLO and HuBERT, outperforms Large Vision / Audio Language Models and other Temporal Logic-based baselines by at least 16% on query matching with temporal properties over objects-in-videos and emotions-in-speech respectively. Similarly, on ranked retrieval with temporal properties over objects and actions in videos, LogSTOP with Grounding DINO and SlowR50 reports at least a 19% and 16% increase in mean average precision and recall over zero-shot text-to-video retrieval baselines respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to lift local property detection scores in videos or audio to sequence-level temporal property scores, for example using YOLO to detect objects in video frames or HuBERT to detect emotions in audio clips, with confidence scores in [0,1]. The authors formalize this task as the “Scoring for TempOral Properties (STOPs)” problem, aiming to map these scores to temporal properties expressed in Linear Temporal Logic (LTL) despite potential noise in local predictions. To address this, the paper proposes a novel scoring function, LogSTOP, which efficiently computes sequence-level temporal property scores and handles temporal patterns such as “eventually” or “until.” Overall, this work presents a theoretically sound and computationally efficient approach that effectively lifts local prediction scores to temporal logic properties and demonstrates significant advantages across multiple downstream tasks.

### Strengths
1.One of the main contributions of this paper is the clear formalization of an important and challenging problem: STOPs (Scores for Temporal Properties), which involves lifting frame-level scores from potentially noisy local detectors to sequence-level scores for complex LTL properties.
2.The LogSTOP algorithm is designed to be efficient and scalable, making it well-suited for practical applications such as large-scale retrieval.
3.Although its theoretical assumptions (e.g., independence) have limitations, LogSTOP is highly effective in practice and is specifically designed to handle noisy real-world data.

### Weaknesses
1.The paper uses $P(\phi_1 \wedge \phi_2) \approx P(\phi_1) P(\phi_2) $(addition in log space) as the conjunction combination rule, which is equivalent to a strong independence assumption: the sub-events are independent and their probabilities are well-calibrated. However, in video or audio, the same attribute is often strongly correlated across adjacent time segments—e.g., an object persists across frames—and different attributes are also frequently correlated, such as a person appearing together with walking actions. The independence assumption therefore generally does not hold. Although the paper acknowledges that this assumption often fails, it represents a core logical inconsistency, as treating LogSTOP as a probabilistic model relies on a fundamentally incorrect assumption. Consequently, the effectiveness of LogSTOP may not stem from the paper’s central conceptual framework.
2.A question regarding the smoothing window is that it modifies the semantics of temporal logic (LTL). In LogSTOP, the “Next” operator means “φ holds in the next w window” rather than the standard LTL meaning of “φ holds at the next time point t+1.” This fundamentally changes the evaluated logical properties. An “eventually” property under the t+1 semantics is completely different from under the t+w semantics. Did the authors provide any explanation for this?
3.A question regarding the experimental section is the inconsistency between evaluation semantics and algorithm semantics. The ground-truth labels for QMTP and TP2VR are generated based on “standard LTL (t+1),” whereas LogSTOP actually computes “coarse-grained LTL (t+w)”. This means the model is evaluated under a different semantic framework, and the authors should provide an explanation for this.

### Questions
1.While the paper notes that the assumption often does not hold in video or audio sequences, the authors could further analyze its impact on the theoretical interpretation of LogSTOP. They could include experiments or analyses to show why the model still performs well empirically even when the independence assumption is violated, and discuss that the performance may stem more from empirical properties than from strict probabilistic modeling.
2.Explain how the use of a window w in LogSTOP changes the semantics of the “Next” operator compared to standard LTL (t+1), and discuss the potential bias or limitations this semantic modification introduces in evaluating temporal properties.
3.Clarify that the ground-truth labels for QMTP and TP2VR are generated based on standard LTL, whereas LogSTOP computes coarse-grained LTL , which may lead to an inconsistency in evaluation semantics. It is recommended to provide qualitative analysis or ablation experiments to assess the effect of this semantic difference on results, or to justify why comparisons remain fair in practice.

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
3

### Summary
This paper propose a scoring function  that can efficiently compute scores for temporal properties represented in Linear Temporal Logic. This module enables reasoning over sequences of predictions. To support the experiment, this paper also built two bench marks. Experiments show they achieve better results.

### Strengths
1, The introduction of the two benchmarks are important for this community
2, The idea on sequence scoring is easy and plug-and-play.
3, The writing is good in the motivation part.

### Weaknesses
1, The experiments presented is not in detail. Its difficult to read their specific numbers and for future citation and comparions. 
2, The quality and collection details of the two tasks are not discussed. This is very important in evaluating the significance of two bench marks.
3, The related work is very brief. Lack discussion on weakness of existing papers and fails to build relations between this work and previous works.
4, Line 52 is difficult to understand
5, Notations in Algorithm 1 is difficult to understand, some of them came with no context or definition.

I consider to improve the score if the significance of benchmarks are further verified.

### Questions
1, Would LogStop work beyond video and speech domains?
2，The logic seems to be manually defined and domain-specific. How generalizable is this approach to future or more complex tasks where temporal relations may not follow fixed rule templates
3, please see weakness related to the benchmarks.

### Soundness
3

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
2

### Summary
This paper introduces STOPs (Scores for TempOral Properties), aiming to address how to lift potentially noisy, low-level scores from local detectors into global scores for complex temporal properties. To this end, the authors propose a scoring function named LogSTOP, which can efficiently compute these scores. Experiments demonstrate that, on query matching and ranked retrieval tasks, LogSTOP significantly outperforms Large Vision/Audio-Language Models and other baselines.

### Strengths
1. The LogSTOP algorithm requires only linear time complexity.
2. The introduced QMTP and TP2VR benchmarks effectively evaluate query matching and ranked retrieval tasks.

### Weaknesses
1. The assumption that local properties represent independent events over time is rarely true for real-world sequences and properties. This raises concerns about potential failures in scenarios with complex temporal dependencies. The simplified theoretical basis may therefore not generalize well to timing-dependent cases.

2. LTL cannot express "counting" constraints (e.g., "there are always 2 cars") or handle numeric attributes.

### Questions
1. As a key hyperparameter, the smoothing window should be discussed in detail.
2. Is there any exploration of ways to avoid such an assumption?

### Soundness
2

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
This paper formalizes the STOP (Assigning Scores for Temporal Properties over sequence) problem and proposes LogSTOP as the efficient scoring function for temporal properties. This paper also proposed two new benchmarks: QMTP and TP2VR for the STOPs problem.

### Strengths
1. The notion of using a predictor to score the sequence with local property along with its temporal property is an interesting problem.
2. The proposed scoring function LogSTOP uses dynamic programming and is efficient to compute the score for a sequence.
3. The paper proposed two new benchmarks.

### Weaknesses
1. As mentioned in the paper, the relationships for local properties are not completely independent and there are some relations between them. In this case, using probability theory is not guaranteed to be optimal.
2. The threshold of LogSTOP is predefined to be log(0.5) and there is no further analysis. I would like to see the ablation study of that.
3. Although LogSTOP is claimed to be efficient, there is no experiment on the efficiency side, and it will be good if there are comparisons with other models.
4. The models used for comparison are limited (only 2 for each dataset and there are only 7B models’ results), which does not give a holistic evaluation of the effectiveness of the proposed method.

### Questions
1. In figure 4, why does the accuracy of LogSTOP decrease after using the log(0.5) threshold? Does it mean it include the  log(0.5) threshold mentioned before or it means a hard log(0.5) threshold?
2. Have authors tested the mixed modality case for the STOP problem?

### Soundness
3

### Presentation
4

### Contribution
3
