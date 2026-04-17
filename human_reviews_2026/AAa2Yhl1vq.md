# LLMs Reading the Rhythms of Daily Life: Aligned Understanding for Behavior Prediction and Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6

## Abstract
Human daily behavior unfolds as complex sequences shaped by intentions, preferences, and context. Effectively modeling these behaviors is crucial for intelligent systems such as personal assistants and recommendation engines. While recent advances in deep learning and behavior pre-training have improved behavior prediction, key challenges remain—particularly in handling long-tail behaviors, enhancing interpretability, and supporting multiple tasks within a unified framework. Large language models (LLMs) offer a promising direction due to their semantic richness, strong interpretability, and generative capabilities. However, the structural and modal differences between behavioral data and natural language limit the direct applicability of LLMs. To address this gap, we propose Behavior Understanding Alignment (BUA), a novel framework that integrates LLMs into human behavior modeling through a structured curriculum learning process. BUA employs sequence embeddings from pretrained behavior models as alignment anchors and guides the LLM through a three-stage curriculum, while a multi-round dialogue setting introduces prediction and generation capabilities. Experiments on two real-world datasets demonstrate that BUA significantly outperforms existing methods in both tasks, highlighting its effectiveness and flexibility in applying LLMs to complex human behavior modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose BUA, a training framework which bridges the modality gap between behavioral sequences and language through a three stages curriculum, sequence level understanding, user level feature modeling, and self-reflective refinement, anchored by pretrained behavior sequence embeddings. The multi-round dialogue mechanism teaches the LLM to reason through and describe behavioral patterns before predicting or generating future actions, enhancing interpretability and performance. Experiments on two real world datasets show that BUA outperforms traditional and LLM based baselines in both behavior prediction and generation, particularly for long-tail behaviors.

### Strengths
1. The idea of aligning behavioral embeddings with language representations via curriculum learning tackles a timely challenge in cross-modal understanding. And the structured progression from sequence understanding to reflective refinement provides a coherent training strategy and contributes to interpretability.
2. The experiments show solid gains over both traditional sequence models and LLM-based baselines, especially on long-tail behavioral patterns where previous methods struggle. So, it could be benefits to personalized assistants, recommendation systems, and behavioral analytics applications.

### Weaknesses
While BUA demonstrates improved performance on long-tail behaviors, the paper lacks sufficient qualitative analysis or visualization to illustrate how the model captures these rare behavioral patterns. For example, since the authors use qwen as the base model, it is possible that the observed improvements partially stem from qwen’s inherent advantages over other LLMs (e.g., DeepSeek). A comparison across different base models or an analysis isolating BUA’s contribution would help clarify this point.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this work, the Behavioral Understanding Alignment (BUA) framework is introduced in an attempt to understand human daily behavioral sequences. It combines a three-stage curriculum learning pipeline with dialogue mechanisms. Experiments show that BUA performs well in prediction and generation tasks.

### Strengths
- Explanations are well-organized and straightforward.
- I like the different types of experiments the authors tried out. Especially, the cross-model enhancement experiment is informative.

### Weaknesses
### 1. Task Clarification
I am not fully convinced about the difference between the two tasks: (1) behavior prediction and (2) behavior generation. To me, generation seems to be a sequential prediction problem that inherently depends on the prediction task. Could the authors clarify what conceptual or methodological differences I might be missing here?
### 2. Fairness in Comparison
The comparison does not appear fully fair. Most baselines are either pretrained for general purposes or trained on unrelated tasks, while BUA is explicitly trained on the downstream task. Wouldn’t it be more appropriate to compare BUA’s results with fine-tuned versions of the baselines under equivalent conditions?
### 3. Curriculum Design and Theoretical Grounding
The three-stage curriculum (sequence-level, user-level, self-reflective) lacks theoretical grounding and is mostly procedural. Please provide a convergence curve proving that the staged training outperforms simple joint training beyond anecdotal evidence.
### 4. Evaluation Metrics and Interpretability Claims
Metrics such as weighted precision/recall, BLEU, TVD, and JSD are low-level and insufficient for the paper’s conceptual claims about understanding or interpretability. If this is a standard metric for the task, I would suggest toning down the conceptual claims. Similarly, the authors claim that interpretability is a feature of BUA, but only a few qualitative examples are provided. How interpretability or explainability is enhanced, compared to previous works, is a bit vague to me.
### 5. Syntehtic Data Experimental Details
In the synthetic data experiment, I think more experimental details are needed. Are "Real + Ours" trained with a commensurable number of gradient steps compared to "Real"? If not, please provide results where you control the number of epochs to make total number of gradient steps similar between the two setups. Also, please provide other experimental details somewhere in the appendix as well.
### 6. Minor Points
- BehaviorGPT --> BehaveGPT
- I suggest breaking Section 4 down into two sections (e.g. Experiments / Analysis).

### Questions
See Weaknesses

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
3

### Summary
This paper proposes Behavior Understanding Alignment (BUA) that integrates LLMs into human behavior modeling. BUA employs sequence embeddings from pretrained behavior models as alignment anchors and guides the LLM through a three-stage curriculum. Experiments on two real-world datasets demonstrate the effectiveness of BUA.

### Strengths
1, This paper enhances the LLM to understand human daily behavior sequences, which is helpful in many downstream tasks.
2. The three-stage curriculum learning pipeline works in aligning two models
3. Experimental results on two real-world datasets demonstrate the effectiveness of BUA method.

### Weaknesses
1, It should discuss more design choices to handle the behavior data.
2. The model design and similar training strategy can be found in different works
3. The presentation can be improved.

### Questions
1.	The human behavior data have the large chances to be in the training corpus. Thus, the LLM can understanding the meaning. Even if the data is collected in a structure form, the data can be translated into the natural language with the explanation of each symbols. Such a method can be more flexible in handling the dynamic data collected. I guess that the LLM method in Table accept the data in the raw form.

2.	The challenges faced by this paper, including long-tail behaviors and human-interpretable reasoning, also exist in other application domains, like adopting LLM into the medical or finance domain. It is better to include the methods in other domain and discuss the differences among them.

3.	In other part of the paper, the major challenges become the ID and embedding, e.g., the statement “a critical modality gap exists: human behavioral data, typically represented as sequences of IDs or embeddings”. Then, there is some kinds of confusion, like the relationship to the long tail mentioned early?

4.	You introduce another model when the user behavior data is too long to be put into the context. In such a context, one choice is to use LLM to summarize the data in the text form, without needing the alignment between two models. Such a method can be found in the agent memory, which also focuses on the user modeling. It is better to include them in the paper.

5.	The model framework is commonly used to handle multiple-modality problem, and the strategy to train/frozen different parts in a large model is also used in other work

6.	Figure 1 is not easy to understand. And I cannot see the challenges mentioned in the paper in this example. The features in the Figure 1(b) had better be explained more.


7.	When we fine-tune the model to understand the embedding from another sequence model, the order and kinds of tokens in the sequence model are important. How about the data format changes in the sequence model? In such a case, the embedding outputted from the sequence model may be changed, and the alignment may be invalid.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose to integrate large language models (LLMs) into human daily behavior modeling and represent human behaviors as event sequence.

The proposed framework BUA leverages sequence embeddings from BehaviorGPT as anchors to bridge the modality gap between behavioral data and natural language through a three-stage curriculum.

The experiments are conducted on two real-world user behavior datasets. The experimental results demonstrate improvements in both weighted precision and recall in prediction when it also outperforms certain baselines like SAND and D2A in generation tasks. The authors also conduct ablation studies to confirm that curriculum and multi-round dialogue settings are crucial.

### Strengths
* BUA effectively bridges the modality gap between behavioral sequences and LLMs through curriculum learning and sequence embeddings, enabling unified handling of prediction and generation while improving long-tail performance.
* This work demonstrates significant improvements over diverse baselines on real-world datasets, with detailed ablations, error analyses, and cross-model enhancements highlighting robustness and practical utility.
* The multi-round dialogue and self-reflective refinement produce human-readable textual summaries with better interpretability.

### Weaknesses
* Performance relies heavily on strong pre-trained behavior models like BehaviorGPT. The ablation study also shows drops when using weaker alternatives.
* The granularity and scope of activity modeling are limited. The study focuses on high-level daily activities and potentially overlooks nuanced behaviors.

### Questions
* How does the BUA framework handle missing or incomplete behavioral data in real-world scenarios? I wonder how the framework would deal the gaps or inconsistencies in behavioral sequences, which are much common in real-world logs and trajectory datasets.
* How does the BUA framework handle multi-user or group behavior modeling? The paper focuses on individual user behavior sequences but does not discuss extending the model to capture interactions or collective behaviors among multiple users.
* What are the limitations of freezing specific model components during different training stages? The paper mentions freezing certain parameters for efficiency, but it does not discuss potential trade-offs or constraints this imposes on model adaptability.

### Soundness
3

### Presentation
3

### Contribution
3
