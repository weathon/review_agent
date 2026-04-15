# ESQA: Event Sequences Question Answering

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 5

## Abstract
Event sequences (ESs) arise in many practical domains including finance, retail, social networks, and healthcare. In the context of machine learning, event sequences can be seen as a special type of tabular data with annotated timestamps. Despite the importance of ESs modeling and analysis, little effort was made in adapting large language models (LLMs) to the ESs domain. In this paper, we highlight the common difficulties of ESs processing and propose a novel solution capable of solving multiple downstream tasks with little or no finetuning. In particular, we solve the problem of working with long sequences and improve time and numeric features processing. The resulting method, called ESQA, effectively utilizes the power of LLMs and, according to extensive experiments, achieves state-of-the-art results in the ESs domain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel approach to handling event sequences (ESs) by adapting LLMs to the ES domain for multiple downstream tasks like extraction and prediction, without the need for extensive fine-tuning. This work addresses challenges in processing ESs, including long sequence lengths and temporal features. Key contributions include: (1) proposing a novel framework for processing long ESs using LLMs, and (2) leveraging the Q-Former to efficiently compress and integrate ESs into LLMs. State-of-the-art results on multiple ES tasks across several datasets are achieved.

### Strengths
- Originality: The paper introduces a creative adaptation of LLMs to the ES domain that has not been extensively explored with such models. While prior work has adapted LLMs to other non-language data (e.g., time series), the specific application to ESs with the question-answering format is novel.
- Quality: The paper offers extensive experiments on relevant datasets from real-world domains. The use of LLMs for ES modeling and the integration of Q-Former as a connector between the LLM and event sequence encoder are sound technical contributions.
- Clarity: The paper is generally clear in its description of the methods and experiments.
- Significance: The proposed approach is highly significant in the context of applied machine learning, particularly in industries reliant on ES data. The ability to leverage LLMs for ES tasks with minimal fine-tuning could lead to substantial efficiency improvements in data processing and decision-making systems.

### Weaknesses
- Quality: The proposed method has certain limitations, particularly in handling imbalanced classes and discretization errors in numeric features, as noted by the authors.
- Clarity: There are several typos and styling issues, for example:
  - Numerous citations should be in parenthesis using \citep command.
  - The authors have mentioned that "the second best results are underlined" in Table 4's caption, but the table contains no underlines.
  - In the captions of Table 3-5, "baseline approaches presented in Section 4.2.1" should actually be Section 4.1 or Appendix B.3.
- Significance: The model has been tested primarily on financial data, but not on more diverse domains, such as healthcare or social media interactions, thus the domain generalizability of the model is not demonstrated.

### Questions
1. What is the compress rate between the query embeddings and the textual representation of the structured data?
2. Have the authors considered alternative approaches for dealing with imbalanced classes, such as integrating class weights into the cross-entropy loss function?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces an LLM-based method for modeling event sequences. By leveraging LLMs, this research aims to improve the generalization and modeling quality of the existing methods for event sequence prediction tasks. Benefiting from LLM's knowledge from other domains, the proposed method is able to perform unseen tasks. The experimental results demonstrate the capability of the proposed method in achieving state-of-the-art results on certain tasks compared to the existing methods.

### Strengths
1. The paper proposes a method to encode features of different formats and integrate the embedded features with questions as input to LLMs. The motivation for using LLMs is clearly presented.
2. By leveraging LLMs, the proposed method can generalise to unseen tasks.
3. The proposed method outperforms the baseline methods on some of the datasets.
4. In general, the paper is well-written and easy to follow.

### Weaknesses
**Novelty**:
1. The paper claimed it is the first to apply LLMs to event sequences. This claim does not seem to be correct. There have studies [1,2] using LLMs for event sequences.
2. The technical novelty of this paper is weak. The event encoder adopts several well-known machine learning techniques, such as binning, to convert the raw features to learnable embeddings. The connector is a direct application of an existing technique Q-former. The training protocol applies LORA, a well-known fine-tuning technique for LLMs. It is hard to see the originality of the proposed method.

**Method Design**:

1. Although discretisation is one way to convert numeric features into categorical features, it also increases the dimensions and parameters of the model. If an event has a large number of features, the event embedding (obtained by concatenating the embedding of each feature) can be very large, which may lead to a large number of parameters in the projection layer. Discretisation may not be parameter-efficient in this case.
2. The method is limited to categorical and numeric data. It lacks support for other data types. For instance, in social media, plain text and images are very common attributes.
3. The motivation for choosing only FLAN-T5 as the backbone is unclear.

**Experiments**:

1. Although the paper claimed event sequences are prevalent in real-world applications, including social networks and healthcare, the paper only evaluates the methods on fintech data.
2. The latest methods compared in the paper were published in 2022. More recent methods could be included as baselines, such as [1,2]. Besides, the proposed method does not have fundamental differences compared to LLMs for time series or temporal knowledge graphs, except for the feature encoder; it is a question why the authors do not compare with those LLM-based methods in other domains.
3. The proposed method underperforms the baselines on certain datasets. However, there is no in-depth analysis of the reason. 
4. LLMs are known to be unstable in text generating. The output of independent runs might be different. However, there is no standard deviation is reported in the experimental results. Compared to existing methods, LLM-based method might have a larger variance. The authors need to justify that the proposed method is stable and that the improvements are statistically significant.
5. The experimental setting for the generalization abilities is not clearly presented. The method ESQA m/t is trained in a multi-task setting. Do these tasks include the task in the testing phase? The low performance of ESQA z/s is not surprising because it is trained only one a single task. 

**Minor comments**: 

1. The citation format seems to be incorrect. For instance, "This type of data is widely spread in domains like geoscience Bergen et al. (2019)" --> "This type of data is widely spread in domains like geoscience (Bergen et al. 2019)".
2. The definition $c_j^{cat} \in |c_{j}|$ is confusing. $|c_{j}|$ is often used to denote the size of a set.
3. On page 3, "a sequence of f-dimensional vectors in$E_{\theta}$". There should be spacing between "in" and "$E_{\theta}$".

**Reference**:

[1] Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei. Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning. NeurIPS 2023.

[2] Matthew McDermott, Bret Nestor, Peniel Argaw, Isaac S Kohane. Event Stream GPT: A Data Pre-processing and Modeling Library for Generative, Pre-trained Transformers over Continuous-time Sequences of Complex Events. NeurIPS 2024.

### Questions
1. Converting the structured features of an event is not difficult. Once converted, it becomes a sequence of embedding, which is identical to other LLM-based sequence modeling methods, e.g., those for time series. What is the uniqueness of this paper compared to other LLM-based sequence prediction problems?

2. What is the reason for only considering numeric and categorical features?

3. Why FLAN-T5 is used as the backbone? Have you tested other LLMs such as GPT4, Llamma3, etc.?

4. Only evaluating the method on fintech data cannot justify its generalisability. Why not compare these methods on social media data and healthcare data?

5. What is the exact reason that the proposed method underperforms the competitors on AlfaBattle and Gender datasets? What are the unique properties of these datasets that lead to the performance degrade of the method? You mentioned the two datasets exhibit class imbalance. Does this mean your method is vulnerable to imbalanced data?

6. What are the tasks used for training ESQA m/t, and how are they different from the tasks in the testing phase?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces ESQA (Event Sequences Question Answering), a framework designed to handle question-answering tasks in the domain of event sequences. Leveraging large language models (LLMs), ESQA addresses unique challenges in processing temporally structured data, such as managing long input sequences and encoding time and numeric features effectively. The framework showcases its adaptability across various downstream tasks without extensive fine-tuning, achieving comparable performance across multiple event-sequence datasets in domains like finance and e-commerce.

### Strengths
S1: Experimental Design: The paper supports its proposed approach with a series of experimental setup, utilizing multiple datasets that span various event sequence types and domains, enhancing the robustness and applicability of the findings. 

S2: Each component of the ESQA framework is backed by experimental results, validating the selection and effectiveness of different modules. 

S3: ESQA achieves state-of-the-art or competitive results on a range of tasks, highlighting its versatility and effectiveness in handling diverse event-sequence tasks.

### Weaknesses
Overall, the paper lacks novelty beyond the designed QA framework with backbone encoder and LLM. Moreover, the organization and experiment settings in paper can be further optimized. 

W1: Inadequate Background Explanation: The paper contains some strongly worded statements in the background, such as “no effort was made to adapt LLMs to event sequences.” However, event sequence tasks share similarities with event prediction tasks, making it essential for the authors to clarify the differences between these two. Since LLMs have already been applied to various event prediction tasks, it would be valuable to consider these models as baselines for comparison. Additionally, there is considerable prior work on LLMs in question answering, and including these models as baselines could further strengthen the experimental design. 

W2: Lack of Rigorous Annotation: The methodology section lacks precise annotation and contains numerous typographical issues. For example, “Q-Former” in the related work section is not properly cited, which could lead to misunderstandings. Some symbols, such as special tokens, are only briefly introduced in figures without corresponding explanations in the text. This could create confusion among readers, highlighting the need for careful annotation and early definition of terms. 

W3: Insufficient Justification of Component Selection: The paper could enhance the methodology by explaining the motivations behind selecting each component of the framework, beyond the experimental advantages. A discussion of why each component was chosen in alignment with the framework’s goals would strengthen the readers' understanding of the architecture. 

W4: Mismatch Between Methodology Text and Figures: Certain descriptions in the methodology section do not fully align with the figures. For example, how the Input prompt, query, and Question are integrated is not clearly explained, and the division in the embedding matrices shown in Figure 2 could lead to misinterpretation. Moreover, although the authors state that the LLM is not fine-tuned, LoRA, a traditional fine-tuning technique—is used. These inconsistencies should be clarified to avoid potential misunderstandings. 

W5: Limited Experimental Baselines: The experimental design could benefit from considering more comprehensive baseline models from existing LLM research, especially LLM applications in question answering and traditional event prediction, to provide a more robust evaluation of the proposed framework.

### Questions
Refer to the weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors present a new approach to apply large language models to the task of event sequence prediction. This approach is a little fine-tuned and less expensive to use. It takes into account both time and numerical sequence processing. The authors chose FLAN-T5 as the base model and conducted experiments on several datasets related to customer transactions. In addition to experiments on specific downstream tasks, the generalization ability of the large language model for event sequence prediction was tested.

### Strengths
1. The authors take advantage of categorical and numeric features from tabular neural networks at the same time in Section 3.2.
2. In Section 3.3, the choice of Whisper as an Encoder after extensive comparative experiments is convincing.

### Weaknesses
1. I think the biggest problem with this paper is the lack of sensible baselines. The authors ignore some recent applications of large models for event prediction in the last year. For example [1], [2]. The authors use earlier baselines and therefore the results obtained are not convincing. In particular, according to Table 3, ESQA does not achieve a consistent lead compared to NPPR either.
[1] Analyzing Temporal Complex Events with Large Language Models? A Benchmark towards Temporal, Long Context Understanding
[2] MIRAI: Evaluating LLM Agents for Event Forecasting

2. I think the novelty of the ESQA method itself is limited. The authors have mentioned time-series prediction in the related work section. In fact, there is a lot of recent work dealing with complex data structures such as time-series, e.g., [3], [4]. They have embedded complex data structures individually and used the combination of freezing and tuning. So, I don't think similar ideas used for event sequence prediction task is very novel.
[3] TIME-LLM: TIME SERIES FORECASTING BY REPROGRAMMING LARGE LANGUAGE MODELS
[4] LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters

3. The authors' descriptions of their methods and the results are often confusing. For example, the Encoder model is used in Figure 1, and the Event Sequence Encoder is used in Figure 3.

### Questions
1. Why did you choose only a few datasets for transactions and not for other domains? You have mentioned in the introduction section that event prediction is widely used in various fields such as healthcare, sociology etc. I have read the sources of the datasets you have used and found that they are rarely used by other works. So, I think the datasets are not representative.
2. Why are all the large language models you use for experiments the FLAN-T5 series? Why don't you use the newer LLaMA series or the Mistral series?
3. How did you get the two empirical values of 0.6 and 1.56 in rows 168-169? Have you tried other values?

### Soundness
2

### Presentation
2

### Contribution
2
