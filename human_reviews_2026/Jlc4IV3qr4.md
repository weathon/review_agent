# Knowledgeless Language Models: Decoupling Linguistic Competence and Factual Knowledge

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 8, 4

## Abstract
Language models capture a broad spectrum of human knowledge due to being trained on large and diverse real-world datasets. However, this knowledge is not always necessary for linguistic tasks and can contribute to hallucinated outputs, as real-world knowledge is inherently dynamic and context-dependent. Such behavior limits their applicability in domains where factual precision is critical, such as healthcare and law. Moreover, LLMs trained on large text corpora inevitably inherit societal biases present in their sources. In this work, we introduce Knowledgeless LMs (KLLMs), a class of models intentionally pretrained to forgo memorization of entity-specific knowledge while retaining structural and semantic understanding of language. We present our approach for designing and training these models and evaluate them across a spectrum of downstream tasks, including language understanding, commonsense reasoning, and context-based factual benchmarks. Our results show that KLLMs achieve competitive or superior performance compared to fully parametric LLMs, particularly when provided with the relevant context, while substantially reducing reliance on memorized world knowledge. This leads to lower hallucination risks and improved calibration, with more reliable confidence estimates. Overall, KLLMs demonstrate that strong linguistic and reasoning capabilities can be maintained without extensive factual memorization, highlighting knowledgeless pretraining as a promising paradigm for building more efficient, faithful, and controllable language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Knowledgeless Language Models, a novel class of models designed to decouple linguistic competence from entity-specific knowledge through a deliberate anonymization strategy during pretraining. The core methodology involves preprocessing training data by identifying named entities via a state-of-the-art NER system and replacing them with type-based placeholders, thereby preventing the model from memorizing real-world facts while preserving grammatical and semantic structures.

### Strengths
- When provided with external context, KLLMs outperform standard models on factual reading comprehension and commonsense reasoning, demonstrating their ability to leverage contextual cues effectively.
- By eschewing entity memorization, KLLMs can be pretrained on smaller, less specialized corpora, reducing computational costs and resource requirements.

### Weaknesses
- The most significant concern lies in the pretraining methodology. The paper acknowledges that the training corpus (a combination of CNN/DailyMail and Wikipedia) is "substantially smaller" than the corpora used for original models like LLaMA. The provided loss curves are critical evidence of potential under-fitting.
- The authors estimate the NER accuracy at only ~87%, meaning 13% of entity mentions remain unmasked, creating a direct source of knowledge leakage.
- In closed-book settings, KLLMs perform near random chance, highlighting their reliance on provided context. This limits applicability in scenarios where real-time retrieval is impractical.

### Questions
- If the KLLM and the baseline SLM are both under-trained due to data constraints, do the reported performance differences truly reflect the merits of the anonymization approach, or merely different convergence states on an insufficient task?
- The paper motivates KLLMs as a means to mitigate societal biases but provides no empirical evidence to support this claim. For example, how does the model avoid perpetuating gender biases present in the training data?

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
This paper introduces Knowledgeless Language Models (KLLMs), which are language models that are trained on data with anonymized entities. The experiments demonstrate that KLLMs can achieve strong performance on some NLU tasks and calibration evaluation.

### Strengths
This paper provides a simple but effective way to decouple linguistic competence and factual knowledge in language models by anonymizing the entity at the training stage.

The knowledgeless language models demonstrate strong performance on the provided benchmarks.

### Weaknesses
The experiments mainly focus on the NLU tasks, the performance on generation and reasoning tasks is not discussed, which is a core capability of the language models. 

The anonymization method is widely used in literatures to disentangle memory and reasoning abilities, which may restrict the novelty of the paper.

The conclusion is not convincing enough. As the SLM and KLLM are trained on data with and without anonymization, the performance should be measured on both anoymized and non-anoymized data.

### Questions
The paper mentions that using RAG is a good way to mitigate hallucination and provide grounded responses. Can KLLMs cooperate with RAG to achieve competitive performance on factual knowledge related tasks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce KnowledgeLess Language Models (KLLMs), a family of LLMs which are trained to retain structural and linguistic understanding while minimizing the reliance on their “world” (i.e., parametric) knowledge. KLLMs are trained by first anonymizing the pre-training corpus and doing standard LLM pre-training. KLLMs show strong results across a variety of benchmarks when compared to the standard language models (SLMs).

### Strengths
- The paper and motivation is very clear and well-written. 
- The research question being asked is novel and important: I believe it is an interesting research direction to develop models that only have the language capabilities of LLMs and don’t rely on memorized knowledge, especially for use-cases in which memorized knowledge can hurt (like RAG). This study provides a strategy to do this that can easily be applied and built on by future research. 
- The method is simple, yet effective. It is straightforward to implement and provides consistent improvement in effectiveness across and variety of tasks, models and datasets.

### Weaknesses
- While I liked the depth of evaluation performed by the authors, I would have liked to see more ablation studies that investigated the robustness of their approach, similar to that of Table 5. For example:
  - How would KLLMs perform if they were first pre-trained using a standard objective, then further pre-trained using the proposed procedure? In other words, imagine starting the procedure from Llama pre-trained weights rather than from scratch. It would be interesting to see, for example, how the original pretrained model can benefit from such an approach. (I.e., does the proposed approach only perform well when trained from scratch?)
  - Understandably, the authors choose to anonymize at inference-time, but I would like to have seen some results if they did not apply such anonymization at inference. Is this necessary? How much does performance get impacted? How does this compare to the SLM?
  - How would effectiveness change if you used other models for corpus anonymization? For example, how would using a strong, larger LLM compare to the OntoNotes model?
  - Furthermore, I would have liked to see some qualitative examples that show why anonymization helps performance. For example, cases in which SLMs fail but KLLMs don’t or vice versa
- Why did you perform experiments with Llama on certain tables but not for others? I think consistency here would have been helpful as it would be interesting to see how different model families might impact results.

### Questions
- Do you always see KLLMs as an approach for smaller, specialized corpora or can this be done at scale?
- In real-world RAG setup, it is likely that relevant context may not be available, how would your method perform in these cases?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The recent development of large language models (LLMs) faces hallucinations and societal biases that are in the training text corpora. In order to mitigate these problems, this paper proposes Knowledgeless LLMs (KLLMs). During training, it identifies named entities such as persons, nationalities, facilities, locations, etc., and replaces them with special tokens (anonymization). Therefore, the training process intentionally reduces factual knowledge information, and the LLM will focus more on the structural and semantic understanding of language. During inference, the framework substitutes the named entities in the LLM’s output back to their normal names. Experiment results verify KLLM’s effectiveness from different perspectives.

### Strengths
(1) The motivation is good and meaningful. LLMs indeed face hallucinations and societal bias originating in the training corpus, so it is meaningful to study methods to solve this problem. This line of research could benefit potential applications.

(2) The method design generally makes sense, and it is quite concise to implement.

(3) Experiment results verify KLLM’s effectiveness on different tasks from different perspectives. KLLM outperforms the standard LLMs on several tasks, and the closed-book QA experiments verify that KLLM effectively prevents the model from learning knowledge.

### Weaknesses
(1) The method and analysis are relatively simple. The method is to recognize the named entities and anonymize them. It would be better if this paper could explore more designs of the anonymization process or named entity recognition, which would contribute to more insights.

(2) As mentioned in Line 126, the F1 score of named entity recognition is 90%. Named entity recognition is a crucial basis of this method. The unrecognized entities could result in knowledge leakage, which is a limitation of the current method.

(3) In the experiments, the model scale might be insufficient. The experiments are conducted on models <= 3B. I am wondering whether these conclusions could generalize to larger models. It would be more convincing if some experiments were conducted on larger models, such as Llama-3.1-8B.

(4) In the experiments, the dataset scale might be insufficient. The CNN/DailyMail pretraining corpus contains 272M tokens, and Wikipedia contains 2.2B tokens. It might be smaller than many pre-training works. Appendix B shows a training loss curve, where the anonymized model still has a high loss, so I am not sure whether its training has fully converged. While the paper mentions that “KLLMs offer practical advantages, including reduced pretraining costs,” there are no experiment results to support this conclusion.

### Questions
Q1. In Appendix B, the training loss of KLLM is much higher than that of standard training. KLLM’s performance is better than the standard training process, why does KLLM have much higher loss?

Please also refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
