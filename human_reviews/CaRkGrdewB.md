# 'No' Matters: Out-of-Distribution Detection in Multimodality Long Dialogue

- Avg Score: 4.60
- Decision: Reject
- Scores: 3, 5, 5, 5, 5

## Abstract
Out-of-distribution (OOD) detection in multimodal contexts is essential for identifying deviations in combined inputs from different modalities, particularly in applications like open-domain dialogue systems or real-life dialogue interactions. This paper aims to improve the user experience that involves multi-round long dialogues by efficiently detecting OOD dialogues and images. We introduce a novel scoring framework named **D**ialogue **I**mage **A**ligning and **E**nhancing **F**ramework (DIAEF) that integrates the visual language models with the novel proposed scores that detect OOD in two key scenarios (1) mismatches between the dialogue and image input pair and (2) input pairs with previously unseen labels. Our experimental results, derived from various benchmarks, demonstrate that integrating image and multi-round dialogue OOD detection is more effective with previously unseen labels than using either modality independently. In the presence of mismatched pairs, our proposed score effectively identifies these mismatches and demonstrates strong robustness in long dialogues. This approach enhances domain-aware, adaptive conversational agents and establishes baselines for future studies.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper studies the Out-of-Distribution (OOD) detection problem in a multimodal context involving an image and a corresponding dialogue discussing that image. The authors introduce a new scoring framework, based on a visual-language model, to detect mismatches between the dialogue and image input pairs, as well as input pairs with previously unseen labels. The experimental results indicate that this approach outperforms using either modality independently for detection. However, I question the significance of this task. Given the capabilities of current large vision-language models, which are quite powerful, would they not be able to handle any domain images effectively? Why is this task still relevant? Regarding the proposed approach, I found it lacking in novelty and differentiation compared to existing methods. Additionally, the comparisons are outdated, with the most recent being from 2018, which diminishes the credibility of the claims.

### Strengths
1. The paper is well-written and clearly presented.
2. Although the proposed scoring function is relatively simple and lacks novelty, the authors provide a detailed analysis and explanation of the intuition behind it.
3. The experiments are extensive and thorough.

### Weaknesses
1. The motivation behind the task is unclear, particularly considering the current capabilities of powerful vision-language models, which may already handle OOD scenarios effectively. The relevance of this task remains questionable.
2. The proposed method lacks substantial novelty and does not significantly differentiate itself from prior approaches.
3. The comparisons with existing methods are outdated, with the most recent being from 2018, which weakens the evaluation of the paper's contributions in the context of current research. Additionally, there is no comparison with current large vision-language models (LVLMs) for detection. How do models like GPT-4o, Claude-3.5-Sonnet, Gemini, and Qwen2-VL perform on this classification task?

### Questions
1. Why is this task still essential in 2024?
2. How do large vision-language models (LVLMs) perform in direct OOD detection?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper addresses the challenge of Out-of-Distribution detection in multimodal long-dialogue systems, where text and image modalities are combined, especially in real-world or open-domain dialogue applications. The authors propose the Dialogue Image Aligning and Enhancing Framework, designed to detect two main types(Mismatch between dialogue and image, Unseen labels) of OOD cases.

### Strengths
1. This paperintroduces a unique approach for OOD detection by combining image and dialogue data, addressing the limitations of using single modalities.

2. The framework's use of an alignment and enhancement scoring mechanism allows for precise multimodal OOD detection.

3. By focusing on mismatched pairs and unseen labels, the framework is suited for real-world applications where dialogue and visual information often co-occur.

### Weaknesses
This paper has chosen multimodal, multi-turn dialogue environments as the task to perform and verify OOD detection. The authors claim this is necessary for user satisfaction and trust, but this part of the argument is not convincing. They need to provide logical supplementation on why multimodal OOD detection tasks are important in dialogue, and why they are particularly important in multi-turn rather than single-turn interactions.
Similar problem definitions were found in other papers [1,2], but references to these papers are missing, and explanations and quantitative metrics are needed to show how this paper differentiates itself from these works. The baseline methods used in this paper are all methodologies from before 2019, and experiments should be designed to include methodologies from recent papers.


[1]. GENERAL-PURPOSE MULTI-MODAL OOD DETECTION FRAMEWORK, V Duong et al. 2023
[2]. MultiOOD: Scaling Out-of-Distribution Detection for Multiple Modalities, H DOng et al. 2024

### Questions
1. It would be better to identify and supplement experiments with use cases where multimodal OOD is important and can be well utilized, rather than focusing on the multi-turn dialogue setting.
2. It would be good to add comparative experimental results with recent papers that have proposed solutions to multimodal OOD problems.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a new framework, Dialogue Image Aligning and Enhancing Framework (DIAEF), to improve the user experience in multi-round dialogues by efficiently detecting out-of-distribution (OOD) instances in multimodal contexts, specifically dialogue-image pairs. DIAEF integrates visual language models with novel scoring mechanisms to identify OOD cases in two main scenarios: mismatches between dialogue and image inputs and previously unseen labels in input pairs. Experimental results show that the combined use of dialogue and image data enhances OOD detection more effectively than using each modality independently, demonstrating robustness in prolonged dialogues. This approach supports adaptive conversational agents and sets a benchmark for future research in domain-aware dialogue systems.

### Strengths
- This paper is well-written, especially considering that the topic of OOD detection is not easy to understand. For instance, the authors explain the problem formulation of "cross-modal OOD detection" clearly.
- The paper introduces a new paradigm and framework for OOD detection in "multi-turn interactive dialogue," along with a new scoring method, DIAEF, which utilizes vision-language models.
- They demonstrate the effectiveness of DIAEF both experimentally and theoretically and suggest its potential as an alternative scoring method, as shown in Table 1.
- Through extensive experiments, the authors show that DIAEF outperforms other OOD detection scoring methods and empirically validate their design choices (e.g., the selection of alpha).

### Weaknesses
- Although the authors clearly present the problem formulation of "cross-modal OOD detection," I still find the use of OOD terminology in the multi-modal dialogue domain unclear. Dialogue inherently has a subjective nature and a one-to-many structure (i.e., diversity [1]), meaning that even with the same query, there are multiple possible responses depending on the situation and the user in real-world interactions. Therefore, I question whether using the term "OOD" is appropriate in this context. The authors should further clarify why handling OOD detection in the multi-modal domain is necessary.
- Additionally, I am concerned that using CLIP or BLIP models may not ensure adequate understanding of dialogue, as CLIP has a limited context length of 77 tokens, and neither CLIP nor BLIP is pretrained on open-domain dialogue datasets—issues highlighted in prior works [2-3]. When determining OOD, it seems that the embedding model reflects its training distribution, yet CLIP embeddings may be ineffective for dialogue. I believe that using LongCLIP [4] could be a better alternative. Therefore, the authors should clarify their choice of CLIP or BLIP for the VLM models.
- In the DIAEF framework, training the "label extractor" is crucial; however, I don’t fully understand what constitutes a "label" in an "open-domain dialogue." Could you explain this?
- While the authors demonstrate the effectiveness of their framework, more experiments are needed to establish its robustness and reliability across additional dialogue datasets. The framework formulation includes multiple hyperparameters (e.g., $\alpha$ and $\gamma$), and the MMD dataset is not a high-quality multi-modal dialogue dataset since it is synthesized using CLIP matching, despite the application of human crowdsourcing to verify contextual relevance. Nevertheless, this dataset lacks both high quality and diversity, which is mentioned in the prior work [5]. I recommend that the authors conduct experiments on additional dialogue datasets, such as PhotoChat [6], MP-Chat [7], ImageChat [8], and DialogCC [5]. Given time constraints, it is unnecessary to experiment on the full datasets; subsampled versions would suffice.
- I am also curious as to why the authors focus on "long dialogue," as, to my knowledge, the datasets used in the experiments emphasize single-session dialogues rather than multi-session dialogues like MSC [9] or Conversational Chronicles [10].

---

**References**

[1] Li, Jiwei, et al. "A diversity-promoting objective function for neural conversation models." arXiv preprint arXiv:1510.03055 (2015).

[2] Yin, Zhichao, et al. "DialCLIP: Empowering Clip As Multi-Modal Dialog Retriever." ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024.

[3] Lee, Young-Jun, et al. "Large Language Models can Share Images, Too!." arXiv preprint arXiv:2310.14804 (2023).

[4] Zhang, Beichen, et al. "Long-clip: Unlocking the long-text capability of clip." arXiv preprint arXiv:2403.15378 (2024).

[5] Lee, Young-Jun, et al. "Dialogcc: Large-scale multi-modal dialogue dataset." arXiv preprint arXiv:2212.04119 (2022).

[6] Zang, Xiaoxue, et al. "Photochat: A human-human dialogue dataset with photo sharing behavior for joint image-text modeling." arXiv preprint arXiv:2108.01453 (2021).

[7] Ahn, Jaewoo, et al. "Mpchat: Towards multimodal persona-grounded conversation." arXiv preprint arXiv:2305.17388 (2023).

[8] Shuster, Kurt, et al. "Image chat: Engaging grounded conversations." arXiv preprint arXiv:1811.00945 (2018).

[9] Xu, J. "Beyond goldfish memory: Long-term open-domain conversation." arXiv preprint arXiv:2107.07567 (2021).

[10] Jang, Jihyoung, Minseong Boo, and Hyounghun Kim. "Conversation chronicles: Towards diverse temporal and relational dynamics in multi-session conversations." arXiv preprint arXiv:2310.13420 (2023).

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

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
This paper take the first attempt for OOD detection in multimodality long dialogue, propose a framework that enhances the OOD detection in cross-modal contexts，achieved the combination of OOD detection and multimodal methods. 

And it demonstrate that integrating image and multi-round dialogue OOD detection is more effective with previously unseen labels than using either modality independently.

### Strengths
The starting point chosen for the paper is quite innovative.

### Weaknesses
The innovation of the methods used in the paper needs to be strengthened.

### Questions
none

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses the challenge of out-of-distribution (OOD) detection in multimodal contexts, particularly focusing on the combined input of dialogues and images in real-life applications such as open-domain conversational agents. It introduces the Dialogue Image Aligning and Enhancing Framework (DIAEF), an approach for detecting mismatches in dialogue and image pairs and identifying previously unseen input labels in conversations. DIAEF integrates visual language models with scoring metrics tailored for two primary OOD scenarios: (1) detecting mismatches between dialogue and image inputs, and (2) flagging dialogues with previously unseen labels. Experiments conducted on several benchmarks indicate that DIAEF’s integrated approach to image and multi-round dialogue OOD detection outperforms single-modality methods, especially in dialogues involving mismatched pairs and extended conversations.

### Strengths
- The paper focuses on two key types of out-of-distribution (OOD) scenarios: (1) mismatches between dialogue and image inputs, and (2) inputs with previously unseen labels. It demonstrates the effectiveness of the proposed method in accurately identifying these OOD cases.
- This work marks the first attempt to address OOD detection in dialogue contexts, specifically for multi-round conversations. To support this, the authors constructed a new dataset for multi-round question-answering, enabling comprehensive evaluation of the framework’s performance in real-life dialogue settings.

### Weaknesses
- Models like CLIP and BLIP are primarily trained for image captioning, and some previous researches suggest that they may not generate optimal text embeddings for dialogue. How does this paper address the potential limitations of using these models in a dialogue context to ensure accurate and meaningful embeddings?
- Does the proposed method consider only yes/no question-answer dialogues as in-domain scenarios? If so, when OOD situations become more complex, it’s unclear how well the method would perform or if it would remain effective in identifying out-of-domain cases accurately.

### Questions
Can you include the example of the test set generated?

### Soundness
2

### Presentation
2

### Contribution
2
