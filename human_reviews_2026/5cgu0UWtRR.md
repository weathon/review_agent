# Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Fine-tuning large language models (LLMs) can lead to unintended out-of-distribution generalization. Standard approaches to this problem rely on modifying the training data, for example by adding data that better specify the intended generalization. However, this is not always practical. We introduce Concept Ablation Fine-Tuning (CAFT), a technique that leverages interpretability tools to control how LLMs generalize from fine-tuning, without needing to modify the training data or otherwise use data from the target distribution. Given a set of directions in an LLM's latent space corresponding to undesired concepts, CAFT works by ablating these concepts with linear projections during fine-tuning, steering the model away from unintended generalizations. We successfully apply CAFT to three fine-tuning tasks, including emergent misalignment, a phenomenon where LLMs fine-tuned on a narrow task generalize to give egregiously misaligned responses to general questions. Without any changes to the fine-tuning data, CAFT reduces misaligned responses by 10x without degrading performance on the training distribution. Overall, CAFT represents a novel approach for steering LLM generalization without modifying training data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the unexpected problems in out of distribution (OOD) generalization of large language models (LLMs), and innovatively proposes a concept ablation fine-tuning (CAFT) framework. By using mechanical interpretability tools to locate the unexpected concept directions in the latent space of the model, the projections of these directions are melted during the fine-tuning process, enabling the model to guide generalization behavior without modifying the training data. The paper validated the effectiveness of the method through three typical tasks (emergence misalignment, gender bias, and double multiple choice), achieving significant suppression of unexpected generalization on models such as Qwen, Mistral, Gemma (such as reducing emergence misalignment rate by 10 times) while maintaining task performance within the distribution. The unique research perspective and clear technical roadmap provide an important solution for solving the problem of LLM generalization control in scenarios where data is difficult to modify. However, there is still room for improvement in terms of theoretical depth, experimental completeness, and practical applicability of the paper.

### Strengths
- CAFT proposed a novel OOD generalization control method that does not rely on data modification. This has significant practical value in worst-case scenarios where data collection is difficult or there are privacy restrictions.
- This method has achieved great success in addressing the emerging and serious security issue of "emergent misalignment", reducing the misalignment response rate by 10 times while maintaining high competitiveness on the target task (Vulnerable%)
- This article is a successful example of interpretability driven control. It demonstrates how to leverage insights from tools such as PCA and SAEs to transform them into actionable training interventions for precise guidance of model behavior.

### Weaknesses
- The efficacy of CAFT relies on a critical human-in-the-loop (or auxiliary model) step to interpret and select the "undesired" PCA or SAE directions for ablation . This introduces a significant bottleneck regarding subjectivity, time, and scalability. While Appendix C explores automated interpretation and shows promise , this manual/semi-automated dependency remains a core limitation.
- The method's success is contingent on the ability of the interpretability tools to effectively disentangle concepts. As shown in the "verbs | pronouns" task, CAFT fails when the intended and unintended concepts are semantically related (e.g., both grammatical) and not clearly separable in the activation space .
- The two multiple-choice tasks (Gender Bias, Double MC) feature artificial, 100% spurious correlations within $D_{train}$. While acknowledged as "unnatural", this extreme setting likely makes the undesired concepts (the shortcuts) more pronounced and easier to identify and ablate than the subtle, diffuse biases present in real-world datasets.
- The paper observes that SAEs excel on spurious correlation tasks while PCA excels on emergent misalignment , but it fails to provide a deep analysis of why. It lacks a principled discussion (e.g., SAEs capturing local features vs. PCA capturing global activation differences) that would provide a clear selection criterion for which tool to use for a given OOD task.

### Questions
- How can the CAFT framework scale beyond the current human-in-the-loop interpretation bottleneck ? 
- Does CAFT's reliance on linear projection  inherently limit its effectiveness to cases where intended and unintended concepts are linearly separable? How would the method perform if concepts are non-linearly entangled?
- How robust is CAFT to more realistic, non-deterministic spurious correlations (i.e., where the correlation is < 100%)? Does the method's ability to identify and ablate undesired concepts degrade as the correlation becomes more subtle or diffuse?
- Why does PCA prove more effective for the abstract "emergent misalignment" task, while SAEs are more effective for the concrete, token-level "spurious correlation" tasks? What does this imply about the nature of these phenomena and the types of features these two interpretability methods capture?

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
This paper presents CAFT, a method to identify unintended latent concepts during fine-tuning in order to improve OOD generalization without modifying training data. By acting during the fine-tuning process (through controlling specific latent properties) CAFT is able to showcase strong empirical results across several models.

### Strengths
The reviewer notes the following strengths of the paper:
- The proposed CRAFT method is intuitive & simple to implement and can be widely implemented across model fine-tuning.
- The authors presents strong results across multiple models alongside a wide range of ablation studies.
- The paper also provides a lot of context to support the validity of the methodology.

### Weaknesses
The reviewer notes the following weaknesses of the paper:
- There are many missing comparative methods which are commonly found in OOD generalization literature. In particular, common fine-tuning regularization baselines used for OOD generalization like L2-SP are missing from the evaluations [1].
- Additionally, the reviewer would also consider the authors to include specific linear-probing techniques which are also commonly used to address concerns of OOD generalizations [2].

[1] Xuhong, L. I., Yves Grandvalet, and Franck Davoine. "Explicit inductive bias for transfer learning with convolutional networks." International conference on machine learning, 2018.

[2] Kumar, Ananya, et al. "Fine-tuning can distort pretrained features and underperform out-of-distribution." (2022).

### Questions
The reviewer's primary concerns lies in the lack of comparative methods between CRAFT and other common OOD generalization using regularization or linear-probing. The reviewer would highly encourage the authors to include these comparative methods or potentially attempt to merge these method into CRAFT to obtain stronger results.

### Soundness
3

### Presentation
2

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
The issue of unintended generalization during the fine-tuning of large language models (LLMs) refers to the phenomenon where LLMs generalize in undesired ways, such as generating harmful responses, when encountering inputs that are out of distribution from the fine-tuning dataset. Current approaches to addressing this issue typically involve adding more representative data or removing the data responsible for misgeneralization. However, these methods are ineffective in scenarios where access to data specifying the intended generalization is unavailable. To address these limitations, the authors introduce Concept Ablation Fine-Tuning (CAFT), a method that identifies the "directions" in the model's latent space corresponding to undesired concepts and ablates these "directions" during fine-tuning. To identify these undesirable directions, two approaches are explored: Principal Component Analysis (PCA) and Sparse Autoencoders (SAEs). CAFT ablates the identified directions by projecting them onto orthogonal components. The authors demonstrate the feasibility of CAFT across three tasks, including emergent misalignment and two multiple-choice tasks.

### Strengths
1. The research topic of this work is very new and cutting-edge.

### Weaknesses
1. The overall writing and coherence of the paper need significant improvement, especially in the method and experiment sections, which are difficult to follow and understand.

2. The core method, CAFT, requires optimization, including but not limited to:

   - CAFT is the central contribution of this paper, but there is no framework diagram or overall introduction of the method.

   - The paper lacks an explanation of the projection method used to ablate directions.

   - CAFT lacks innovation, as it primarily combines existing methods into a pipeline rather than introducing novel ideas.

3. Although the paper conducts experiments on three tasks, two of the multiple-choice tasks (aside from emergent misalignment) are poorly designed, with many unclear and incomplete descriptions.

4. The paper does not provide reasonable baselines. The methods mentioned in *section 4.3* only serve to demonstrate the rationality and effectiveness of the CAFT method, and do not qualify as competitive or valid baseline methods.

### Questions
1. The authors mention that one limitation of the CAFT method is the need for humans to interpret directions. However, the term "directions" is not clearly defined in the paper. 

   Could you provide a formalized representation to help clarify this concept?

2. Could the authors explain the prompt design for the Double Multiple Choice task in Figure 6 on the OOD dataset? Specifically, why is the expected choice `"A. positive, football"`?

3. Referring to the description of emergent alignment by Wang et al. [1], the problem the authors aim to address is the undesired generalization that occurs on OOD datasets when models are trained on a **narrow and incorrect** dataset. 

   If my understanding is correct, could the authors further explain the design considerations for the other two tasks, apart from emergent misalignment, by completing the table below?

   |         Task           |      Train Target      | Undesired Generalization on OOD |
   | ------------------ | :--------------------: | :-----------------------------: |
   | Emergent Alignment | Generate insecure code |          Misalignment           |
   | Gender Bias        |                        |                                 |
   | Double MC          |                        |                                 |

4. In the right subfigure of Figure 5, the model's performance after fine-tuning is evaluated using "vulnerable". What evaluation metrics were used for the gender bias and double multiple-choice tasks? the corresponding results?

5. In the emergent misalignment task, `Qwen2.5-Coder` and `Mistrial-Small` were included in the experiments. However, for the other two tasks, only the `Gemma-2-2B model` was tested. Why were different models used for the latter two tasks? Does conducting experiments on a single model provide sufficient evidence for the effectiveness of the CAFT method?

   

[1] Wang, Miles, et al. "Persona Features Control Emergent Misalignment." *arXiv preprint arXiv:2506.19823* (2025).

### Soundness
2

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
The paper proposes Concept Ablation Fine-Tuning (CAFT) -- a technique that leverages interpretability tools to control how LLMs generalize from fine-tuning.

### Strengths
- Interpretability-guided linear subspace ablation during fine-tuning is simple, training-data-agnostic, and leaves inference unchanged. 
- It showed the utility of the method on three tasks such as emergent misalignment and two multiple choice tasks with spurious correlations.
- Compares PCA vs SAE latents and shows task-dependent advantages

### Weaknesses
- For the evaluation GPT-4.1 has been used as a judge for coherence, misalignment, and code vulnerability, some human evaluation could have been conducted for the validity of the evaluation. 

Typos/Grammatical issues
- L322: in A.7 -> in Section A.7

### Questions
How sensitive are results to the number of directions, layers chosen, and projection strength? Any ablation on "how many" and "which" layers to ablate?

### Soundness
3

### Presentation
3

### Contribution
2
