# Precise Localization of Memories: A Fine-grained Neuron-level Knowledge Editing Technique for LLMs

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Knowledge editing aims to update outdated information in Large Language Models (LLMs). A representative line of study is locate-then-edit methods, which typically employ causal tracing to identify the modules responsible for recalling factual knowledge about entities. However, we find these methods are often sensitive only to changes in the subject entity, leaving them less effective at adapting to changes in relations. This limitation results in poor editing locality, which can lead to the persistence of irrelevant or inaccurate facts, ultimately compromising the reliability of LLMs. We believe this issue arises from the insufficient precision of knowledge localization. To address this, we propose a Fine-grained Neuron-level Knowledge Editing (FiNE) method that enhances editing locality without affecting overall success rates. By precisely identifying and modifying specific neurons within feed-forward networks, FiNE significantly improves knowledge localization and editing. Quantitative experiments demonstrate that FiNE efficiently achieves better overall performance compared to existing techniques, providing new insights into the localization and modification of knowledge within LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a Fine-grained Neuron-level Knowledge Editing (FiNE) method for precise localization and modification of specific knowledge within large language models. By identifying and modifying individual neurons within feed-forward networks, FiNE enhances editing locality and demonstrates improved performance over existing techniques.

### Strengths
1. The FiNE method introduced in this paper enhances editing performance and significantly improves locality control, showing high effectiveness across most key metrics.

2. The paper is well-structured and clearly presented. It includes thorough ablation studies, efficiency evaluations, and detailed comparisons with existing models, offering a comprehensive assessment of FiNE's effectiveness.

### Weaknesses
1. While FiNE shows higher performance than ROME and MEMIT, its approach introduces significantly more complexity. Unlike ROME, which can perform edits after a single causal tracing run, FiNE requires neuron-level localization for each knowledge item to be edited (if I understand correctly). This additional computational step raises questions about whether the performance gains justify the increased complexity, particularly in cases where fast, scalable edits are required.

2. FiNE’s focus on feed-forward layers in transformers as the primary knowledge storage may restrict its adaptability to other architectures and use cases.

### Questions
1. FiNE introduces a unique locate step with neuron-level localization, but is there any substantive difference in the actual edit step compared to methods like ROME?

2. The experimental models are similar in size, around 7 billion parameters. How would FiNE perform on larger or smaller models, and would neuron-level localization retain its efficacy across varying model scales?

3. How does FiNE scale in terms of time efficiency compared to other locate-then-edit methods? Are there noticeable gains or limitations when applying FiNE to time-sensitive applications?

4. What would be the impact on FiNE’s performance if edits were restricted to fixed layers or specific neurons? Could such restrictions improve interpretability or efficiency without sacrificing accuracy?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the problem of knowledge editing in LLMs, i.e., how to update knowledge "in the LLM". To do so, they use a methodology that precisely locates neurons responsible for a piece of specific knowledge and updates only a limited number of weights. This approach allows to efficiently change the belief while limiting the impact of the modification to other aspects of the LLM. The experiments clearly show the advantages of the approach.

### Strengths
S1. The paper is extremely well presented and easy to follow. All the points are clearly explained, with many connections with previous works.

S2. The approach is novel and brings a new perspective on locate-and-edit methods.

S3. The authors ran an extensive analysis of their approach, with many metrics reported (even in the Appendix on more datasets and models). They tested the parameters of their model in various configurations through the ablation study.

### Weaknesses
W1. The authors claim the code will be available, but I do not understand why they did not upload a zip on OpenReview or used an anonymized git.

W2. Some plots are hard to read, particularly when reading the paper in black and white. The scale of plots in Figures 2 and 5 are strange (they should all be the same; it is better if they start at 0).

W3. Fluency disappeared in Table 3. Besides, "Succ." is sometimes called "Edit Succ." Is it the same thing?

### Questions
See above.
Q1. Can you give a small definition of the "over-editing rates" and "unchanging rates"?

Q2. Line 220, is it possible to have several times the same neuron i selected in layer l? If yes, what happens in this case?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This article points out that ineffective localization leads to poor editing locality. In this hypothesis, the author proposes a fine-grained knowledge editing method (FiNE), suggesting that knowledge is stored in different neurons rather than being stored in specific layers. The author demonstrates the accuracy of FiNE's localization by experiments. And ablation studies prove the effectiveness of the editing method.

### Strengths
* The paper is well-motivated: it points out that the existing editing methods lack effectiveness in locating, and then proposes a fine-grained approach to improve the editing accuracy.
* The method is easy to deploy: memory usage and editing time costs indicate that FiNE is more efficient.

### Weaknesses
**Main Weaknesses**

The effectiveness of the FiNE method may not be convincing enough to me.
* *W1*: I suggest the author to conduct some comparative experiments with the KN method: 1) compare the overlap rate of the neurons located by FiNE and KN to investigate whether FiNE is able to locate neurons that KN cannot. 2) use the method in Section 4.2 to edit the neurons located by KN to demonstrate that the neurons located by your method are indeed highly related to knowledge.
* *W2*: Additionally, I suggest that the author compare the overlap rate of the localization results before and after modifying the prompt to demonstrate the robustness of the FiNE localization method. For example, the author could compare the overlap rate between "X's wife is Y" and "X married Y".

**Minor Weaknesses**

* *W3*: Line 206 has $W_uW^l_{out}\in \mathbb{R}^{d_m \times v}$. However, in Appendix A, we have $W_u \in \mathbb{R}^{d_h \times v}$ and $W^l_{out} \in \mathbb{R}^{d_m \times d_h}$. So it might be $W^l_{out}W_u\in \mathbb{R}^{d_m \times v}$.
* *W4*: The description of selecting neurons from Line 211 to Line 215 may be a bit vague. Therefore, I suggest the author to add pseudocode to help understand it.

**Missing References**

* A Survey on Knowledge Editing of Neural Networks. (2023)
* Editing Large Language Models: Problems, Methods, and Opportunities. (2023)
* Knowledge Editing for Large Language Models: A Survey. (2023)

### Questions
**Main Questions**
* *Q1*: There is "The ineffectiveness of localization may lead to overfitting of editing." in Line 49. So my question is: inaccurate localization should lead to underfitting, why does the author say it is overfitting here? For example, for the localization method for ROME and MEMIT, selecting the last subject token can give the model a confidence of about 98% in the new knowledge. If the last token is selected for editing the model, the model can only achieve a confidence of about 20%.
* *Q2*: The existing knowledge editing methods can have a detrimental impact on the performance of the model [1]. Therefore, I am curious whether a more precise positioning method can alleviate the damage to the model's capabilities.

**Minor Questions**
* *Q3*: Can fine-grained FiNE handle more difficult tasks such as multi-hop editing [2]?

$Ref$:

[1] Model Editing Can Hurt General Abilities of Large Language Models. (2024)

[2] Mquake: Assessing knowledge editing in language models via multi-hop questions. (2023)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a Fine-grained Neuron-level Knowledge Editing (FiNE) method for Large Language Models (LLMs). The authors identify limitations in existing locate-then-edit methods, such as poor locality and over-reliance on subject entities, leading to ineffective knowledge updates. FiNE addresses this by targeting individual neurons within feed-forward networks (FFNs) responsible for specific knowledge, thereby enhancing the precision of knowledge localization. Quantitative results demonstrate FiNE's superior performance in terms of edit success, locality, and efficiency compared to state-of-the-art methods like ROME and MEMIT.

### Strengths
S1: The paper presents a clear and well-structured exposition of the motivation and background knowledge, making it easy to understand the context of the proposed method and enhancing the impact of its contributions.

S2: The method greatly reduces the number of modified parameters compared to existing approaches (e.g., ROME, MEMIT), making it faster and more memory-efficient.

S3: The paper focuses on the precise localization of knowledge. The use of extensive benchmarks, ablation studies, and metrics (e.g., locality, fluency) offers a thorough validation of FiNE's effectiveness.

### Weaknesses
W1: My primary concern is about the novelty. The paper draws on previous work for calculating contribution scores in multi-modal LLMs and extends the approach of modifying neuron parameters within the FFNs. The description in the Methodology section is relatively brief. Could you please clarify the main contributions of this paper?

W2: The paper assumes that the FFN layer is the primary location for knowledge storage but lacks discussion on whether components like the self-attention layer also play a significant role in knowledge representation and generation when conducting knowledge editing.

### Questions
Q1: One advantage of FiNE is its smaller number of modified parameters. Could this be a major reason for its significant lead over other methods in the Locality metric?

Q2: In Table 9, under the LLaMA-2 model in the Locality - FA column, it seems that the highest value should be 73.7, achieved by PMET, rather than 72.1?

### Soundness
3

### Presentation
3

### Contribution
2
