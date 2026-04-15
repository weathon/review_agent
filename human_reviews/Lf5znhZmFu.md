# Computational Limits of Low-Rank Adaptation (LoRA) Fine-Tuning for Transformer Models

- Decision: Accept (Poster)
- Scores: 6, 3, 6, 8

## Abstract
We study the computational limits of Low-Rank Adaptation (LoRA) for finetuning transformer-based models using fine-grained complexity theory.
Our key observation is that the existence of low-rank decompositions within the gradient computation of LoRA adaptation leads to possible algorithmic speedup.
This allows us to (i) identify a phase transition behavior of efficiency \blue{assuming the Strong Exponential Time Hypothesis (SETH)}, and (ii) prove the existence of almost linear algorithms by controlling the LoRA update computation term by term.
For the former, we identify a sharp transition in the efficiency of all possible rank-$r$ LoRA update algorithms for transformers, based on specific norms resulting from the multiplications of the input sequence $X$, pretrained weights ${W^\star}$, and adapter matrices $\alpha B A/r$.
Specifically, we derive a shared upper bound threshold for such norms and show that efficient (sub-quadratic) approximation algorithms of LoRA exist only below this threshold.
For the latter, we prove the existence of almost linear approximation algorithms for LoRA adaptation by utilizing the hierarchical low-rank structures of LoRA gradients and approximating the gradients with a series of chained low-rank approximations.
To showcase our theory, we consider two practical scenarios: partial (e.g., only $W_V$ and $W_Q$) and full adaptations (e.g., $W_Q$, $W_V$, and $W_K$) of weights in attention heads.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work shows the existence of almost linear approximation algorithms for LoRA on transformer-based models. This paper also proves a phase transition behavior in the efficiency of LoRA. A detailed proof sketch is provided in the paper to support the results.

### Strengths
1. As a theoretical work, this work is well-written and not difficult to follow.
2. The proof idea makes sense and is solid.

### Weaknesses
1 There are two notations that may not be necessary. I suggest considering whether it is possible to remove or simplify some definitions to derive all the lemmas and theorems in the main body.
2 The practical insights of this paper are not very clear. It is better to highlight the practical significance of the theoretical analysis in the paper.

### Questions
1 In lemmas 3.3, 3.4, and 3.5, the amount of time needed to construct the matrices is included. What is the algorithm used for construction here? Why is discussing the required amount of time meaningful since in practice these parameters are learned by gradient methods instead of any construction?
2 Why do you put the analysis of full LoRA in the appendix instead of in the main body? 
3 What are SAT algorithms in line 142?
4 Is it $\frac{\partial L}{\partial A_\mu}$ rather than $\frac{\partial L}{\partial A_Q}$ in line 97?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper examines the computational constraints inherent in current LoRA algorithms, focusing particularly on the O(L^2) computational complexity encountered when updating attention blocks. The authors aim to establish a unified upper-bound threshold for these norms, demonstrating that efficient approximation algorithms for LoRA can indeed operate below this threshold. Consequently, they provide proof of the existence of a nearly linear approximation algorithm, advancing the understanding of computational efficiency within the LoRA framework.

### Strengths
1. Exploring the computational limits of parameter-efficient fine-tuning (PEFT) algorithms is a timely and relevant area of study.
2. By utilizing the tensor vectorization tricks, the authors prove the existence of nearly linear approximation algorithms for LoRA adaptation.
Notably, the authors also establish necessary conditions that could inspire the development of more efficient adaptation methods.These conditions are critical for future research aimed at accelerating the approximation process.

### Weaknesses
1.	Could the authors clarify why Equation 1.2 holds? The expression on the right-hand side appears to minimize the discrepancy between the attention output and the labels Y. Do we only consider 1-layer attention here?
2.	The Strong Exponential Time Hypothesis currently seems to serve only as a counterexample in the context of gradient approximation. Its relevance to the subsequent analysis is unclear in its present form. The reviewer suggests incorporating a more precise and directly relevant statement to clarify its role in the argument.
3.	While purely theoretical contributions may not always necessitate empirical validation, this paper's objective—improving the efficiency of optimizing large language models (LLMs) with LoRA—suggests that experimental results are essential for substantiating its claims. Specifically, an empirical evaluation could verify whether the bounded gradient approximation indeed holds in practice, as this is critical for the practical applicability of the proposed methods.
4.	The authors assert that 'the existence of low-rank decompositions leads to potential algorithmic speedup.' Could full parameter updating also yield similar benefits? Additionally, how the speedup is related to the rank r?
5.	An alternative approach might involve updating the feed-forward network (FFN) layer rather than the attention block, for example [1][2]. Could this adjustment also avoid/alleviate the O(L^2) computational complexity issue in the paper?
6.	The current title may not fully align with the paper’s objectives, as it does not primarily address computational drawbacks but rather focuses on **solving**  these computational limitations.

[1] AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition. 
[2] Parameter-Efficient Fine-Tuning with Controls.

### Questions
See the questions in "weakness".

### Soundness
2

### Presentation
2

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
This work investigates the computational limits of Low-Rank Adaptation (LoRA) for finetuning transformer models through fine-grained complexity analysis. The central insight is that the low-rank structure within LoRA's gradient computations can enable algorithmic speedup. Two main contributions are highlighted:

1. **Efficiency Phase Transition**: The study identifies a sharp efficiency threshold for rank-\(r\) LoRA update algorithms based on norms derived from the interactions between input sequences, pretrained weights, and adapter matrices. Efficient (sub-quadratic) algorithms are possible only when these norms fall below a specified threshold.

2. **Nearly Linear Algorithms**: By leveraging hierarchical low-rank structures in LoRA gradients, the authors construct nearly linear approximation algorithms for LoRA adaptations, assuming the Strong Exponential Time Hypothesis (SETH).

To validate the theoretical findings, the authors explore partial and full weight adaptation scenarios within transformer attention heads, focusing on weights like $W_V$, $W_Q$, and $W_K$.

### Strengths
This paper tackles a highly relevant and timely topic: Low-Rank Adaptation (LoRA). LoRA has gained widespread popularity in practice for its effectiveness in fine-tuning large models efficiently. Despite its practical success, there has been a notable gap in the theoretical understanding of LoRA, making this study’s contributions especially valuable to the field. Developing a rigorous theoretical foundation for LoRA will not only solidify its current applications but also open doors for future research and refinement of the technique.

The abstract and introduction are clear, engaging, and well-crafted. They effectively set the stage for the paper, with a strong motivation that highlights both the practical importance of LoRA and the need for a deeper theoretical exploration. The authors have done a commendable job in outlining the primary contributions, making it easy for readers to understand the key takeaways from the study. The literature review is thorough and provides an excellent context for where this work fits within the broader landscape of model adaptation techniques.

The inclusion of a paragraph on paper organization is appreciated, as it offers a helpful roadmap for readers. It ensures that the structure of the paper is transparent from the outset, allowing readers to follow the flow of ideas with ease.

The authors have also set a high standard for notation and technical formalism. The clarity and consistency of notation enhance readability.

While the technical results appear sound, I should note that, as someone not fully specialized in this type of analysis, I may not have caught every nuance. However, the reasoning seems robust, and the proofs are presented in a way that suggests a careful, thorough approach.

Finally, the conclusion is well-done and reinforces the main points. It synthesizes the findings effectively and reflects on their implications, offering insights into how these results might shape future research in model adaptation. Overall, this paper makes a significant contribution to bridging the gap between LoRA’s practical success and its theoretical understanding, providing a strong foundation for ongoing exploration in the field.

### Weaknesses
This paper currently feels dense and challenging to navigate, as it primarily consists of a series of definitions, lemmas, and theorems, often presented without sufficient explanation, clarification, or intuitive context. For readers who are not already experts in this area, this can make it difficult to grasp the key concepts and results. 

There are several opportunities to improve accessibility and readability. Some of the definitions would be more appropriately placed in an appendix, as the main text is quite heavy with formal definitions, lemmas, and theorems. Moving certain definitions to an appendix could help streamline the main text, allowing readers to focus on the core arguments without getting overwhelmed by technical details.

In addition, a few of the lemmas lack clear statements, which can lead to confusion. For instance, in Lemma 1.1, the term \(L\) is introduced without any accompanying explanation. It would be helpful to review the formulations to ensure clarity, even if the technical details are correct. Improving the precision in how terms are introduced will aid readers in following the logical flow more naturally, without having to reread sections to understand each step.

Given the extensive number of definitions, it would be beneficial to assign descriptive names to them. Naming each definition provides a quick reference, helping readers keep track of terms and concepts as they reappear later in the paper. Otherwise, it’s easy to lose track of which definition corresponds to which concept, especially in a highly technical document.

Sections 3 and 4, which seem intended to present the main results, read more like collections of lemmas, theorems, and technical details without adequate discussion or contextualization. This is a significant area for improvement. The results would be far more impactful if they were accompanied by clear explanations and discussions. Theoretical contributions are valuable only if they can be understood and appreciated, and in their current form, the key insights may be difficult for readers to discern.

Finally, the paper would benefit from including plots, diagrams, or simple illustrations that clarify the results. While extensive experimental results may not be expected in a theoretical paper, even a few toy examples or visual aids could significantly enhance reader comprehension and provide concrete illustrations of the theoretical findings.

Overall, with some adjustments to the structure, added explanations, and a few visual aids, this paper could become far more accessible and impactful, allowing a broader audience to appreciate the significance of the work.

While the technical content appears sound, I am unable to recommend the paper for acceptance in its current form due to significant organizational issues outlined above. Addressing these concerns would greatly improve the paper’s clarity and accessibility.

### Questions
Could you please provide additional explanation of the theoretical results? 

Additionally, would it be possible to include some toy or controlled experiments, or other illustrations to help clarify the findings?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies how to make large Transformer models more computationally efficient during fine-tuning. The main contributions are:
* Identifying a critical point of efficiency where models can be fine-tuned with less computation if they are below this threshold.
* Proposing a new method that can complete model fine-tuning in almost linear time, which is much faster than traditional methods.

### Strengths
The paper introduces a novel theoretical analysis on Low-Rank Adaptation (LoRA) for Transformer models, marking an innovative contribution to the fields of natural language processing and machine learning. It approaches the problem of LoRA adaptation from a fresh perspective, focusing on computational limits and efficiency, which is particularly novel in the context of large foundation models. Additionally, the paper presents an innovative method by proposing an almost linear-time algorithm for LoRA adaptation, which is a significant advancement over existing methods that typically have quadratic complexity.

### Weaknesses
* The paper's primary focus seems to be on theoretical analysis. To strengthen the claims, experimental validation with real-world datasets would be beneficial. Specifically, demonstrating the practical efficiency of the proposed algorithms on standard benchmarks could provide actionable insights into their performance.
* It would be valuable to see how the proposed methods compare to current state-of-the-art techniques in terms of both efficiency and accuracy. This comparison could highlight the advantages and potential limitations of the new algorithms.models and datasets?

### Questions
* Can the authors discuss the potential impact of LoRA adaptation on model generalization? 
* How does the efficiency of the proposed methods scale with larger models and datasets？

### Soundness
4

### Presentation
4

### Contribution
3
