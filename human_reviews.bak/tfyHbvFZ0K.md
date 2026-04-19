# Knowledge Localization: Mission Not Accomplished? Enter Query Localization!

- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
Large language models (LLMs) store extensive factual knowledge, but the mechanisms behind how they store and express this knowledge remain unclear.
The Knowledge Neuron (KN) thesis is a prominent theory for explaining these mechanisms. This theory is based on the **Knowledge Localization (KL)** assumption, which suggests that a fact can be localized to a few knowledge storage units, namely knowledge neurons.
 However, this assumption has two limitations: first, it may be too rigid  regarding knowledge storage, and second, it neglects the role of the attention module in  knowledge expression. 
 
In this paper, we first re-examine the KL assumption and demonstrate that its limitations do indeed exist. To address these, we then present two new findings, each targeting one of the limitations: one focusing on knowledge storage and the other on knowledge expression.
We summarize these findings as **Query Localization** assumption and argue that the KL assumption can be viewed as a simplification of the QL assumption. 
Based on QL assumption, we further propose  the Consistency-Aware KN modification method, which improves the performance of knowledge modification,  further validating our new assumption. We conduct 39 sets of experiments, along with additional visualization experiments, to rigorously confirm  our conclusions. Code will be made public soon.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper challenges the Knowledge Localization (KL) assumption, a core concept of how large language models store and retrieve factual knowledge. It reveals that the KL assumption is not universally applicable. The authors also argue that the KL assumption is limited because it only considers the storage aspect of knowledge and overlooks the mechanisms of knowledge selection.

This paper introduces the Query Localization (QL) assumption: (1) For knowledge that does not conform to the KL assumption (KII), the localization of knowledge is tied to the query itself rather than the underlying fact. (2) The attention module plays a crucial role in selecting the appropriate knowledge neurons for answering a query, especially when dealing with KII facts that are distributed across multiple neurons. The paper supports these hypotheses through extensive experiments and proposes a Consistency-Aware KN modification method that leverages the QL assumption to improve the performance of knowledge modification tasks.

### Strengths
**Originality**: This paper demonstrates a high level of originality by identifying flaws in the KN hypothesis. Through extensive experimentation, the authors have shown that these flaws are widespread and have proposed effective solutions to address them.

**Quality**: This paper is of high quality as the authors have conducted numerous experiments using various models, datasets, and knowledge localization methods, ensuring the reliability of the results.

**Clarity**: The paper is logically and structurally clear, providing well-reasoned problem descriptions, metric designs, and detailed explanations of findings for each section.

**Significance**: This paper is significant for advancing research on the interpretability of LLMs. By identifying flaws in the KN hypothesis and proposing improvements, the paper contributes to the broader goal of making LLMs more interpretable and transparent.

### Weaknesses
The paper conducts extensive experiments for each step of problem identification and resolution independently. However, some paragraphs lack smooth transitions, and certain experimental results are omitted in the analysis.

### Questions
(1)In Table 1, the KII ratio for GPT-2 is lower compared to other models. What causes this discrepancy? Is it because GPT-2 has significantly fewer parameters than the other two models?

(2)Table 2 presents metrics such as Reliability, Generalization, and Locality, but not all of these metrics are analyzed in the paper.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper critiques the Knowledge Localization (KL) assumption, which assumes that a piece of factual knowledge can be localized to knowledge neurons (KN) in LLMs. The authors identify two limitations of this assumption, which pertain to both knowledge storage and expression, and reveal that inconsistent knowledge is a common occurrence. To address these, they propose the Query Localization (QL) assumption, which introduces Query-KN Mapping and Dynamic KN Selection. These concepts suggest a more flexible, query-dependent view of knowledge storage and retrieval in LLMs. Using these insights, they introduce a Consistency-Aware Knowledge Neuron modification method to improve the performance of model edits. Experimental results validate the QL assumption and suggest it as a more effective model of factual knowledge in LLMs.

### Strengths
The paper presents a novel critique of the widely adopted KL assumption in LLMs, questioning its validity for all factual knowledge and proposing a more advanced alternative. The QL assumption, with its concepts of Query-KN Mapping and Dynamic KN Selection, offers an innovative framework that shifts from static knowledge representation to a more query-dependent, dynamic view. This fresh perspective is a notable contribution to the field, as it fundamentally rethinks how LLMs store and express knowledge, integrating the role of the attention mechanism for the first time in this context. The proposed QL assumption further addresses fundamental limitations in how knowledge is localized and retrieved within LLMs, which has implications for a wide range of applications, including model interpretability, knowledge modification, and dynamic knowledge editing.

The paper provides rigorous, well-designed experiments across multiple LLM architectures to evaluate the prevalence and implications of Inconsistent Knowledge that does not adhere to the KL assumption. The introduction of Consistency-Aware KN modification is grounded in empirical analysis and validated by statistically significant improvements in model editing performance metrics, adding strong evidence to validate the new assumption.

### Weaknesses
The QL assumption adds the attention module to knowledge retrieval, which could potentially increase computational load. However, the paper does not analyze how this change impacts model efficiency. Including a comparison of computational costs and scalability between KL-based and QL-based methods—such as inference time or memory usage—would clarify the impact of attention-based neuron selection.

### Questions
It would be better if further details, such as interpretability analyses (e.g., visualization), on how the Dynamic KN Selection process works can be provided.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors investigate the mechanisms behind factual knowledge storage and expression in large language models. It re-evaluates the Knowledge Localization (KL) assumption, which posits that specific knowledge neurons can store distinct facts. Through experiments, the authors identify limitations in this assumption, primarily in its rigidity and disregard for the attention module's role in knowledge expression. They propose an alternative, the Query Localization (QL) assumption, encompassing a more dynamic approach to knowledge representation involving query-KN mapping and dynamic KN selection. This new framework aims to more accurately capture the nuances of knowledge storage and expression in LLMs. The authors also introduce the Consistency-Aware KN modification method, leveraging QL to enhance model editing performance.

### Strengths
- The paper presents an innovative approach by challenging a widely accepted notion in LLM research and proposing a refined framework. The introduction of the Query Localization assumption is a creative rethinking of knowledge localization, addressing observed limitations with a novel perspective.

- The authors conduct extensive empirical evaluation, using 39 experimental setups to validate their claims. The paper demonstrates the prevalence of inconsistent knowledge under the KL assumption. This comprehensive experimental approach solidifies the credibility of the findings.

- The paper is well-structured and easy to follow.

- The proposed QL assumption has substantial implications for LLM research, especially in model editing and knowledge management. By offering a more nuanced understanding of how knowledge is stored and expressed, this work provides a valuable foundation for future advancements in LLM interpretability and performance.

### Weaknesses
- The identification of consistent versus inconsistent knowledge relies on specific thresholding techniques, such as Otsu’s threshold. These could introduce variability in findings if altered. A deeper analysis of threshold sensitivity might strengthen the paper's robustness claims.

- Although the QL assumption demonstrates improved performance in model editing, the discussion on how this improvement translates into real-world applications could be expanded. For instance, it remains unclear how this new approach impacts efficiency, especially in scenarios demanding large-scale knowledge modifications.

### Questions
- I double checked throughout the paper, but failed to find the detailed implementation for the results plotted in Figure 1. What instructions are used to prompt these models?

- Sometimes a query might involve multiple, possibly related knowledge (suppose all of them are consistent knowledge). Do they activate their according neurons simultaneously, or is there an additive effect on knowledge neurons?

- How does Consistency-Aware KN Modification affect model behavior? A case study might help.

### Soundness
4

### Presentation
3

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
This paper investigates the widely accepted knowledge localization (KL) assumption within related research fields by conducting a series of experiments. Through statistical and modification-based analyses, the authors reveal inconsistencies in the KL assumption, leading them to propose a revised assumption. They further demonstrate how this new assumption can enhance a modification method for knowledge management.

### Strengths
1. The paper critically examines the KL assumption by employing both statistical and modification-based evidence, providing a thorough analysis of why the assumption does not hold in certain contexts.
2. Building on this analysis, the authors propose a novel assumption and develop a modification method that leverages this refined perspective.
3. A set of experiments are conducted to provide their findings

### Weaknesses
1. The motivation behind challenging the KL assumption could be further elaborated. For example, from an application perspective, it would be beneficial to explain the practical significance of proving the KL assumption's limitations and demonstrating the advantages of the new assumption.
2. The performance gap between the proposed consistency-aware modification method and the two baseline approaches appears to be minimal, which may reduce the impact of the proposed method.
3. The limited number of relations used in the experiments may constrain the generalizability of the findings, making the results less convincing.

### Questions
See the weaknesses part

### Soundness
3

### Presentation
3

### Contribution
2
