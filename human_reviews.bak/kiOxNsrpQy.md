# Reconsidering Faithfulness in Regular, Self-Explainable and Domain Invariant GNNs

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
As Graph Neural Networks (GNNs) become more pervasive, it becomes paramount to build reliable tools for explaining their predictions.
A core desideratum is that explanations are *faithful*, i.e., that they portray an accurate picture of the GNN's reasoning process.
However, a number of different faithfulness metrics exist, begging the question of what is faithfulness exactly and how to achieve it.
We make three key contributions.
We begin by showing that *existing metrics are not interchangeable* -- i.e., explanations attaining high faithfulness according to one metric may be unfaithful according to others -- and can *systematically ignore important properties of explanations*.
We proceed to show that, surprisingly, *optimizing for faithfulness is not always a sensible design goal*.  Specifically, we prove that for injective regular GNN architectures, perfectly faithful explanations are completely uninformative.
This does not apply to modular GNNs, such as self-explainable and domain-invariant architectures, prompting us to study the relationship between architectural choices and faithfulness.
Finally, we show that *faithfulness is tightly linked to out-of-distribution generalization*, in that simply ensuring that a GNN can correctly recognize the domain-invariant subgraph, as prescribed by the literature, does not guarantee that it is invariant unless this subgraph is also faithful.
All our code can be found in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors study the faithfulness of explanations generated concerning Graph Neural Networks (GNNs). Specifically, they claim the following main contributions:

1. They claim that current faithfulness metrics are *not interchangeable* and cannot evaluate the explanation quality precisely. These differences can cause the explanations to ignore important elements.
2. They prove that perfectly faithful explanations may be completely uninformative in injective GNNs.
3. They explore the connections between faithfulness and domain invariance.

### Strengths
1. GNN explanations' faithfulness is crucial in real-world applications and academic research.

2. The experiments on four popular datasets including Motif2 show an empirical evaluation of the proposed method.

### Weaknesses
> W1. Concerning contribution 1:

They claim that current faithfulness metrics are **not interchangeable** and cannot evaluate the explanation quality precisely. In other words, the faithfulness performance of an explanation can vary concerning different metrics (Proposition 1 and Sec. 3.1).

However, "faithfulness" naturally can be evaluated by various metrics, and it is **unnecessary** that the evaluation results should be consistent with each other. 

Take the ranking similarity metrics as an example: top-k intersection (precision at k), cosine similarity, NDCG, and other metrics can lead to different results. E.g., ranking A is more similar to ranking B by top-k, while ranking C is more similar to ranking B by cosine similarity, and the divergence among metric results can vary as large as possible. Similar cases can be found in prediction performance cases.


> W2: Concerning contribution 2:

They prove that perfectly faithful explanations may be completely uninformative in injective GNNs (Prop. 4).

However, the details are missing and hard to follow. Specifically, which faithfulness metric is adopted, and what does "match" mean in "$p_r$ matches $N_L(u)$" and "$p_R$ matches $G_A$"?


> W3. Concerning the target tasks:

Are the graph- or node-level classification tasks considered? Definition 1 will be strict for graph-level tasks, especially when the "output" is the distributions rather than categories.

> W4. Concerning the experiments:

The purpose of some experimental results is unclear. 

One of the contributions is the new faithfulness metric (as well as Nec and Suf), while their superiority is not explored by further experiments. 

Instead, in Table 3 and Table 4, the authors evaluate GSAT (and other backbone models) with/without augmentations by Acc and faithfulness. These models may achieve better or worse performance while their connections to the paper's key contribution are unclear. Besides, the relationship between Acc and faithfulness is underexplored.

In other words, the paper proposes to "reconsider" the faithfulness metric, while comprehensive experiments of comparing metrics are missing.

> W5. Concerning the writing:

W5a. Most figures and tables are really far away from their corresponding statements, requiring to jump over the pages repeatedly.

W5b. Some key statements are missing, e.g., the choice of d in Lines 279, and what does "value" mean in the y-axis? 

W5c. Are the graph- or node-level classification tasks considered? Or do different propositions/statements fit different tasks?

### Questions
Please address the concerns and questions in the *weakness* part.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the faithfulness of explanations in modular GNNs. It addresses several problems in explanation metrics, and claims through analysis: 1) not all metrics are the same; 2) faithfulness metrics are not interchangeable; not all metrics are equally reliable; 3) for regular injective GNNs, strict faithful explanations are trivial. Finally, it claims faithfulness is key to OOD generalization. Extensive experiments justify the claims in the paper.

### Strengths
1. The paper show that a) existing metrics are not interchangeable. b) optimizing for faithfulness is not always a sensible design goal. c). faithfulness is tightly linked to out-of-distribution generalization. The novelty is good.
2. The paper present an in-dept analysis of existing graph XAI metrics.
3. The experiments are solid.

### Weaknesses
1. After reading the paper, I am in question of what metrics are suggested for a GNN explanation task. Considering a normal setting, a post-hoc explainer for a trained to-be-explained GNN, how to choose the metric? Can you draw a clear conclusion throughout the analysis in the paper? Therefore, I question the significance of the paper. To improve the paper, I highly suggest the authors include a section summarizing key takeaways and practical guidance on selecting appropriate faithfulness metrics for various GNN explanation settings, tailored to different types of models (e.g., modular GNNs).
2. In section5, how to determine if the explanation is not strictly sufficient? It would be beneficial to improve the paper by providing a concrete example or procedure for determining if an explanation fails to be strictly sufficient.

### Questions
See the weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Faithfulness has been one of the widely used metrics in explainable artificial intelligence research. Intuitively, an explanation is said to be faithful if it accurately reflects the true reasoning process of the underlying model. However, the investigation of faithfulness with respect to graph neural networks has been limited to empirical analysis of generated explanations and proposing new metrics. In this work, the authors highlight the difference between existing faithfulness metrics and argue about the trade-off between them, i.e., existing faithfulness metrics are at odds with each other. Further, the authors show that optimizing for faithfulness is not ideal and leads to a trade-off between the expressiveness of the model and usefulness of faithful explanations.

### Strengths
1. The paper is well-written and the motivation is very clear.

2. The authors detail the importance of the faithfulness metric in graph explanations, highlighting the pitfalls of diverse metrics and the trade-offs between them.

3. The paper presents an interesting analysis between regular, self-explainable, and domain-invariant graph neural networks.

### Weaknesses
1. The relation between faithfulness and out-of-distribution performance is unclear. The statement "faithfulness is tightly linked to out-of-distribution generalization" is weak and doesn't highlight the exact implication of faithfulness on OoD performance. While the authors show the analysis in Section 5, it would be beneficial to the readers if the authors could provide a brief explanation describing this relation in simple terms.

2. The paper lacks a clear description and implication of the theoretical analysis. Formally, each theorem statement is followed by a proof sketch, which describes the main techniques used to derive the lower/upper bound and the practical implications of the bounds with respect to the GNN, explanation, and faithfulness. Addressing this will make the theoretical analysis of the paper much stronger.

### Questions
Please look at the weaknesses for open questions.

### Soundness
3

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
2

### Summary
This paper addresses a timely and impactful topic by exploring the limitations of current metrics used to evaluate the faithfulness of explanations in graph neural networks (GNNs). The authors make three primary contributions: they highlight that existing metrics are not interchangeable, challenge the assumption that faithfulness should always be a design goal, and link faithfulness to out-of-distribution generalization.

### Strengths
This paper addresses a timely and impactful topic by exploring the limitations of current metrics used to evaluate the faithfulness of explanations in GNNs. The paper presents a range of experiments to support these claims, adding depth and rigor to the discussion.

### Weaknesses
1. Clarity in Introduction: The logical flow between sentences in the introduction could be strengthened. For example, the connection between lines 35 and 36 feels disjointed, affecting readability and coherence.

2. Writing Clarity: There are instances of unclear terminology and abbreviations. For instance, the abbreviation “cf.” on line 46 may be confusing. Additionally, on line 77, there’s a mixture of notation, specifically "|G|" and "∥G∥", which could be standardized for consistency.

3. Lack of Proposed Solutions: While the paper highlights significant limitations in existing metrics, it would be more impactful if it suggested potential solutions or proposed a new metric that addresses these limitations. Presenting an alternative would strengthen the contributions and provide practical guidance for future research.

### Questions
See in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
