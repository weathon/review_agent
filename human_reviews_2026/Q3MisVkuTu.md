# Oversmoothing, "Oversquashing'', Heterophily, Long-Range, and more: Demystifying Common Beliefs in Graph Machine Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 2

## Abstract
After a renaissance phase in which researchers revisited the message-passing paradigm through the lens of deep learning, the graph machine learning community shifted its attention towards a deeper and practical understanding of message-passing's benefits and limitations. In this paper, we notice how the fast pace of progress around the topics of oversmoothing and oversquashing, the homophily-heterophily dichotomy, and long-range tasks, came with the consolidation of commonly accepted beliefs and assumptions -- under the form of universal statements -- that are not always true nor easy to distinguish from each other. We argue that this has led to ambiguities around the investigated problems, preventing researchers from focusing on and addressing precise research questions while causing a good amount of misunderstandings. Our contribution is to make such common beliefs explicit and encourage critical thinking around these topics, refuting universal statements via simple yet formally sufficient counterexamples. The end goal is to clarify conceptual differences, helping researchers address more clearly defined and targeted problems. The hope is to clarify the distinction between the different issues and promote separate but intertwined research directions to address them.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper clarifies recent concepts in the graph machine learning community, such as oversmoothing and heterophily, and raises simple but noteworthy counterexamples.

### Strengths
1. The paper is well motivated as a position paper, covering a wide range of recent trends in graph machine learning.
2. The paper is well structured, clarifying the nine beliefs step by step.

### Weaknesses
1. Section 3.1: The example is too specific—showing a heterophilic graph where a DGN can achieve perfect classification. Most examples in this paper are toy problems that can be solved by simple graph functions (degree, distance). These problems don't require DGNs.
2. Some explanations are insufficient. For example, in Section 3.3, the authors argue that distinguishing the task is what really matters. However, there's no further analysis of which graph tasks should consider heterophily/homophily or distance. The argument relies on only two specific examples (degree, distance). As a position paper that may direct the future of this area, a deeper dive into this question is necessary.
3. Line 358: Is it correct to refer to Figure 2? Also, please explain **message filtering** in Errica et al. (2025) and Figure 4 (middle) more clearly, as readers may not be familiar with it.

### Questions
1. Could you provide real-world examples of interactions between features, structure, and class labels?

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
4

### Summary
The submission under review argues that several “folklore beliefs” in graph ML — oversmoothing, oversquashing, homophily vs heterophily, long-range reasoning — are overstated or misinterpreted. It provides counterexamples, simple constructions, and some empirical plots (e.g., 1–64 layer GNNs on Cora with different scalings). 

12154_Oversmoothing_Oversquash

 The paper also proposes to retire the overloaded word “oversquashing” and instead talk about “computational bottlenecks” vs “topological bottlenecks,” which it claims are distinct.

### Strengths
1. The paper openly challenges sloppy community narratives about depth, heterophily, and “oversquashing,” and reminds readers that degradation in deep GNNs is not the same thing as classic oversmoothing, and that heterophily alone does not doom GNNs. 
2. The paper proposes to split “oversquashing” into two ideas: “computational bottlenecks” (information from exponentially many nodes being crammed into a fixed hidden vector); “topological bottlenecks” (structural choke points / narrow cuts in the graph).

### Weaknesses
1. The primary claims of this paper overlap significantly with established findings in the GNN literature.
* On Over-smoothing: The observation that removing feature transformations and non-linearities can boost performance in moderately deep GNNs has been demonstrated by prior work, such as SGC [1] and a KDD'22 paper [2]. These studies have already established that performance degradation in models of the depth used in this paper (e.g., 64 layers) is often attributable to feature transformation and non-linearity, rather than over-smoothing alone, which typically manifests in much deeper architectures.
* On Heterophily: The challenges of heterophily have been extensively analyzed. A well-known paper [3] provides a comprehensive treatment of this topic. Unfortunately, the current work does not seem to offer new perspectives or insights beyond what was presented in [3].
------
2. Given that the paper addresses topics with a rich history of research, a deeper investigation from novel perspectives or a rigorous theoretical analysis would be expected. However, the current analysis lacks the necessary depth.
* The investigation into over-smoothing is confined to a few simple settings on a single dataset (Cora). This limited experimental setup is not convincing enough to reveal generalizable patterns or provide robust evidence for the claims made.
* The discussion on heterophily relies on reusing a toy example from [3] without introducing new conceptual or empirical contributions. A more compelling argument would require novel experiments or theoretical formulations.

------
3. About the statement ‘if different classes imply different feature distributions, why would one need a DGN rather than a simple MLP?’

I also disagree with the way that’s phrased in the submission.
Even if different classes have different feature distributions, classification boundaries may still be ambiguous (e.g., overlapping Gaussians, noisy features, etc.). A plain MLP on node features alone may misclassify nodes in that overlapping region.
A message-passing layer can still add value by smoothing / denoising via neighbors: even if node i’s own features are ambiguous, aggregating its neighbors’ features can push it toward the correct side of the decision boundary. This is exactly the classic semi-supervised “feature smoothing” or “label propagation” intuition behind GCNs.

So “why would you ever need a GNN if features are already somewhat class-dependent?” is a false dichotomy. You still may want graph convolution to regularize local decision boundaries and reduce local noise — especially when each node alone is borderline but its neighborhood is consistent.

----
4. Difference between “computational bottleneck” and “topological bottleneck” is unclear. As a core concept to support your claims, it is not wise to put it to appendix. The main text should be self-contained.

### Questions
See weaknesses.

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
4

### Summary
The paper analyzes nine common beliefs in Deep Graph Networks concerning issues such as over-smoothing, over-squashing, homophily/heterophily, and long-range tasks. It systematically elucidates their origins, identifies misconceptions through experimental validation and counterexamples, and aims to correct erroneous understandings, thereby contributing to the advancement of the graph learning field.

### Strengths
S1. The article systematically examines prevalent misconceptions and imprecise definitions within the field of DGN. It offers a reflective analysis aimed at clarifying commonly misunderstood concepts such as over-smoothing and over-squeezing, thereby facilitating researchers' comprehension and preventing misinterpretations. This critical review encourages ongoing reflection on entrenched perspectives, fostering a more rigorous and accurate understanding in the domain.

S2. The paper refines the understanding of the "over-squashing" problem, elucidating the relationships among over-smoothing, over-squashing, homophily/heterophily, and long-range tasks. By redefining these issues and their key aspects, the discussion aims to promote advancements in the development of DGN methodologies.

### Weaknesses
W1. In refuting the claim that OSM is a property of all DGNs, the article relies solely on a simple experiment, which lacks rigorous proof and suffers from limited experimental evidence. The experiment, on one hand, uses only the Cora dataset, thereby failing to establish the universality of the conclusion; on the other hand, certain inferences merely suggest that the DE and RQ metrics, which reflect smoothing, are insufficiently convincing, yet they do not demonstrate that OSM is inevitably unavoidable. It is recommended that the authors provide more rigorous theoretical justification or more comprehensive experimental evidence when discussing the underlying causes of OSM.

W2. The definitions of "computational bottleneck" and "topological bottleneck" rely primarily on intuitive explanations, lacking a unified mathematical framework or quantifiable metrics.

### Questions
Q1. The paper suggests that research should focus on node separability rather than OSM metrics. Could you provide a novel, quantifiable "node separability metric" as an alternative?

Q2. Refining the tasks of distinguishing "computational bottlenecks" from "topological bottlenecks" is meaningful. I hope the authors can further discuss whether there are actual examples of such distinctions in real-world graph data, whether these distinctions are feasible, and how to differentiate them: is it possible to propose theoretically distinguishable metrics?

Q3. Do the conclusions drawn in the paper remain valid in the context of Transformer-based GNNs or Graph Diffusion Models?

Q4. The paper highlights that certain common misconceptions lack universality. Could the authors further elaborate on the scope of applicability for traditional beliefs? For example, in Section 3.1, the counterexample "if a node is at a distance greater than five from a specific node" relates solely to the topological properties of the graph, independent of homophily or heterophily. Under what circumstances is it meaningful to discuss homophily and heterophily?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper revisits and critiques nine widely discussed beliefs in the field of Graph Machine Learning (GML), covering issues such as over-smoothing (OSM), over-squashing (OSQ), homophily/heterophily, and long-range dependency. Through a series of counterexamples, the authors reveal common misconceptions, for instance, that over-smoothing is not inevitable; it depends on model architecture, hyperparameters, and evaluation metrics, and bears no direct causal relation to performance. The paper aims to clarify these misunderstandings and calls for a more rigorous research in graph learning.

### Strengths
1. The paper’s central motivation—to critically examine and challenge long-held opinions and assumptions in the GML community—is very interesting and valuable.

2. The writing is clear and well-structured, using intuitive examples and accessible language to convey complex ideas effectively.

### Weaknesses
1. Lack of justification for the representativeness of rebutted claims. The paper does not convincingly establish whether the criticized viewpoints truly represent mainstream consensus in the field. For instance, is the claim that “all GNNs inevitably suffer from over-smoothing” genuinely a widely accepted belief, or is it only mentioned in a limited subset of works? To make the critique compelling, the authors should first demonstrate the prevalence and influence of these viewpoints in prior literature.

2. Insufficient theoretical and empirical validation. The arguments rely heavily on qualitative reasoning and illustrative examples, with very limited mathematical formalism or empirical validation. I am not convinced. To strengthen credibility, the paper should incorporate formal theoretical derivations, quantitative experiments, or counterfactual analyses that can substantiate its claims.

3. The manuscript attempts to address too many issues simultaneously, resulting in a lack of depth and weakened persuasiveness. I strongly recommend focusing on one or two truly dominant misconceptions and providing a rigorous, well-supported, and empirically validated analysis of these specific points. For example, the classic paper “Adversarial examples are not bugs, they are features” offers a good example: it concentrates on a single misconception and dismantles it thoroughly through both theoretical modeling and empirical evidence.

4. Several arguments (e.g., regarding differences between homophilic and heterophilic graphs) have already been discussed—explicitly or implicitly—in prior works. For example, the development of heterogeneous graph neural networks (HGNNs) stems from recognizing that aggregation mechanisms in homophilic graphs are inadequate for heterophilic structures.

### Questions
Please see the weakness.

### Soundness
1

### Presentation
2

### Contribution
3
