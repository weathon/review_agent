# Quantifying Mechanistic Gaps in Algorithmic Reasoning via Neural Compilation

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Neural networks can achieve high prediction accuracy on algorithmic reasoning tasks, yet even effective models fail to faithfully replicate ground-truth mechanisms, despite the fact that the training data contains adequate information to learn the underlying algorithms faithfully. 
We refer to this as the \textit{mechanistic gap}, which we analyze by introducing neural compilation for GNNs, which is a novel technique that analytically encodes source algorithms into network parameters, enabling exact computation and direct comparison with conventionally trained models.
Specifically, we analyze graph attention networks (GATv2), because of their high performance on algorithmic reasoning, mathematical similarity to the transformer architecture, and established use in augmenting transformers for NAR.
Our analysis selects algorithms from the CLRS algorithmic reasoning benchmark: BFS, DFS, and Bellman-Ford, which span effective and algorithmically aligned algorithms.
We quantify faithfulness in two ways: external trace predictions, and internal attention mechanism similarity. 
We demonstrate that there are mechanistic gaps even for algorithmically-aligned parallel algorithms like BFS, which achieve near-perfect accuracy but deviate internally from compiled versions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes using neural compilation with a graph program to encode source algorithms directly into network parameters as a form of ground-truth, expected behavior. It then compares these compiled attention patterns with those learned by GATv2 when simulating graph algorithms. Two faithfulness measures are introduced: an external one, which evaluates alignment with algorithmic traces, and an internal one, which compares the expected attention maps derived from the graph program with those learned by the model. The paper shows that even for BFS, where GAT achieves near-perfect accuracy, there remains a noticeable gap between the expected and learned attention matrices, suggesting that high task accuracy does not necessarily imply mechanistic faithfulness.

### Strengths
- The paper tackles an important and timely topic. While neural algorithmic reasoning has achieved strong performance across a range of algorithms and architectures, few works have examined its interpretability. Understanding this dimension is crucial for developing more robust solutions, and could even inform algorithmic discovery.

- The use of neural compilation as a tool for mechanistic interpretability is interesting, and its application to graph neural networks and algorithmic reasoning tasks is novel.

- The paper is generally well-written, with clear explanations and a thorough review of relevant literature and background material.

### Weaknesses
- The paper builds upon the assumption that the attention mechanism defined by the graph program serves as the ground-truth reference for measuring faithfulness. While this provides a basis for comparison, it represents one possible realization of the underlying algorithm rather than a unique solution. This could partly explain why the GNN achieves near-perfect accuracy while exhibiting differences in its learned attention patterns, as it may have discovered an alternative but equally valid internal mechanism. Given that this assumption underlies the core analysis, a discussion of its implications and possible limitations would be needed.

- The analysis is limited to GAT, chosen for its attention mechanism, but this restricts the generality of the findings to other architectures, particularly MPNNs, which are often the most effective for neural algorithmic reasoning tasks.

- It is unclear how the proposed graph program language generalizes beyond the graph algorithms studied, especially since those are inherently parallel. Sequential algorithms such as Prim’s are not evaluated, so the broader applicability of the approach remains uncertain.

- The main result that there is no strong correlation between faithfulness and accuracy is interesting but somewhat limited in actionable insight. It would be valuable if the authors could discuss how these findings might guide the design of more faithful or generalizable NAR models.

- The external faithfulness measure based on comparing algorithmic traces is relatively straightforward. The observation that models perform well early in execution but degrade over time is somewhat expected, as errors naturally accumulate.

- The citation formatting does not follow ICLR style guidelines, which require author last names followed by the publication year. This can potentially impact the number of pages.

### Questions
- What are the different experimental settings in Table 6? More detailed explanations would help interpret the results.

- To what sizes did the test graphs generalize? It would be useful to analyze in-distribution versus out-of-distribution performance separately.

- How does the proposed approach differ from transformer-specific languages such as RASP? The conceptual similarities suggest the distinctions should be made clearer.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In Neural Algorithmic Reasoning (NAR), neural networks (typically Graph Neural Networks) are trained to learn to execute algorithms. This paper studies whether NAR models actually learn execute algorithms by introducing two metrics to measure the faithfulness between what the model learns and the actual algorithm. This is done by creating a "ground truth model", i.e., a NAR model that perfectly performs an algorithm through Neural Compilation techniques. The trained model is then compared to the ground truth in terms of trace prediction accuracy (how well it follows the executing trace) and attention mechanism similarity. The results show that even when NAR models achieve very high downstream accuracy, they do not faithfully learn the real algorithms.

The main contribution of the paper consists in the definition of the two metrics, and the experimental study. Existing Neural Compilation techniques are used to obtain the "ground truth models".

### Strengths
- Using Neural Compilation to understand NAR is a novel and interesting idea. I think it could also lead ot potentially new strategies of training NAR models.
- The paper is well written and easy to follow.

### Weaknesses
- The analysis does not provide any real direction for improving NAR models
- The proposed internal metric is strictly tied to attention, so cannot be easily applied to other architectures
- Unless the Neural Compilation is unique, it is hard to make decisive claims. Could it be that a different compilation methods lead to better results?
- Some methodological choices need more justification (in particular the fact that the number of message-passing steps is set to 1, and that only a single attention head is used) as they differ from what is typically done in NAR

### Questions
- For the internal metric (eq. 25), would it make more sense to compare the rankings of the elements (in terms of attention scores) instead of the actual scores? Maybe a model can put a different distribution, but match the overall ranking of importance.
- I am not familiar with Neural Compilation, but it seems straightforward that the mapping from algorithm to parameters (line 111) is not unique (one way to look at it is simply to consider that permuting the neurons in a layer of an MLP leaves the output unchanged). This means that there can be a potentially very larger number of possible weight combinations that perfectly produce the given algorithm (i.e., a lot of different "compilations" leading to the same correct network). To understand the quality of the proposed metrics I think it is important to ask whether they would indicate perfect faithfulness to all possible weight combinations that perfectly "implement" the algorithm. I think this is true for the external metric, but is it also for the internal one? 
Another aspect would be if two different "ground truth" networks would obtain the same scores, and I think this would not be the case for the internal metric.
I think the authors should add some discussions along these lines.
- It is true that attention offers a comparison that is easy to understand, but at the same time attention is only one component of the architecture, so it could be that the learned solutions might implement the algorithm via different internal mechanisms. Could the authors comment on this please?
- Related to above, it would be useful to quantify how much attention similarity should be expected between two learned models trained from different seeds?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies whether graph-based models for algorithmic reasoning actually implement the intended algorithms internally. The authors focus on GATv2 processors in the CLRS benchmark (BFS, Bellman–Ford, and DFS). They introduce a neural compilation procedure that analytically sets a GATv2’s parameters to execute a given graph algorithm. Faithfulness is then measured in two main ways, external hints, and internal (similarity between the learned attention distributions and those of the compiled reference). Empirically, then they find mechanistic gaps that models with near-perfect BFS accuracy still diverge from the compiled attention patterns, and trace predictions often degrade over time despite high final-answer accuracy.

### Strengths
I think internal faithfulness via attention-trace similarity really complements standard external trace metrics and predicted OOD accuracy, for the NAR community. I also really like the central empirical message of the paper that accuracy $\neq$ faithfulness and believe it is of importance for both NAR and interpretability community. I loved the writing style and clarity of the paper.

### Weaknesses
1. So, the internal metric compares learned attention to one compiled attention trace, but BFS and Bellman–Ford can admit many low-level realisations (e.g., different tie-breaking, temperature/scaling of logits, distributing computation into decoders). The paper works with algorithmic phase space and unique solutions, but does not formalize equivalence classes or prove that the chosen $\alpha*$ is the mechanism to match. The divergence might actually reflect an equally faithful but different mechanism. Please either justify uniqueness (up to known invariances) or report faithfulness against multiple compiled references (e.g., different tie-breakers) and/or invariance-aware distances. 
2. I think external faithfulness mixes hidden-state quality with decoder power. Appendix A highlights that decoders can perform substantial computation, and the main text notes learned models are only good at predicting traces early on, possibly due to under-convergence or decoding (Fig. 3). Without fixing decoder capacity or probing states with a constrained linear probe (e.g 2210.13382 or 2408.14915), the external metric partly measures decoder design, not mechanism. I think using linear probes for core variables (visited, dist, $\pi$) would increase the soundness of the study. For example, If you probe $[visited, dist, \pi]$ directly from hidden states, how do external faithfulness curves change (Fig. 3/10)? 

3. I suspect that there is a high possibility that a single-head constraint may mask faithfulness.
All core results use one attention head because one head is sufficient, but there are evidences from various studies that emergent mechanistic modularity and expressivity might require multiple heads. Please report internal/external faithfulness with more heads (keeping capacity roughly fixed) to test whether additional heads close the gap.

### Questions
0. The questions above in the Weaknesses. 
1. This is a personal concern. since the compiled BFS uses a pre-attention bias, doesn't it harm the length-generalization? The pre-attention bias simplifies compilation but ties size to problem size, breaking length generalization (Sec. 3.2; Table 8). So, my concern is that comparing a compiled model with the bias against learned models trained for length generalization is not apples-to-apples comparison. Can you elaborate of this? 
2. Eq. (24) seems to miss a sum (it is ambiguous). please specify the sum/average over time and variables.
3. In Sec. 3.3, Eq. (18) for the predecessor looks wrong. for Bellman–Ford we expect ($\pi = \arg\min_j (d_j + w_{ij})$) and then copy ($x_j$) from that argmin. Please correct.
4. Upon more reading, I wonder whether you can quantify the "no correlation" claim, by providing a Pearson/Spearman coefficients between final accuracy and both faithfulness metrics for BFS and Bellman–Ford, and the sample size underlying.
5. Can you precisely define Eq. (25): which norm (I think it is l1), over which axes (nodes, edges, time), and what normalization? Did you try KL/JS/EMD? How sensitive are conclusions to temperature scaling of logits? 
6. 2203.15544 and 2505.17190 (two important missing references on expressivity) argue that many DP-style algorithms live natively in the max-plus semiring and that softmax attention’s normalization blurs the underlying polyhedral decision structure. (the later) shows better OOD behavior when the attention kernel respects this tropical geometry. Given that your internal faithfulness compares softmax attention maps to a compiled softmax reference, could your observed mechanistic gap be partly a kernel/geometry mismatch rather than a learning failure? I think an analysis/discussion here would strengthen arguments of the study and create a deep dialogue in the NAR community.

### Soundness
2

### Presentation
4

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
Authors introduce a neural compilation method for compiling algorithms into graph attention networks, and then utilize intermediate attention states of the compiled model as a reference for ideal behavior. They show that mechanistic gaps exist even for algorithms where NN is algorithmically aligned. On top of this authors propose several architecture modifications for target architecture (GATv2)

### Strengths
High accuracy on algorithmic reasoning tasks in distribution doesn't now imply that NN really learned algorithm and will have high OOD performance. Authors develop metrics (internal and external faithfulness) which could help better understand this behaviour. 
On top of this they develop neural compilation method for compiling algorithms into graph attention networks  and improve GATv2

### Weaknesses
- limited number of algorithms considered
- internal attention mechanism similarity seems very interesting but it is not clear that 1-1 match should be expected due to potentially different representation or algorithm. would be good to explore this deeper

### Questions
Potentially there could be many different implementations for the same algorithm (and/or different representations). How do we know that the mechanistic gap found for BFS (internal faithfulness) is not due to this?

### Soundness
3

### Presentation
3

### Contribution
3
