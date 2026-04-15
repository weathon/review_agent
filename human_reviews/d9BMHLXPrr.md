# FSW-GNN: A Bi-Lipschitz WL-Equivalent Graph Neural Network

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
Many of the most popular graph neural networks fall into the category of Message Passing Neural Networks (MPNNs). Famously,  MPNNs ability to distinguish between graphs is limited to graphs separable  by the   Weisfeiler-Leman (WL) graph isomorphism test, and the strongest MPNNs, in terms of separation power, are those which are WL-equivalent. 

Recently, it was shown that the quality of separation provided by standard WL-equivalent MPNN can be very low, resulting in WL-separable graphs being mapped to very similar, hardly distinguishable features.

This paper addresses this issue by seeking bi-Lipschitz continuity guarantees for MPNNs. We demonstrate that, in contrast with standard summation-based MPNNs, which lack bi-Lipschitz properties, our proposed model provides a bi-Lipschitz graph embedding with respect to two standard graph metrics. Empirically, we show that our MPNN is competitive with standard MPNNs for several graph learning tasks, and is far more accurate on long-range tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Motivated by recent observations that the quality of separation by common Weisfeiler-Leman (WL) expressive message passing graph neural networks (MPNNs) can be quite low, this work presents an MPNN architecture which is -- in contrast to standard architectural choices -- WL-equivalent. This means that (almost) *all* functions in the function class can distinguish *all* pairs of WL-distinguishable graphs. The new architecture, called FSW-GNN, follows the standard MPNN framework, with the essential difference being that multiset functions used for neighborhood aggregation are replaced by the ~~(almost)~~ injective Fourier-Sliced Wasserstein (FSW) embedding for multisets. Further, Lipschitzness of this architecture is analyzed w.r.t. two WL distances, the tree mover's distance (TMD) and the tree metric. The new architecture is also empirically analyzed on a set of standard benchmarks, specifically long-range tasks, geared at understanding how FSW-GNN alleviates oversmoothing and oversquashing.

### Strengths
- The paper is generally well-motivated and easy to follow.
- Designing an MPNN that, unlike practically all existing architectures, can *simultaneously* differentiate all WL-distinguishable graphs is an intriguing research direction. This approach aligns well with recent lines of work aimed at understanding GNN embeddings and stability.
- I found the generalization of the DS metric to node-featured graphs particularly insightful, which I believe to be an important step towards establishing it further as a WL metric alongside the TMD etc.

### Weaknesses
- While the extension of the DS metric to node-featured graphs is an important step, to me this seems somewhat disconnected from the rest of the paper, considering that the proof of the lower Lipschitzness of FSW-GNN solely relies on the vector norms on the features being $\ell^1$, giving rise to piecewise linear functions.
- In its current stage, to me the contribution seems rather incremental on top of [2], which proposed the FSW multiset embeddings. The fact that using these injective multiset embeddings in a MPNN also yields WL-equivalent MPNNs, which is emphasized as one of the main results of the work, does not come surprising to me, and the brevity of the proof of Theorem 3.2 underlines this.
- In the analysis of the bi-Lipschitzness (Theorem 3.3), the obtained Lipschitz constants are not made explicit. The proof also seems rather generic to me, as it relies only on the set of all graphs up to a certain number of nodes being finite, and the underlying distance functions to be piecewise linear w.r.t. the features. With a similar reasoning, one could also argue that all ReLU MLPs are Lipschitz continuous -- however, the constant could be arbitrarily bad. As such, I can only see limited practical utility in such a statement.
- While the method is evaluated on a range of standard benchmarks, in my opinion the work is missing a careful smaller-scale evaluation, potentially on synthetic data, which validates the theory (i.e., if most instances of FSW-GNN are really WL-equivalent), and empirically investigates the lower Lipschitz constant of the method. I am also unsure about the ability of FSW-GNN to alleviate oversquashing. Here, I would define oversquashing as the difficulty that MPNNs have at exchanging messages over multiple hops by having to squash information from exponentially increasing neighborhoods into a fixed-size vector. I believe that this might only be tested insufficiently by the graph transfer task, as only the information out of a single node is important for the task. I am unsure how FSW-GNN would tackle this differently than a standard MPNN, considering that the underlying graph used for message passing and the vectors for representation of the features stay unchanged.

In my opinion, the scope of the work in its current form, especially considering that the Lipschitz constant of FSW-GNN is not further quantified, is not yet ready for a full conference paper. Nonetheless, I find this a very interesting topic and genuinely enjoyed reading the paper, and with the above weaknesses addressed, I believe that this would be a strong contribution.

[1] Yair Davidson and Nadav Dym. On the Hölder Stability of Multiset and Graph Neural Networks. arXiv preprint arXiv:2406.06984, 2024.

[2] Tal Amir and Nadav Dym. Fourier sliced-Wasserstein Embedding for Multisets and Measures, 2024. URL https://arxiv.org/abs/2405.16519.

### Questions
- Did you attempt to further quantify the Lipschitz constant of Theorem 3.3, and/or conducted experiments to study this?
- There are several typos and minor errors in the script, to the point that reading flow is sometimes a bit interrupted.  I encourage the authors to further proofread the script in order to improve the presentation. Some examples I found include the following:
   - In the definition of lower-Hölder continuity (lines 220-222), "*in expectation*" should either be removed or the correct notion as introduced in [1] should be described.
   - In lines 1044/1045, it might say $\boldsymbol{h}_{\tilde{G}}$ instead.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposed FSW-GNN, which makes use of Fourier Sliced-Wasserstein (FSW) embedding and guarantees the bi-Lipschitz property of the output graph embeddings. Empirically, FSW-GNN is on par with MPNN on standard graph learning tasks, but it achieves superior performance on long-range tasks.

### Strengths
1. Bi-Lipchitzness is an interesting idea to understand the expressive power of MPNNs.

2. It is quite interesting to see that FSW-GNN is particularly strong for long-range tasks, meaning that enhancing Bi-Lipchitzness is particularly effective for long-range tasks, where standard GNNs fall short of.

### Weaknesses
Since bi-Lipschitz continuity is stated for graph embeddings, it is less clear from a theoretical perspective in what way it is connected with node embeddings and node-level tasks. In particular, the long-range tasks considered are node-level tasks. The intuition stated in line 216-236 is also based on the graph level: I can see that "deep GNNs are bad for graph level tasks, and as a result through improving bi-Lipschitz continuity, FSW-GNN makes it less bad for graph level tasks", but it is hard to see why this intuition explains why FSW-GNN makes it less bad for node level tasks.

### Questions
See above.

### Soundness
3

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
This paper introduces FSW-GNN, a novel bi-Lipschitz GNN that maintains stable distance-preserving embeddings through the FSW aggregation method. FSW-GNN outperforms traditional MPNNs on tasks requiring deep message-passing, effectively addressing issues like over smoothing and over squashing in long-range tasks.

### Strengths
1. The paper introduces the FSW-GNN, a novel graph neural network (GNN) model that achieves bi-Lipschitz continuity, which is a significant advancement over traditional MPNNs that lack such properties.
2. Empirical evaluations show that FSW-GNN performs competitively on standard graph tasks and excels in long-range tasks.

### Weaknesses
1. The FSW-GNN requires more complex operations, potentially increasing runtime compared to simpler GNN architectures like GCN and GIN.
2. FSW-GNN’s runtime is considerably higher for large datasets, which might limit its application in highly scalable scenarios.
3. While the paper provides proof for the bi-Lipschitz properties, it relies on empirical evidence and some conjecture to suggest that the model’s stability holds as depth increases.

### Questions
1. TMD requires proper features for every vertex, which can not be available in practical cases, where it fails. Can this method be extended in those cases where partial node features are missing?
2. What are the average node numbers and edge numbers of the dataset that are used in the experiments?
3. How does the FSW-GNN handle graphs with different structures, such as highly dense versus sparse graphs, in terms of both performance and computational efficiency?
4. How critical is the bi-Lipschitz property for real-world applications beyond synthetic tasks?
5. The paper focuses on long-range tasks that benefit from deep message passing. How does FSW-GNN perform on tasks involving smaller graphs or shallow networks, like the MUTAG dataset?
6. How sensitive is FSW-GNN to changes in hyperparameters compared to traditional MPNNs?

### Soundness
3

### Presentation
3

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
The authors of this paper propose FSW-GNN, a new message-passing neural network that achieves bi-Lipschitz continuity while maintaining Weisfeiler-Lehman (WL) equivalence. Unlike typical MPNNs, which struggle with feature separation in Euclidean space, FSW-GNN provides bi-Lipschitz guarantees by leveraging the Fourier Sliced-Wasserstein (FSW) embedding for message aggregation.

### Strengths
- The authors did an exceptionally good job in presenting and structuring the paper, which I enjoyed reading. The way the authors introduce the related work throughout the paper is very nice and detailed, as it gives a clear message to the readers about the contributions and the relations with previous works. 
- FSW-GNN is one of the first GNNs to offer bi-Lipschitz guarantees with respect to two significant WL-metrics: the DS metric and Tree Mover’s Distance. 
- The experimental results are promising, as the proposed method outperforms in various datasets simple GNN baselines.

### Weaknesses
- My main concern with this paper is its close similarity to SortMPNN, as many ideas of the paper, such as the use of sorted message aggregation, are directly inspired by that work. While FSW-GNN introduces bi-Lipschitz guarantees, SortMPNN already established a similar approach to achieving Lipschitz properties in expectation. Furthermore, in the experimental results, SortMPNN actually outperforms FSW-GNN. The authors also did not provide a comparison with SortMPNN on some of the other datasets used. Therefore, besides that FSW-GNN is computationally cheaper than SortMPNN, it is not clear to me why someone should use FSW-GNN instead of SortMPNN. 

- The authors argue that FSW-GNN excels in long-range tasks, however, the reported results in peptides-struct and peptides-func, which are considered long-range datasets, are not that good. How do the authors explain this?

### Questions
- Could the authors clarify the specific advantages of FSW-GNN over SortMPNN beyond computational efficiency? Given the performance gap on some benchmarks, what key factors would make FSW-GNN better than SortMPNN?

- Could the authors provide the experimental results for SortMPNN in the examined datasets?

- Could the authors elaborate on why FSW-GNN did not perform as expected on these long-range datasets, and how this aligns with the claim of improved long-range performance?

- Since the sorting operation is not differentiable, how do the authors handle this in the training process?

### Soundness
3

### Presentation
4

### Contribution
3
