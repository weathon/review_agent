# Hypergraph-Native Message Passing: An Incidence-Centric Perspective

- Decision: Reject
- Scores: 2, 8, 6, 4

## Abstract
While hypergraphs encapsulate higher-order interactions among entities and transcend the pairwise connections characteristic of traditional graphs, their prevailing learning approaches predominantly inherit from graph neural networks, adhering to the established message passing paradigm.
These methods frequently conceptualizes hyperedges as special nodes, facilitating the transmission of aggregated messages through hyperedges instead of direct messages between adjacent nodes.
Such a paradigm is prone to information loss, especially in the context of large hyperedges that bridge a heterophilic array of nodes.
To mitigate this shortcoming and enhance high-order message passing, we propose the Hypergraph-native Message Passing (HMP) framework, which leverages full-rank interactions among the incidences along the underlying hypergraph and its dual.
In contrast to the conventional node-centric approaches, this incidence-centric perspective adeptly manages incidence-level tasks, such as hyperedge-dependent labelling, and seamlessly integrates virtual incidences for both hyperedge- and node-level tasks.
Empirical evaluations demonstrate that HMP achieves a substantial improvement over state-of-the-art methods on 6 hyperedge-dependent labelling benchmarks, with an increase in accuracy ranging from 2.3% to 28.9%, while also delivering competitive results on 13 node classification benchmarks.
Code to reproduce all our experiments is available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Hypergraph-native Message Passing (HMP) framework for learning on hypergraphs. The central motivation is to overcome the "information squashing" problem that affects many existing Hypergraph Neural Networks (HNNs) based on the star expansion. The authors substantiate the framework's efficacy with both theoretical proofs and empirical validation.

### Strengths
1.	The manuscript is well-written. The intuition of the "information squashing" problem is explained clearly, which makes the paper's goals easy to understand.

2.	Experimental validation demonstrates that the proposed method can outperform many baseline methods on the selected benchmarks.

### Weaknesses
1.	My primary concern is regarding the claimed novelty of the "incidence-centric" idea, which appears to have significant overlap with established previous works. Specifically, the core concept strongly resembles the line expansion (LE) approach [1] and, even more so, the co-representation learning (CoNHD) framework [2]. The paper's dismissal of LE for treating the converted graph as "homogeneous" seems superficial, as LE constructs a well-defined graph of (vertex, hyperedge) pairs with a clear structure. Furthermore, HMP shares nearly identical motivations and high-level solutions with CoNHD, which also models interactions as "multi-input multi-output functions" to avoid information loss. The paper's main attempt to distinguish itself, by critiquing CoNHD's specific SetTransformer implementation as less adaptive, focuses on a low-level implementation choice rather than a fundamental difference in the learning paradigm. This positions HMP less as a novel framework and more as an alternative (e.g., self-attention-based) implementation of the CoNHD paradigm, which diminishes the claimed conceptual contribution.

2.	Following the first point, the paper's central claim of achieving "Adaptive Dimensions" as a key advantage over CoNHD is vague and not well-substantiated. The authors contend that their self-attention mechanism is inherently more adaptive to hyperedge size than CoNHD's SetTransformer. However, this argument seems flawed, as standard multi-head self-attention (used in HMP) also aggregates inputs into a fixed-size output representation, with its 'adaptiveness' coming from learned attention weights. This weighting principle is also central to the SetTransformer. It is therefore unclear how HMP offers a fundamentally more adaptive representation capacity, and this core claimed benefit remains unconvincing.

3.	The empirical analysis feels incomplete and would be strengthened by addressing two key omissions. First, while Theorem 2 commendably identifies the high computational complexity per hyperedge, the proposed solution, referencing efficient transformers, is presented without any empirical validation. The paper would be much more convincing if it included experiments on runtime and scalability to demonstrate the practical viability of this approach. Second, the paper reports that HMP underperforms on key homophilic benchmarks (Cora, Citeseer, Pubmed) but then offers no analysis for this important negative result. A detailed discussion is required to explain why the proposed message-passing mechanism struggles in these standard settings, as this insight is crucial for understanding the model's limitations and applicability.

[1] Yang, C., Wang, R., Yao, S., & Abdelzaher, T. (2022, October). Semi-supervised hypergraph node classification on hypergraph line expansion. In Proceedings of the 31st ACM international conference on information & knowledge management (pp. 2352-2361).

[2] Zheng, Y., & Worring, M. (2024). Co-Representation Neural Hypergraph Diffusion for Edge-Dependent Node Classification. arXiv preprint arXiv:2405.14286.

### Questions
1.	Could you clarify the fundamental theoretical difference between HMP's learning paradigm and that of LE and CoNHD, beyond the specific implementation choice?

2.	Given that both multi-head self-attention and SetTransformers aggregate inputs to a fixed-size output, what is the specific mechanism in HMP that enables a fundamentally more "adaptive" representation capacity?

3.	Can you provide the empirical runtime and scalability experiments that validate your claim that efficient transformers are a practical solution to the complexity identified in Theorem 2?

4.	What is your analysis for why the proposed HMP message-passing mechanism underperforms on key homophilic benchmarks like Cora, Citeseer, and Pubmed?

5.	Could you elaborate on how your critique of LE as "homogeneous" holds, given its well-defined bipartite structure of (vertex, hyperedge) pairs?

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
This paper proposes an interesting way of thinking for hypergraph learning.  Instead of the usual star expansion, hypergraph native message passing is introduced to pass messages directly between node hyperedge pairs, to avoid information squashing, which is further shown to be generalisation of prior work like AllSet, with gains on extensive tasks.

### Strengths
The core idea is interesting. The incidence-centric perspective is more natural fit for hypergraphs. The experiment results are promising.

### Weaknesses
The theory says complexity scales with the size of the largest hyperedge, what will happen in practice with a dataset that has massive hyperedges? A short discussion or experiment on this would be more convincing.

### Questions
Could you provide practical guidance on when to use virtual node incidences or hyperedge incidences?

Will the model performance degreade as the hyperedge become extremely large and if there is a point that self attention within a hyperedge become the bottleneck?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Hypergraph-native Message Passing (HMP), a novel framework for learning on hypergraphs. The authors argue that existing Hypergraph Neural Networks (HNNs), which often adapt graph neural network (GNN) paradigms, suffer from information squashing. HMP proposes an incidence-centric perspective, instead of the traditional 'node-hyperedge-node' path.

### Strengths
The paper clearly identifies a key limitation in existing HNNs, i.e., the "information squashing" bottleneck. Doing the same kind of incidence exchange on the dual hypergraph is elegant and makes the framework naturally handle hyperedge-centric signals as well. The synthetic hyperchains are set up specifically to test whether a method can preserve higher-order paths without over-squashing. HMP is indeed the most robust

### Weaknesses
The primary concern is computational cost. While they mention using linear-complexity attention and parallelization, this feels like a partial solution. For hypergraphs with very large hyperedges, this quadratic cost within every hyperedge, every layer, could be prohibitive compared to the simple aggregation in methods like AllSet.

### Questions
Could the authors elaborate on the practical runtime/memory trade-offs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Hypergraph-native Message Passing (HMP) framework, a novel approach for representation learning on hypergraphs. The key motivation is to address the "information squashing" problem prevalent in existing Hypergraph Neural Networks (HNNs), which often rely on a star-expansion paradigm where messages from multiple nodes are aggregated into a single, fixed-size hyperedge representation. This bottleneck is particularly detrimental in large or heterophilic hyperedges. The authors demonstrate both theoretical and empirical results to support the effectiveness of the proposed HMP framework.

### Strengths
- The manuscript is clearly written and logically organized.
- The proposed method demonstrates strong empirical performance, outperforming many baseline methods on the selected benchmarks.
- The paper provides a useful theoretical contribution, offering theorems that prove HMP can be seen as a generalization of AllSet and, by extension, many other HNNs.

### Weaknesses
- **Incremental Novelty:** The paper's claim to propose a "pioneering learning paradigm" is unsubstantiated. The core "incidence-centric" idea is predated by frameworks like Line Expansion (LE) [1], which is not adequately discussed. HMP's novelty over LE appears to be the application of self-attention, not the paradigm itself. Similarly, the paper claims novelty over CoNHD [2] by handling adaptive hyperedge representations, but this overlooks the fact that adaptive representation sizes are already a key contribution of CoNHD. Consequently, the novelty appears incremental.
- **Insufficient Motivation:** The paper is motivated as a solution to the "information squashing" problem of HNNs, but this issue is primarily demonstrated in the context of star-expansion. The paper fails to establish that this problem even exists in more relevant, related works like LE and CoNHD. If these prior methods already resolve information squashing, the paper's core motivation is not strong enough.
- **Lack of Theoretical Analysis:** Given that related works (LE, CoNHD) also appear to mitigate the information squashing problem, the paper must provide a more rigorous theoretical analysis to differentiate its contribution. The authors should formally prove what advantages HMP offers in terms of expressive power or other theoretical properties when compared directly against these existing methods.
- **Insufficient Empirical Analysis:** The paper's empirical analysis is incomplete. It reports that HMP underperforms on key homophilic benchmarks (Cora, Citeseer, Pubmed) but offers no analysis for this important negative result. An analysis is required to explain why HMP's message-passing mechanism fails in these settings. For example, the paper should investigate the trade-off between HMP's flexible attention and the strong, beneficial smoothing bias of the baseline methods that outperform it.

[1] Yang et al., "Semi-supervised hypergraph node classification on hypergraph line expansion,” Proceedings of the 31st ACM international conference on information and knowledge management, 2022.

[2] Zheng and Worring, ”Co-Representation Neural Hypergraph Diffusion for Edge-Dependent Node Classification,” arXiv, 2024.

### Questions
- Could you please clarify the theoretical difference between the proposed 'incidence-centric' paradigm and the Line Expansion (LE) framework [1]?
- Given the similarity to LE, how would you redefine HMP's core novelty beyond the specific application of self-attention?
- Since CoNHD [2] also features an adaptive representation mechanism, could you provide a direct comparison to demonstrate what unique advantages HMP's mechanism offers?
- Can you provide theoretical or empirical evidence that the information squashing problem persists in more advanced frameworks like LE and CoNHD?
- Could you provide a formal analysis of expressive power that theoretically establishes the advantages of HMP's self-attention mechanism over the schemes used in LE and CoNHD?
- Could you provide a detailed analysis for the negative result that HMP underperforms on key homophilic benchmarks like Cora and Citeseer?

[1] Yang et al., "Semi-supervised hypergraph node classification on hypergraph line expansion,” Proceedings of the 31st ACM international conference on information and knowledge management, 2022.

[2] Zheng and Worring, ”Co-Representation Neural Hypergraph Diffusion for Edge-Dependent Node Classification,” arXiv, 2024.

### Soundness
3

### Presentation
3

### Contribution
2
