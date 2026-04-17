# Rethinking Graph Out-Of-Distribution Generalization: A Learnable Random Walk Perspective

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Out-Of-Distribution (OOD) generalization has gained increasing attentions for machine learning on graphs, as graph neural networks (GNNs) often exhibit performance degradation under distribution shifts. Existing graph OOD methods tend to follow the basic ideas of invariant risk minimization and structural causal models, interpreting the invariant knowledge across datasets under various distribution shifts as graph topology or graph spectrum. However, these interpretations may be inconsistent with real-world scenarios, as neither invariant topology nor spectrum is assured. In this paper, we advocate the learnable random walk (LRW) perspective as the instantiation of invariant knowledge, and propose LRW-OOD to realize graph OOD generalization learning. Instead of employing fixed probability transition matrix (i.e., degree-normalized adjacency matrix), we parameterize the transition matrix with an LRW-sampler and a path encoder. Furthermore, we propose the kernel density estimation (KDE)-based mutual information (MI) loss to generate random walk sequences that adhere to OOD principles.  Extensive experiment demonstrates that our model can effectively enhance graph OOD generalization under various types of distribution shifts and yield a significant accuracy improvement of 3.87\% over state-of-the-art graph OOD generalization baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose LRW-OOD, a framework that replaces the conventional invariant topology or spectrum assumptions with a learnable random walk (LRW) representation to capture invariant knowledge across environments. The LRW encoder parameterizes the random walk transition matrix using a neural sampler and a path encoder, and optimizes it through a kernel density estimation (KDE)-based mutual information (MI) loss, which jointly enforces sufficiency and invariance conditions for OOD robustness.

### Strengths
1. LRW-OOD introduces a learnable transition matrix to address the limitations of existing topology- or spectrum-invariant assumptions.
2. The paper provides theoretical guarantees supporting the proposed method.
3. The experimental setup is extensive, covering synthetic, cross-domain, and temporal distribution shifts.

### Weaknesses
1. The baselines include classic models but omit very recent methods. Without these comparisons, the reported 3.87% improvement may not fully reflect the current state-of-the-art competitiveness.
2. The theoretical sections contain many notations and are difficult to follow without intuitive explanations. Several variables (e.g., m, S, u) are only implicitly defined.
3. Although Theorem 3.4 provides asymptotic complexity analysis, no empirical runtime or memory benchmarks are reported. It remains unclear whether the KDE-based MI computation scales efficiently to large graphs such as ogb-ArXiv.
4. The experimental discussion is relatively brief and lacks deeper interpretation or analysis of the results.
5. Figure 1 does not reflect the advantages of random-walk-based methods over existing Topology- and Spectrum-based methods.

### Questions
1. How sensitive is the KDE-based MI loss to the choice of kernel bandwidth and kernel type? Were these parameters tuned or fixed across datasets?
2. Have the sufficiency and invariance conditions discussed in Theorem 3.1 been empirically verified, for example by tracking mutual information metrics during training?
3. Are there optimization conflicts between MI sufficiency maximization and risk extrapolation minimization, and if so, how are they balanced during training?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LRW-OOD, a model for graph Out-Of-Distribution (OOD) generalization. It replaces traditional fixed transition matrices with a learnable random walk (LRW) sampler and path encoder to capture invariant knowledge across distribution shifts. The model uses a KDE-based mutual information loss to generate random walk sequences aligned with OOD principles. The paper theoretically demonstrates the connection between random walk sequences and graph OOD generalization, achieving an average improvement of 3.87% in experimental performance.

### Strengths
Originality: The idea of using learnable random walk (LRW) sequences to capture invariant features across distribution shifts is innovative and interesting, offering a fresh perspective on graph OOD generalization.

Theoretical Completeness: The paper provides solid theoretical grounding, demonstrating through rigorous proofs that LRW sequences can effectively capture invariant knowledge and align with OOD principles.

Experimental Validation: Extensive experiments on Synthetic Datasets, Cross-domain, and Temporal Evolution Datasets show that the proposed LRW-OOD model is effective in various OOD scenarios, outperforming existing methods.

### Weaknesses
Clarity and Presentation: Some sections could benefit from additional clarification. For example, the introduction's example is somewhat unclear and could be more straightforward. Additionally, the experimental setup on different datasets, while detailed in the appendix, would be clearer if briefly introduced in the main text.

Insufficient Explanation of Random Walk Approach: The paper uses random walk sequences to address the limitations of existing methods, but more detailed explanation or case studies are needed to better clarify how this approach effectively addresses the challenges and to enhance understanding of LRW-OOD's underlying mechanisms.

Limited Task Generalization: The paper primarily evaluates the model on node classification tasks. Further exploration of its performance on other graph-based tasks, such as graph classification, would provide a better understanding of the model's general applicability.

### Questions
Aside from the weaknesses mentioned, a technical question arises: For single-variable probability prediction, KDE can be used as shown in the right-hand side of Equation (4). How is the joint probability distribution for two variables specifically modeled in this case?

Some notations in the methodology section need further clarification. For example, in Equation (5), does V represent variance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of out-of-distribution OOD generalization for graph neural networks, a challenge due to distribution shifts in topology or node features. The authors propose LRW-OOD that reinterprets invariant knowledge as learnable random walk sequences rather than relying on invariant topology or spectral properties. The framework combines an LRW-sampler, a path encoder, and a kernel density estimation-based mutual information loss to enhance invariance and reduce spurious correlations. Experiments on seven benchmark datasets show improvements of about 3–4% over existing baselines under various distribution shifts.

### Strengths
The paper addresses an important and timely problem in graph learning OOD generalization under distribution shifts. The proposed idea of viewing invariance through learnable random walks provides an interesting alternative to existing topology- and spectrum-based paradigms. The method is clearly described, with mathematical formulations and theoretical discussions that enhance readability. Experimental evaluations are good, covering multiple datasets and baseline comparisons.

### Weaknesses
The paper’s conceptual novelty is somewhat limited. The learnable random walk formulation largely reformulates known ideas about adaptive sampling and mutual-information-based regularization, without a fundamentally new theoretical insight. The related work section lacks a full discussion of prior random-walk-based or path-level GNN models, making it unclear how LRW-OOD truly departs from those approaches. Moreover, the experimental baselines, while broad, omit several recent and strong graph OOD methods that incorporate causal intervention or environment simulation. The result analysis is descriptive; the claimed theoretical benefits are not convincingly validated through ablation or controlled tests linking the KDE-based MI loss to invariance improvement.

### Questions
Could the authors elaborate on how the proposed learnable transition matrix differs in effect from existing adaptive adjacency mechanisms? How does the model behave when applied to large-scale graphs where random walk sampling becomes computationally expensive? Thanks.

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
3

### Summary
This paper tackles the critical challenge of out-of-distribution generalization for graph neural networks on node-level tasks. To overcome the limitations of existing methods, such as invariant topology or spectrum, the authors propose a novel framework called LRW-OOD. This framework is designed to extract invariant topology and knowledge patterns, enabling improved OOD generalization for node-level predictions.

### Strengths
1. The shift from seeking invariant topology/spectrum to learning invariant transition probabilities via random walks is interesting.

2. The paper is theoretically rigorous: rather than relying on heuristics, it formally links the learnable random walk framework to the OOD generalization objective.

3. The evaluation is thorough, covering multiple types of distribution shifts and a wide array of nine competitive baselines

### Weaknesses
1. While the proposed random walk method effectively extracts invariant knowledge through feature similarity, it appears less capable of capturing invariant subgraphs. This limitation may reduce its effectiveness under strong topological distribution shifts.

2. The ablation study on LRW uses a vanilla random walk GNN. A more informative comparison would include a variant employing a non-random-walk method, which could more clearly highlight the specific contribution of LRW.

3. The paper mentions a "two-stage training paradigm", but the description in Section 3.2 and Algorithm 1 makes it seem like an end-to-end process where the LRW encoder and the GNN classifier are trained jointly. Authors are expected to clarify it.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
