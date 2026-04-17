# EGGS-PTP: An Expander-Graph Guided Structured Post-training Pruning Method for Large Language Models

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
As Large Language Models (LLMs) become more widely adopted and scale up in size, the computational and memory challenges involved in deploying these massive foundation models have grown increasingly severe. This underscores the urgent need to develop more efficient model variants. Faced with this challenge, the present work introduces EGGS-PTP: an Expander-Graph Guided Structured Post-training Pruning method. The proposed approach leverages graph theory to guide the design of N:M structured pruning, effectively reducing model size and computational demands. By incorporating concepts from expander graphs, EGGS-PTP ensures information flow within the pruned network, preserving essential model functionality. Extensive numerical experiments demonstrate that EGGS-PTP not only achieves significant acceleration and memory savings due to structured sparsity but also outperforms existing structured pruning techniques in terms of accuracy across various LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes EGGS-PTP, a novel expander-graph-guided parallel training paradigm designed to improve training efficiency and scalability for large neural networks. The key idea is to leverage expander graph connectivity to ensure balanced communication and gradient propagation among model partitions, addressing bottlenecks in distributed deep learning. The authors theoretically analyze convergence under expander connectivity and demonstrate improved training throughput and convergence stability compared to existing methods.

### Strengths
- The paper introduces an interesting perspective by connecting structured sparsity with expander graph theory, aiming to preserve information flow in pruned LLMs.
- The authors evaluate on multiple LLMs and datasets, providing a broad empirical view.
- The paper provides sufficient implementation details for reproduction.

### Weaknesses
- Overstated theoretical claims.
- Insufficient ablation on graph hyperparameters.

### Questions
1. State computational complexity of the diagonal selection ($\mathcal{O}(M^2)$ per block?) to justify scalability.
2. Will different $B$ values affect performance?
3. Add standard deviation or at least average of three runs to show statistical stability.
4. Add a column for sparsity ratio (%) kept weights for fair comparison.
5. Should specify evaluation datasets’ sources (Hellaswag, BoolQ, etc.) and brief descriptions.
6. It is suggested to report actual wall-clock time / GPU hours for pruning each model in “Pruning Overhead” section.
7. Please clarify whether the adjacency is block-diagonal across layers or if cross-layer expansion is modeled.
8. For Lemma 1, the authors claim that each pruned layer is a two-sided expander graph. The provided proof merely ensures that every input node retains at least $B$ connections and each output node maintains $(M-N)$ neighbors, which are only local degree constraints. However, the formal expander definition (Eqs (3)–(4)) requires a global neighborhood expansion property, namely that for any small subset $T \subseteq I$, the ratio $|\Gamma(T)| / |T| > a_I > 1$, and symmetrically for $S \subseteq O$.

### Soundness
3

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
EGGS-PTP is a post training pruning method that emphasizes information flow in N:M pruning. The method builds upon the recent RIA pruning method and is inspired by the idea of expander graphs. Theoretical results explore the expander graph framing and experimental results demonstrate the performance of the method relative to several baselines. The EGGS-PTP method performs favorably across a variety of models and tasks.

### Strengths
The additional diagonal selection leads to improvements over the RIA metric. Results are positive across perplexity and zero-shot task results, across several models.

### Weaknesses
Results are quite close to RIA; confidence intervals would help strengthen the claims of improved performance. 

The theory seems a bit disjointed. Why does it matter that we produce a two-sided expander? It is unclear what contribution this theory adds aside from some inspiration for the method. It would be nice to see either an improved explanation of why the expander graph theory is useful, or some further connections to claims made in the paper. For instance, if this framework improves information flow, are there any experiments that can demonstrate this?

### Questions
What is the pruning cost of EGGS-PTP vs other methods? I see runtime results in Table 5 but what is the complexity of computing the importance score and performing the pruning process?

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
3

### Summary
EGGS-PTP is an expander-graph guided structured pruning method for large language models that balances efficiency and accuracy. It combines importance-aware and connectivity-aware pruning to maintain information flow under N:M sparsity, outperforming prior methods like Wanda and RIA in both perplexity and zero-shot tasks.

### Strengths
1. EGGS-PTP introduces expander-graph theory into post-training pruning, presenting the first framework that applies expander graph concepts to large language model pruning. It innovatively leverages graph-theoretic properties such as connectivity and expansion to maintain robust information flow in pruned models.
2. It combines importance-aware and connectivity-aware pruning to balance compression efficiency and model accuracy.
3. The method enforces N:M structured sparsity compatible with GPU acceleration, achieving up to 1.6× inference speed-up.
4. Experiments show consistent improvements over existing methods such as Wanda and RIA in both perplexity and zero-shot performance across multiple LLaMA and OPT models.

### Weaknesses
1. EGGS-PTP mainly integrates expander-graph theory with existing pruning frameworks rather than introducing a fundamentally new learning mechanism, relying on heuristic rules instead of adaptive structures.
2. The method incurs higher pruning overhead than baselines like RIA, and its scalability beyond 34B-parameter models remains untested.
3. It depends on manual tuning of the hyperparameter (B), limiting automation and generalization across different architectures.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3
