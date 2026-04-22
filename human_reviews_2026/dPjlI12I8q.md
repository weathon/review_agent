# Is Random Attention Sufficient for Sequence Modeling?

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
The transformer architecture is central to the success of modern Large Language Models (LLMs), in part due to its surprising ability to perform a wide range of tasks -- including mathematical reasoning, memorization, and retrieval -- using only gradient-based learning on next-token prediction. While the core component of a transformer is the self-attention mechanism, we question how much, and which aspects, of the performance gains can be attributed to it. To this end, we compare standard transformers to variants in which either the attention weights or the MLP layers are frozen at initialization. Surprisingly, we find that attention with frozen key and query weights is not only able to form induction heads, but can also perform competitively on language modeling. We formalize this by proving a new expressivity result for transformer models with frozen attention weights. To further isolate the contribution of attention, we design MixiT -- the Mixing Transformer -- an architecture variant with entirely random attention scores, with provably stable signal propagation that overcomes prior depth-wise scaling challenges in random transformers. We use the successes and failures of our spectrum of models to pinpoint the role each main transformer component plays. Our results suggest that the transformer architecture has a built-in inductive bias towards in-context reasoning, as it can form specialized circuits even without learnable attention weights.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This article investigates the model performance under fixed random QK, random attention score, and random MLP, and proves that some tasks do not require learnable QK, and some tasks do not even require learnable attention score.

### Strengths
1. The result is very interesting. The author conducted sufficient experiments to verify the impact of different degrees of weakening on sequence modeling. Even tasks such as induction heads can still be learned well when fixing Wq and Wk. 

2. The author proposed a randomized attention score method and theoretically demonstrated its stability. In addition, the author also proved the universal approximation under random matrices.

### Weaknesses
1. 
The presentation of MixiT is confusing.  

From the experimental results, it can be seen that the performance of MixiT is not very good, especially in tasks related to language modeling. And only frozen-QK is well. So the "random attention" in your title is refers to frozen-QK only? I think if we could more clearly distinguish between random attention weight (frozen-QK) and random attention matrix (MixiT) in this article, it would be more conducive to reading.  Besides, the appearance of Theorem 2.1 in the seciton is also very abrupt, It should perhaps be placed in the section 5 where it will be used.

### Questions
1. 
The boundary between what tasks Mixit can and cannot do well is very empirical. Do you have any further elaboration on this？On the other hand, I find that Mixit seems to be not able to do very well on long context tasks, such as induction heads and language modeling, Is that so? 

2. 
The results of Frozen-mlp are missing in Table 2 and Table 3.

### Soundness
3

### Presentation
2

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
The paper
- finds that Frozen-QK with random attention weights can perform competitively with the standard transformer on language modeling tasks. It's expressiveness is also enough for a wide class of sequence-level functions.
- proposes MixiT, proves its training stability. It achieves performance comparable to fully trained tf and Frozen-QK expect induction heads tasks and language modeling.

### Strengths
The paper did abundant experiments and derived theory to support the papers claim, which makes the paper sound.

### Weaknesses
W1: Your main claim is that the performance separation between input-dependent and input-independent attention is largely driven by the latter's inability to form induction heads. However, there are some papers [1], [2] explaining the mechanistic of induction heads theoretically which I think is missing from your paper. In their papers, they all have data-dependent attention for the second layer and it must contribute to the proof. What is the contribution of your paper to the (theoretical) understanding of the mechanics?

W2: You proved Frozen-QK has enough expressivity, but it does not answer the question how can Frozen-QK form the induction head. The former doesn't guarantee the latter because the latter question is about optimization.

[1] Nichani, E., Damian, A., & Lee, J. D. (2024). How transformers learn causal structure with gradient descent. arXiv preprint arXiv:2402.14735.

[2] Chen, S., Sheen, H., Wang, T., & Yang, Z. (2024). Unveiling induction heads: Provable training dynamics and feature learning in transformers. Advances in Neural Information Processing Systems, 37, 66479-66567.

### Questions
Q1: I didn't understand your third contribution and didn't find the part of the paper it correlated to. 

Also see the weaknesses above.

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
The work studies what aspects of transformer performance rely on the self-attention mechanism by comparing the transformer to variants with frozen attention weights or MLP layers. They find that the transformer with frozen QK layers can achieve competitive performance on the induction head and language modeling tasks. The authors also propose the Mixing Transformer (MixiT) which uses random attention and overcomes prior depthwise scaling challenges of random transformers.

### Strengths
- The writing and methodology are clear and easy to follow.
- The experiments cover a diverse set of tasks that highlight which components are most important.
- The finding that the Frozen-QK model can form induction-head-like behavior and achieve strong performance on retrieval and k-hop tasks is particularly interesting.

### Weaknesses
- The discussion of MLPs being important for knowledge storage mostly confirms prior findings and does not extend existing insights.
- The motivation and practical usefulness of the MixiT architecture are not entirely clear. The paper emphasizes that the MixiT architecture provides stable signal propagation as compared to a random transformer, but still underperforms on the induction heads task and on language modeling.
- For the language modeling experiments the sequence length is 256 which is quite short. Do results hold for longer sequence length?

### Questions
- How sensitive are the Frozen-QK and MixiT models to the specific random initialization used?
- How do the authors see MixiT being used beyond this study -- as a diagnostic model or are there some implications for architecture design?

### Soundness
3

### Presentation
3

### Contribution
3
