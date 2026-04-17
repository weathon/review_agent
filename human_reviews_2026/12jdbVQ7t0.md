# Lattice: Learning to Efficiently Compress the Memory

- Decision: Reject
- Scores: 6, 4, 4, 4, 8

## Abstract
Attention mechanisms have revolutionized sequence learning but suffer from quadratic computational complexity. This paper introduces Lattice, a novel recurrent neural network (RNN) mechanism that leverages the inherent low-rank structure of K-V matrices to efficiently compress the cache into a fixed number of memory slots, achieving sub-quadratic complexity. 
We formulate this compression as an online optimization problem and derive a dynamic memory update rule based on a single gradient descent step.  The resulting recurrence features a state- and input-dependent gating mechanism, offering an interpretable memory update process. 
The core innovation is the orthogonal update: each memory slot is updated exclusively with information orthogonal to its current state, hence incorporating only novel, non-redundant data, which minimizes the interference with previously stored information. 
The experimental results show that Lattice achieves the best perplexity compared to all baselines across diverse context lengths and model sizes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Lattice, a novel recurrent neural network (RNN) mechanism designed to address the quadratic computational and memory complexity of attention mechanisms in sequence modeling. Lattice leverages the low-rank structure of key-value (K-V) matrices to compress the memory cache into a fixed number of slots, achieving sub-quadratic complexity. The memory update is formulated as an online optimization problem, leading to a dynamic, interpretable, and state/input-dependent update rule. The core innovation is an orthogonal update: each memory slot is updated only with information orthogonal to its current state, thus minimizing redundancy and interference. Experimental results show that Lattice outperforms strong baselines (including Transformer++, Mamba2, DeltaNet, TTT, and others) in perplexity and zero-shot reasoning tasks across various context lengths and model sizes.

### Strengths
* The paper proposes a principled, optimization-based approach to memory compression in sequence models, moving beyond heuristic or ad-hoc memory management.
* The orthogonal update rule is mathematically elegant and well-motivated, with clear connections to Riemannian optimization and online dictionary learning.
* The update mechanism is highly interpretable: each memory slot only absorbs new, non-redundant information, which is a desirable property for long-context modeling.

### Weaknesses
* While ablations are provided, the analysis could be deepened. For example, the impact of the number of memory slots (m), or the sensitivity to the choice of normalization and regularization, could be explored more systematically.
* Although the method is theoretically efficient, the actual training/inference wall-clock time and memory usage compared to baselines (especially on very long sequences) are not deeply analyzed.

### Questions
* How sensitive is Lattice’s performance to the number of memory slots (m)? Is there a clear trade-off between slot count, computational cost, and model expressivity?
* How does the model utilize the memory slots in practice? Are some slots consistently more active than others? Can you visualize or analyze the diversity of information stored in different slots?
* Can you provide a pytorch-like pseudo code to help clarify the algorithm and facilitate implementation?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Lattice, a novel sub-quadratic RNN architecture. The core innovation is the orthogonal memory update for mitigating memory interference with previously stored information by updating each memory slot exclusively with information orthogonal to its current state。The expriments analyze the proposed model on common-sense reasoning tasks and the effectiveness of each proposed component.

### Strengths
1. The idea of ​​orthogonal memory update is interesting and reasonable for alleviating memory conflict problems.
2. It provides a thorough theoretical analysis, starting with online gradient descent, to derive the form of orthogonal memory update, which has an elegant mathematical form and high interpretability.

### Weaknesses
1. The paper only reports common-sense reasoning tasks to evaluate language modeling capabilities, lacking experiments on recall-intensive and long-context tasks (such as MQAR[1] and LongBench[2], etc.), making it difficult to know the model's contextual understanding ability and thus difficult to fully evaluate the effectiveness of the proposed architecture.
2. Lacking open-source code and models, I cannot know the specific details and effectiveness of the proposed architecture implementation, or whether the implemented operators are truly hardware-efficient and parallel-trained.

[1] Simran Arora, et al. Zoology: Measuring and improving recall in efficient language models. ICLR, 2024.
[2] Yushi Bai, et al. Longbench: A bilingual, multitask benchmark for long context understanding. ACL, 2024.

### Questions
If each update of memory is orthogonal to the previous memory, does this mean there will be no interference from historical memories? How can we experimentally verify that this can alleviate memory conflict and thus improve model performance with the same memory/state capacity?

### Soundness
2

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
4

### Summary
See Below

### Strengths
1. comprehensive related work coverage: this paper covered almost all related like Transformer++ / GLA / Mamba2 / DeltaNet / TTT, especially the TTT, but missing RWKV

### Weaknesses
1. Novelty: The paper's core contribution is an incremental modification of TTT-Linear. Both methods use online optimization with reconstruction loss to update memory states. The main difference is that lattice applies normalization to state columns and adds orthogonal projection, while TTT-Linear normalizes the output. The orthogonal projection itself has been standard practice for rnn stabilization since unitary rnn.
2. The experimental results show minimal improvements over existing methods.

### Questions
From theoretical analysis, Lattice should have higher computational cost than Mamba2 because the orthogonal projection requires additional operations for each memory slot, and tensor contractions are more complex than simple outer products. However, the appendix claims Lattice shows "better throughput scalability" without providing concrete measurements. Given that Mamba2 has highly optimized CUDA kernels, the throughput claims appear contradictory.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Lattice, a novel RNN mechanism leveraging the low-rank structure of K-V matrices to compress cache into fixed memory slots, achieving sub-quadratic complexity via online optimization and gradient-based dynamic updates. Its core innovation is orthogonal updates that add only non-redundant information to each memory slot, minimizing interference. Experiments show Lattice outperforms baselines in perplexity across diverse context lengths, with more significant gains as contexts grow longer.

### Strengths
- The work addresses fundamental and forward-looking architectural challenges in sequence models, which are of significant importance to the field. 

- The paper is well-structured, and its theoretical framework is rigorous.

### Weaknesses
- The introduction cites recent non-linear RNN works (e.g., LaCT [1]) but lacks a detailed discussion of the relationship between the proposed method and these existing approaches. Clarifying these connections is crucial for readers to contextualize and understand the paper's novel contributions.
- The authors should explicitly differentiate their proposed state normalization from the Fast-weight normalization used in LaCT [1]. Furthermore, for the orthogonal update mechanism, a comparison (either theoretical or empirical) of the pros and cons of the proposed method against LaCT's Muon would greatly benefit the reader's understanding.
- In the experiments, it is unclear if the chunk_size used is consistent with that in LaCT [1]. LaCT reported significant GPU efficiency issues when chunk_size=1. The authors should clarify if their method shares this limitation. A practical efficiency comparison (e.g., training speed, memory footprint) against baselines like TTT [2], LaCT, and the proposed Lattice method is necessary.

[1] Test-Time Training Done Right.

[2] Learning to (Learn at Test Time):RNNs with Expressive Hidden States

### Questions
- There appears to be a discrepancy regarding the LAMBADA dataset. It is mentioned in the text (line 460), but its performance results are missing from Table 2.
- The experiments are primarily focused on commonsense reasoning. To better demonstrate the potential and generalizability of the proposed Lattice as a foundational LLM architecture, could the authors provide results on a wider range of tasks (e.g., vision, time-series, or biomedical data)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces Lattice, a new architecture that replaces transformer attention with a compressed memory representation. Instead of a growing K-V cache they use a streaming RNN with a fixed number of 'memory slots'. Each new token updates a small set of memory slots by writing only the orthogonal part of its information, enabling efficient streaming with constant memory.

Across many language-modeling benchmark tasks, Lattice matches or achieves lower perplexity than state-of-the-art transformers and modern linear RNNs.

### Strengths
By removing the growing KV cache they successfully avoid a problem with standard transformers at long contexts. 

Results are competitive with state-of-the-art transformers and SSMs.

### Weaknesses
Hard to know if the model will continue to outperform transformers at much larger parameter scales and very long contexts. 

Given that the fixed number of memory slots is so important to this new architecture, it is surprising that the paper does not discuss how the size of this dimension is selected or varied in experiments.

It is not clear from the paper how much Lattice affects training time or whether it increases compute requirements at inference.

minor typo:
1174 "final learning rate of 3e5" -> 3e-5

### Questions
I wasn't entirely sure how you choose the number of memory slots, how do you pick this hyperparameter? How sensitive is performance to your choice of the number of memory slots? 

A table with the choice of number of memory slots and performance would be interesting to see. 

Could you provide information about wall-clock training time in your comparisons?

### Soundness
3

### Presentation
3

### Contribution
3
