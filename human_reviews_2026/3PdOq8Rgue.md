# MoM: Linear Sequence Modeling with Mixture-of-Memories

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Linear sequence modeling methods, such as linear attention, state space modeling, and linear RNNs, offer significant efficiency improvements by reducing the complexity of training and inference. However, these methods typically compress the entire input sequence into a single fixed-size memory state, which leads to suboptimal performance on recall-intensive tasks. To address this limitation, we introduce a novel architecture called Mixture-of-Memories (MoM). MoM utilizes multiple independent memory states, with a router network directing input tokens to specific memory states. This approach greatly enhances the overall memory capacity while minimizing memory interference. MoM serves as a general framework that can be seamlessly combined with diverse memory update mechanisms across linear models. As a result, MoM performs exceptionally well on recall-intensive tasks, surpassing existing linear sequence modeling techniques. Despite incorporating multiple memory states, the computation of each memory state remains linear in complexity, allowing MoM to retain the linear-complexity advantage during training, while constant-complexity during inference. Our experimental results show that MoM outperforms current linear sequence models on downstream language tasks, particularly recall-intensive tasks, and even achieves performance comparable to Transformer models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Mixture-of-Memories (MoM), a new architecture designed to improve the performance of linear sequence modeling methods such as linear attention, state space models, and linear RNNs. The key idea is to maintain multiple independent memory states instead of a single fixed-size state, with a router network assigning input tokens to specific memory slots. This design aims to enhance memory capacity and reduce interference between token representations, addressing a common limitation in existing linear models when handling recall-intensive tasks. MoM is presented as a general and flexible framework that can be integrated with various linear memory update mechanisms. The authors report that MoM achieves better results than prior linear sequence models across multiple language tasks—particularly those requiring long-term recall—and approaches the performance of Transformer-based architectures, while preserving linear training complexity and constant inference cost.

### Strengths
1.	The paper is well-written, with a clearly motivated method and strong experimental evidence to support its claims.

### Weaknesses
1.	Experiments with larger model sizes could further strengthen the paper.

2.	Some hyperparameter choices—such as the top-k value in the routing process and the predefined number of memory slots—are not thoroughly discussed.

### Questions
1.	How is the optimal predefined number of subsequences/memories determined?

2.	How is the best top-k parameter selected?

3.	The intuition of using different memories to store different tokens is compelling. However, the pre-routing step before the input hidden representation is somewhat unclear. How can the router determine which memory to use without having access to the information stored in each memory, given that routing occurs before the key-value projection and memory update?

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
3

### Summary
This paper introduces Mixture-of-Memories (MoM), an architecture designed to enhance linear sequence models by addressing their fundamental limitation: compressing entire sequences into a single fixed-size memory state. MoM employs multiple independent memory states with a router network that directs input tokens to specific memories using a top-k selection mechanism. The approach is inspired by theta-gamma oscillations in the hippocampus and borrows conceptually from Mixture-of-Experts (MoE) architectures. The authors demonstrate that MoM maintains O(n) training complexity and O(1) inference complexity while significantly improving performance on recall-intensive tasks. Experiments on 380M and 1.3B parameter models show MoM outperforms existing linear sequence models and approaches Transformer performance on recall tasks, though with modest increases in parameter count. The work provides a general framework compatible with diverse memory update mechanisms and includes a hardware-efficient implementation using varlen operations.

### Strengths
1. Well-motivated problem formulation: The paper clearly identifies memory interference and limited capacity as core issues in linear sequence models, providing both intuitive explanations and neuroscience-inspired motivation from hippocampal theta-gamma oscillations.
2. General and flexible framework: MoM's compatibility with various memory update mechanisms (Table 1 lists 11 different methods) demonstrates its generality. This is a significant practical advantage, allowing easy integration with future linear modeling innovations.
3. Comprehensive experimental evaluation: The paper includes extensive experiments across multiple benchmarks (recall-intensive tasks, long-context tasks, commonsense reasoning), multiple model scales (380M, 1.3B parameters), and thorough ablations (memory count, activation ratios, auxiliary loss, shared memory).
4. Strong empirical results on target tasks: MoM achieves substantial improvements on recall-intensive tasks (28.16 vs 24.78 average for Gated DeltaNet at 380M; 36.04 vs 32.30 at 1.3B), effectively narrowing the gap with Transformers (31.70 and 37.31 respectively).
5. Hardware-efficient implementation: Section 3.3 provides a mathematically rigorous description of the varlen-based implementation, demonstrating practical viability. Figure 3 shows clear efficiency advantages over Transformers at long sequences.

### Weaknesses
1. While related work in linear sequence modeling and MoE has been mentioned, it lacks in-depth comparisons and differential analyses with recent methods (such as RWKV-7 and Titans). In particular, Table 1 lists these methods but does not systematically experimentally compare the performance differences of MoM under different memory update mechanisms.
2. Table 5 claims to have discovered "specialization" in different memories, but it is based solely on qualitative observations of an intermediate layer and lacks systematic and statistical significance verification. This "discovery" is more like cherry-picking data than rigorous analysis.
3. The authors claim that the core advantage of MoM is "eliminates memory interference" by separating memory states, but also introducing shared memory to capture global information. Does this contradict the core design? In particular, in the ablation of table 6, there seems to be a lack of settings for "shared memory only" and "neither shared memory nor single memory exists". This lack of ablation experiment is fatal because it directly relates to whether "mixture" is truly necessary.

### Questions
1. Could shared memory become a new bottleneck? If all tokens access shared memory, will its capacity quickly become saturated? Have you analyzed the actual utilization of shared memory across different layers?
2. More details about reproduction：
	a. What is the computational overhead of the router network?
	b. How sensitive is performance to router initialization?
	c. What are actual wall-clock training times compared to baselines?

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
This paper addresses a well-known weakness of linear sequence models (such as linear RNNs and state-space models): their reliance on a single, fixed-size memory state, which leads to memory interference and poor performance on recall-intensive tasks. The authors propose Mixture-of-Memories (MoM), a new architecture that utilizes multiple independent memory states. A top-k routing network directs each input token to a specific subset of these memories for updating. This mechanism aims to increase the model's effective memory capacity and reduce interference by allowing different types of information to be stored in separate states. The final output is computed by querying a weighted mixture of the activated memory states. The paper demonstrates through extensive experiments that MoM significantly outperforms strong linear model baselines on a variety of downstream tasks, particularly those requiring information recall. Notably, the model achieves performance comparable to standard Transformer models while retaining the linear-time training and constant-time inference benefits of linear RNNs.

### Strengths
- The paper targets a critical and widely recognized weakness of linear-time sequence models: their poor performance on recall-intensive tasks due to the bottleneck of a single fixed-size memory state.
- The experimental results are comprehensive and robust. 

- The core idea of using a top-k router to manage multiple, independent RNN memory states is a novel and clever application of sparse activation principles.

### Weaknesses
- The ablation study in Table 6 indicates that the "Shared Memory" component is critical. Removing it causes a large performance drop (e.g., 2.1 points on Recall tasks). This makes it difficult to disentangle the gains from the "Mixture" mechanism (routing to $k$ of $N$ memories) from the gains of simply having a parallel, always-on "Shared Memory" state. The paper's narrative focuses heavily on the top-k mixture, but a large portion of the gains might be coming from this simpler shared component.
- The main results in Table 2 appear to compare models of slightly different sizes. For example, Table 4 shows the "Gated DeltaNet MoM" model has 444M parameters, while its baseline "Gated DeltaNet" is 380M. This ~17% parameter difference complicates the main comparison. While a fairer study with equal activated parameters is commendably included in Appendix G, the primary results table in the main body of the paper should ideally be based on a more strictly controlled comparison.
- The paper repeatedly claims MoM is a "new paradigm" and "fundamentally differs" from MoE (e.g., in Appendix B). This distinction feels overstated. The core mechanism, a top-k router that sparsely activates a subset of modules, which is the defining feature of MoE. Applying this concept to RNN states and key/value projections instead of FFN layers is a novel and useful application, but it does not seem to be a fundamental departure from the MoE paradigm.
-  The analysis of memory specialization in Section 4.2.6 and Table 5 is weak. The paper provides no quantitative methodology for how token categories (e.g., "Basic nouns/verbs" vs. "Proper nouns/scientific terms") were defined or how the "Potential Function" (e.g., "Simplify semantic information") was determined. As presented, these conclusions appear entirely subjective and anecdotal, lacking the empirical rigor demonstrated in the rest of the paper.

### Questions
See my weakness part.

### Soundness
3

### Presentation
2

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
This paper presents Mixture-of-Memories (MoM), a framework for linear sequence modeling that replaces a single fixed memory with multiple independent ones combined through a routing network. Drawing inspiration from both neuroscience (multi-item memory mechanisms) and the mixture-of-experts (MoE) concept, MoM expands memory capacity and reduces interference, making it especially effective for recall-intensive tasks. Experiments on language benchmarks show that MoM delivers clear gains in recall-focused and long-context tasks, often matching or surpassing Transformer baselines in performance.

### Strengths
1. The paper pinpoints the weakness of compressing an entire sequence into a single memory state in linear models and connects this limitation to memory interference.

2. The empirical study is extensive, covering recall-intensive benchmarks, long-context tasks, memory behavior analysis, scaling and ablation studies.

3. The authors provide qualitative insights into memory specialization and routing distributions, enhancing the interpretability of MoM.

4. Implementation details show careful consideration of computational cost, employing Triton kernels for batched memory routing and maintaining O(n) training and O(1) inference efficiency.

### Weaknesses
1. The router formulation specifies a scheme combining softmax and Top-K but lacks crucial implementation details. It does not clarify how ties in Top-K selection are resolved, whether gradients are propagated through the Top-K operation or treated as non-differentiable, or how the learned matrix affects the sparsity of routing and the memory load balance. It also leaves open whether the router is robust to input distribution shifts or token imbalance.

2. Table 5 provides qualitative insights into memory specialization, but the analysis remains descriptive. Quantitative measures would better substantiate claims of interpretability.

3. The theta-gamma memory analogy provides an appealing motivation but lacks a rigorous computational mapping. The biological discussion lack ablation or formal modeling to validate the link between oscillatory cycles and MoM’s memory dynamics.

### Questions
1. Could the authors clarify how gradients are propagated through the Top-K router? 

2. For recall-intensive benchmarks, the source of improvement is unclear. It is not specified whether the gains come from increased memory capacity, larger parameter count, or better routing dynamics. Could a control experiment with a single enlarged memory help isolate these factors?

3. In the hybrid architecture experiments, do the observed gains mainly result from increased depth, or is there measurable synergy between global attention and mixture-based memory modules?

### Soundness
3

### Presentation
3

### Contribution
3
