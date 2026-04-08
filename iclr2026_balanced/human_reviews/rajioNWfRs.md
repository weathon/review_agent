## Human Reviewer 1

### Summary
This paper, proposes TNT, a two-stage training paradigm for deep memory modules in RNNs to address the conflict between training efficiency and inference performance. Stage 1 uses a hierarchical memory (global for long-range context, local with periodic resets for parallelism) and Q-K Projection to boost efficiency. Stage 2 fine-tunes local modules for small chunks. Experiments show up to 17.37× faster training and better accuracy on Titans. Its contributions include identifying three challenges, introducing Q-K Projection, hierarchical memory with resets, efficient fine-tuning, and the TNT paradigm.

### Strengths
1. The paper shows strong originality by proposing the two-stage TNT paradigm, decoupling deep memory modules’ training efficiency and inference performance to break the balancing bottleneck in existing work. Its hierarchical memory (with locally periodic resets) and Q-K Projection solve non-linear module parallelization and memory compression-retrieval mismatch, as targeted innovations.
2. It features rigorous technicality and thorough experiments, defining core mechanisms via clear formulas for reproducibility. Covering multiple baselines and scenarios, with ablation studies verifying key components, its 17.37× training speedup and performance gains are supported by detailed tables/figures, ensuring high reliability. 
3. It has research and application value: breaking deep memory modules’ scalability bottleneck to enable RNNs as Transformer alternatives, identifying three core challenges. Its linear runtime and low fine-tuning overhead fit long-sequence tasks, and it applies to various deep memory modules

### Weaknesses
1. The paper only tests up to 4 local memory modules and does not analyze performance saturation points or optimal chunk size selection for multi-local configurations, leaving gaps in guiding practical hierarchical memory setup.
2. TNT lacks custom kernel optimization, and the speed comparison with optimized Transformer baselines (e.g., Gated Transformer with FlashAttention) is unfair due to hardware optimization mismatch, failing to highlight inherent efficiency advantages.
3. The paper does not validate TNT on sequences longer than 32K or complex long-sequence tasks (e.g., document summarization), limiting demonstration of its practical utility.
4. It fixes global memory chunk size ($C_G=2048$) without testing variations or analyzing the impact of ($C_G$)-($C_L$) ratio, leaving gaps in understanding optimal hierarchical memory coordination.

### Questions
1. Could you clarify if adding more than 4 local modules brings further performance gains or overhead, and is there a method to select optimal chunk sizes for multi-local setups?
2. Have you done preliminary tests on custom kernels for TNT, and would TNT’s speed advantage be more pronounced if Transformer baselines use only JAX (no custom kernels)? 
3. Have you validated TNT on sequences longer than 32K (e.g., 64K, 128K) to check linear scaling, and why not test it on complex long-sequence tasks like document summarization?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper introduces a two-stage training method for test-time memorization modules, which typically suffer from low hardware efficiency during training. In the first stage, a global memory module and several local memory modules are employed to model long-sequence dependencies while maintaining efficiency. The second stage further fine-tunes the memory modules to achieve finer context lengths for the local memories, thereby enhancing inference performance. Experimental results demonstrate the effectiveness and efficiency of the proposed TNT method, showing its potential to achieve performance comparable to standard Transformer models.

### Strengths
- The paper is well structured and easy to follow.
- The proposed two-stage training method effectively balances performance and training efficiency.
- Extensive experiments across different model architectures demonstrate the method’s effectiveness and robustness.

### Weaknesses
- There is no hyperparameter study on $C_G$. How does the global chunk size influence performance?
- TNT only outperforms the FlashAttention method at a 32K sequence length. Is this due to an under-optimized kernel or the limitation of maintaining additional memory modules?
- What are the sizes of the global and local memory modules? Since TNT introduces additional parameters for the global memory module, comparable parameter sizes should be used for Titans when comparing performance.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents TNT, a novel training paradigm designed to address the efficiency–accuracy trade-off in Recurrent Neural Networks (RNNs) with deep test-time memorization modules such as Titans and TTT. Existing parallelization approaches rely on a fixed chunk size, where larger chunks improve training speed but degrade performance, creating a fundamental bottleneck. TNT overcomes this limitation through a two-stage training process. In the first, efficiency-oriented pretraining stage, a hierarchical memory structure is employed: a global module processes large chunks to capture long-range context efficiently, while multiple parallel local modules handle fine-grained details. By periodically resetting local memory states, TNT removes sequential dependencies and enables large-scale context parallelization. In the second, fine-tuning stage, only the local memory modules are adapted to smaller chunk sizes, restoring high-resolution accuracy with minimal computational overhead. Experiments on Titans and TTT models demonstrate that TNT achieves up to 17× faster training while simultaneously improving accuracy, effectively removing a key scalability barrier and establishing RNNs as a promising alternative to Transformers.

### Strengths
The work could bring some advantages: 1) stage 1 in the method can increase training throughput by introducing a novel hierarchical memory architecture that enables unprecedented parallelism. By the clear framework overview figure and TNT Memory Compression Rule, authors provide clear explanation for their method. 2) stage 2 can bridge the gap between the large chunk sizes required for efficient training and the small chunk sizes that yield the best performance at inference. 3) By the experiments given by authors, the results are impressive that can support the main points in the paper.

### Weaknesses
1) The presentation of the paper still needs more improvement. For example, in the main experimental results, I can not understand the meaning of column in table 2.
2) The datasets used by the paper are not clear. I am confused on this.
3) The four baselines seems to be not enough to better support the efficacy of method.
4) I think the table1 can show the effectiveness of method. But other results in paper can not show obviously better performance.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3