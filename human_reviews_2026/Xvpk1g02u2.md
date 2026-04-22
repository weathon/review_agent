# QKV Projections Require a Fraction of Their Memory

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
The Multi-Head Attention mechanism is central to LLM operation, and multiple works target its compute and memory efficiency during training.
While most works focus on approximating the scaled dot product, the memory consumption of the linear projections that compute the $Q$, $K$, and $V$ tensors from the input $x$ is often overlooked.
To address this, we propose Point-Approximate Matrix Multiplication (PAMM), a novel tensor compression technique that compresses the activations of the $Q,K,V$ projections in attention layers by a factor of up to $\times 512$, effectively erasing their memory footprint, while achieving similar or better final perplexity. PAMM is fully composable with efficient attention techniques such as FlashAttention, making it a practical and complementary method for memory-efficient LLM training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Point-Approximate Matrix Multiplication (PAMM) to reduce memory consumption of the QKV projections, by a factor of up to 512x, thereby improving memory footprint while achieving similar PPL. PAMM can be composable with other efficient attention techniques such as FlashAttention and LoRA, making it practical and complementary for memory-efficient LLM training.

### Strengths
1. This paper shows that QKV projections are highly redundant along the sequence dimension and thus can be compressed by a factor of >100x for efficient training with almost no accuracy loss, marking a step toward large-size model training on limited hardware devices.
2. The throughput degradation decreases when the model size gets larger, making PAMM more practical for large-size model training on limited hardware devices -- highly reduced memory consumption (hard constraint, must be satisfied) and low throughput degradation (soft constraint, the lower the better).

### Weaknesses
1. The memory consumption of QKV projections is significantly reduced, but that is just the memory consumption of QKV projections and accounts for 20% of peak usage -- there are many other tensors/activations consuming non-negligible memory. Therefore, the overall impact of this work may be limited (by 20%).
2. The throughput degradation is proportional to batch_size * seq_len -- but only the effect of model size on the throughput degradation is discussed in the paper. The proposed method may be (much) less efficient with large batch size or long sequence length -- this should also be addressed.
3. The largest model used in the experiments is LLaMA-1B -- larger model(s) should be considered.

### Questions
My questions and suggestions are basically from "Weaknesses" as aforementioned.
1. From Weakness 1: If my opinion is correct, PAMM can save up to (and at most) 20% of peak memory usage.
2. From Weakness 2: Please discuss the effects of batch size and sequence length on the throughput degradation.
3. From Weakness 3: Please use larger model(s) to demonstrate the consistency of PAMM's efficacy.

### Soundness
3

### Presentation
3

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
PAMM proposes to approximate GEMMs using a sequence of point-wise multiplications. These are obtained by projecting one of the GEMM operands into lines spanned by a small catalog of points randomly sampled from the rows of said operand.  Each point in the activation tensor is represented by its closet point from a generator, under a proposed "neighborhood condition".

### Strengths
- The proposed method identifies the redundancy in the sequence dimension, which is typically enormous in modern LLM training.
- Outperforms other methods like CompAct and Uniform-CRS in memory-performance tradeoffs.

### Weaknesses
- the experiments are limited to small to medium size LLMs. Scaling up models can introduce performance degradation of the approximation method. 
- Activations are dynamic during training. Random sampling is slow, particularly on GPUs. Note that random sampling requires a pseudo-random number generator, such as an LFSR or others. Most algorithms in this family are sequential and model generation from an irreducible in a Galois field. The authors should check the intrinsic details of cudaranddx. Can we just sample once and stick with the choice of catalog points Cjs, or perhaps periodically refresh? 
- The algorithm requires a matmul followed by an argmin on one of the dimensions. This means a reduction over one of the reduction, this is slow on a GPU.
- The breakage of computation into a sequence of small GEMMs can harm tensor core utilization.

### Questions
No questions, but please address the above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the less explored problem of reducing activation memory during LLM's training. The authors propose Point-Approximate Matrix Multiplication (PAMM), a simple technique that compresses the input activations of the Q, K, and V projections in attention layers using a small set of representative rows and scaling factors. This allows approximate gradient computation without storing full activations. Experiments on LLaMA models (60M–1B) and RoBERTa-base show that PAMM reduces activation memory by $512 \times$ with little to no loss—and sometimes improvement—in perplexity and task accuracy. The method introduces only a small training throughput penalty and is compatible with existing efficiency techniques.

### Strengths
* The paper is well written, clearly structured, and easy to follow.
* The proposed method is supported by solid theoretical analysis.
* The approach effectively reduces training-time memory consumption while maintaining, or even slightly improving, model accuracy with minimal degradation.

### Weaknesses
* **Limited profiling on sequence redundancy:** The paper offers initial empirical evidence of sequence‑axis redundancy (Appendix F: PCA clustering; relative error and coverage), but the analysis is confined to a narrow slice (one layer/model/step). A broader study would better ground the motivation.

* **Missing complexity/scaling analysis:** While the runtime breakdown is helpful (Table 2), the paper lacks an explicit complexity analysis of the compression and approximate multiply and a scaling study with respect to batch size, sequence length, and model dimension.

* **Narrow evaluation scope:** Pretraining results are limited to a specific setup, with sequence length 256 and batch size 512. It is unclear how PAMM performs under different training regimes (e.g., smaller batch but longer sequences).

* **End‑to‑end memory not reported:** Memory savings are reported for QKV activations only. End‑to‑end peak memory (including other activations, parameters, and optimizer states) and a memory timeline would better quantify the overall training benefit.

* **Throughput comparability:** Throughput is reported for single‑GPU batches of 16K tokens (Section 4.4), whereas pretraining uses a global batch of 128K tokens (Appendix D). Aligning these settings or clarifying per‑GPU vs. global configurations would strengthen the runtime claims.

### Questions
* Do similar sequence-level redundancies exist in pretrained LLMs during inference. Does the degree of redundancy change over the course of training?

* Could the authors report the total training-time memory consumption (including all activations, parameters, and optimizer states) for both the baseline and the PAMM-compressed models?

* The current experiments use relatively short sequences (256 tokens) and moderate batch sizes. How does PAMM’s computational cost scale as sequence length or batch size increases?

* How sensitive is PAMM to the shape of the training batch—for example, using fewer long sequences versus more short ones with the same total number of tokens? Does this affect model accuracy or compression effectiveness?

* Can PAMM be applied to other activation tensors beyond the QKV projections, such as feed-forward or output-projection layers? How much redundancy do those tensors exhibit in comparison?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes PAMM (Projected Attention Memory Mapping), a memory-efficient training framework for Transformer-based architectures.
The method introduces learnable low-dimensional projection matrices for the Q/K/V representations, allowing the model to compress activations during training while maintaining recoverable gradient information through a projection-reconstruction scheme.
The approach achieves up to 97% memory reduction in attention activation storage with negligible performance loss, and generalizes well across multiple Transformer variants and tasks.

### Strengths
The paper presents a clear motivation and strong practical relevance — training large Transformers is often constrained by activation memory, and this work directly addresses a crucial scalability challenge. The proposed method introduces learnable projection mappings that preserve gradient reconstructability, an elegant and theoretically grounded idea that distinguishes itself from prior techniques such as activation checkpointing or reversible layers. The empirical results are convincing, demonstrating consistent memory reduction with comparable performance across language modeling and vision benchmarks. Importantly, there is no additional computational overhead during inference, as PAMM operates without decompression or re-materialization, making it highly practical for deployment. Theoretical reasoning is also solid — the authors provide analytical evidence that the projection mapping retains full-rank subspace properties, explaining why gradient information is preserved.

### Weaknesses
While the proposed PAMM framework is conceptually sound and practically motivated, the experimental evaluation is somewhat limited. The current experiments are mainly conducted on mid-scale Transformer models, and it remains unclear how the method scales to very large architectures (e.g., >1B parameters) or longer sequence settings. Moreover, the ablation analysis on the projection dimension ratio and per-layer compression strategies is rather sparse—more systematic studies could strengthen the empirical conclusions.  Finally, while results across language and vision tasks are encouraging, broader validation on other modalities or architectures would further demonstrate the generality of the approach.

### Questions
See my Weakness

### Soundness
3

### Presentation
3

### Contribution
3
