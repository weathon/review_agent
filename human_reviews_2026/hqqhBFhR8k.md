# INTRA: Interleaved Non-contiguous Token spaRse Attention

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Transformer models have revolutionized generation tasks, but the quadratic complexity of the attention mechanism limits scalability to long input sequences. 
Prior work introduces sparse attention to mitigate this cost, but relies on contiguous memory access patterns that result in wasted computation for their proposed sparsity layouts, limiting practical efficiency. 
In this paper, we propose $\textbf{I}$nterleaved $\textbf{N}$on-contiguous $\textbf{T}$oken spa$\textbf{R}$se $\textbf{A}$ttention (INTRA), a token-wise sparse attention framework that supports flexible sparsity by redesigning memory access patterns. INTRA's loading unit is a single token, and it can load potentially non-contiguous tokens in global memory to contiguous space in shared memory. We formalize this design as the $\textbf{ISPD}$ (Intra Sparse Pattern Design) principle, a general guidance for constructing sparsity layouts that are efficient for GPUs.
INTRA achieves competitive performance on both image and language generation tasks, while accelerating attention by more than $3.3\times$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to speed up attention with sparse grouping patterns. Query and keys are grouped into disjoint groups (attention complexity is quadratic in each group and linear in the number of groups). The method is simple to implement and allows to set a speed/accuracy trade-off by selecting the number of groups. Experimental comparisons with alternative sparse attention methods over text and image models are reported.

### Strengths
1. The proposed sparse attention method is simple: order (into groups) - attend - reorder and seems effective, even if the evaluation is partial (see weakness 1, 2).
2. The paper is well motivated, references to related literature are appropriate.
3. Experiments over image generation and text with long context are presented.

### Weaknesses
1. The reader cannot understand the speed/accuracy tradeoff in the number of groups. All sparse attention models can operate between regular attention (dense, slower, more accurate, i.e. one single group for INTRA) and no attention (maximum sparsity, fastest, less accurate, i.e. <input length> groups for INTRA), the results should be reported as an operating curve speed/accuracy between the two extremes. Sparse attention baselines also have a similar knob (e.g. local window size in CLEAR, or sparsity stripe for XATTN) and their results should be compared on the same plot.
2. The reader does not know if sparse attention is better than picking a smaller model. The speed/accuracy curve should also show regular attention models with varying numbers of parameters (smaller = faster but less accurate). 
3. The presentation of the design of the attention groups is unclear. Could you show how “good” vs “bad” grouping pattern impact accuracy? Or speed? Or are all grouping patterns performing similarly?

### Questions
1. Timings are difficult to understand.
   1. Could you report generation timings for the non-attention part of the network (replacing attention with a dummy operation)?
   2. Are the speed-ups dependent on batch size?
   3. For text generation, could you report separately timings for prompt inference and autoregressive inference? Similarly, what is the network speed without attention?
2. Could you report FIDs results for the flashattn2 in Table 1, the original model in Table 3? Could you compare the FIDs with dense models of small size?
3. Table 5. Why did you drop two test sets from SCROLL? (ContractNLI and Narrative QA).
4. Could you also compare theoretical FLOPs per inference for the different models?
5. Have you considered methods that learn the grouping of query and keys like routing transformers https://aclanthology.org/2021.tacl-1.4/? 
6. You say that methods with dynamic grouping e.g. https://arxiv.org/abs/2203.03937 are more complicated than static grouping but it would be good to report how well they perform on the speed / accuracy curve. Are static methods comparable?

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
3

### Summary
This paper introduces INTRA, a sparse attention kernel. The key innovation is decoupling memory loading from computation, enabling efficient non-contiguous token access while preserving GPU blockwise efficiency. INTRA alternates between two complementary patterns across layers: "Scatter" (which divides tokens into groups attending within-group) and "Gather" (which enables cross-group information exchange). The authors propose the ISPD (Intra Sparse Pattern Design) Principle as a general framework for hardware-efficient sparse pattern design. Experimental results show significant speedups on FLUX.1-dev image generation (66s to 43s for 2K images) and competitive performance on LLaMA-3.1-8B with sequence lengths up to 16K on the SCROLL benchmark.

### Strengths
1. Novel kernel design approach: The decoupling of memory loading from computation is elegantly engineered. 
2. Practical efficiency gains: Demonstrates decent speedups (up to 7.67× on 16K sequences) with minimal quality degradation.
3. Practical fine-tuning approach: Shows that LoRA adaptation is sufficient to adapt pretrained models to INTRA, making the method accessible without prohibitive retraining costs (43 GPU hours for FLUX, 320 for LLaMA).

### Weaknesses
1. Limited theoretical justification for pattern interleaving

The paper relies heavily on intuition from image subsampling (Fig.4) but lacks rigorous theoretical analysis of why Scatter/Gather interleaving preserves global information flow. While the ablation (Table 5a) shows empirical necessity, there's no formal characterization of:
a. How many layers of interleaving are required for full connectivity
b. What properties of the patterns guarantee information propagation
c. Whether certain pattern combinations are provably better than others

2. Inconsistent and concerning performance on Qasper

Table 4 shows INTRA achieving only 36.68 on Qasper vs. 39.94/39.99 for baselines which is a decent gap. The explanation in Appendix G attributes this to "data hunger" and dataset size (2,567 examples), but:
a. Other SCROLL tasks have similar or smaller training sets
b. No ablation investigates whether this is fundamental to the sparse pattern design
c. This raises concerns about generalizability to complex reasoning tasks.

3. Missing critical baselines and comparisons

No comparison with block-sparse methods that also maintain GPU efficiency
The comparison with CLEAR is limited (only on image generation), and CLEAR represents just one prior approach

4. Insufficient analysis of overhead sources

Section 6.3 mentions overhead from modulo operations and uncoalesced memory access but provides no quantitative breakdown:
a. What fraction of runtime is spent on index computation vs. memory access vs. computation?
b. How does this overhead scale with sequence length?
c. Table 2 shows lower speedups for Scatter vs. Gather 0, but no detailed profiling explains why

5. Limited scope of experimental validation

a. Maximum sequence length tested is 32K (Table 2), which is modest by current long-context standards
b. Authors acknowledge inability to test on video due to resources, but this is a key claimed application...
c. No experiments on domains like code generation, retrieval-augmented generation, or other long-context tasks beyond SCROLL
d. The 16K context length claim as "standard" is outdated (many recent models support 100K+)

6. Incomplete formalization of the ISPD Principle
The ISPD Principle (Section 4) is presented as a "general guideline" but:
a. Condition 3 (Block Alignment) uses "approximately divisible" without defining the approximation tolerance
b. The claim that "sparse patterns that have only intra-CQS attention are efficient" is not proven or formally connected to GPU performance
c. The optimization formulation in Appendix D is incomplete (no algorithm provided, just problem statement)

### Questions
Please address some implicit questions in the weakness.

Quality-efficiency trade-off

Tables 1 and 2 show GFLOPS reductions, but can you provide Pareto curves showing the quality-efficiency trade-off as you vary the number of groups or interleaving frequency? Where does INTRA sit relative to baselines?

Statistical significance

Are the differences in Table 3 and 4 statistically significant? Can you provide confidence intervals or statistical tests, especially for close comparisons (e.g., INTRA vs. CLEAR on 1K generation)?

### Soundness
3

### Presentation
4

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
The paper proposes INTRA (Interleaved Non-contiguous Token Sparse Attention), a static, token-level sparse attention kernel that decouples memory loading from attention computation. By enforcing “intra-CQS” (computational query set) attention, the kernel keeps GPU-friendly blockwise access while still supporting non-contiguous token sparsity—avoiding the mixed-block masking overhead common in CLEAR-style block sparsity. To restore global information flow that a single sparse pattern cannot provide, the model interleaves complementary Scatter and Gather patterns across layers. The authors further generalize this into the ISPD Principle, a recipe for designing hardware-efficient sparse patterns. On FLUX.1-dev, INTRA cuts 2K image generation latency 66s → 43s without quality loss (with LoRA self-distillation); on LLaMA-3.1-8B, INTRA attains comparable SCROLL long-context quality up to 16K while giving 5–7× single-attention speedups over FlashAttention 2.

### Strengths
1. **Hardware-aligned idea.** The key insight “only tokens in the same CQS attend to the same keys” is a good GPU sight.

2. **Interleaved patterns actually matter.** Ablation shows Scatter-only or Gather-only collapses quality, while alternating recovers FID/LPIPS/CLIP to dense-like levels.

3. **Cross-modality evidence.** Vision (FLUX.1-dev, 1K/2K) + LLM (SCROLL 16K/32K), showing that method is not overfitted to one workload.

### Weaknesses
1. **Limited LLM task coverage.** Performance on Qasper is clearly worse; authors only hypothesize it’s data-hungry. A small controlled study (more LoRA steps / full FT on one QA set) would make the claim stronger.
2. **Overhead not fully quantified.** Section 6.3 admits modulo + uncoalesced loads hurt speed; paper should give a table: “ideal vs actual” for Scatter/Gather separately.
3. **Static-pattern requirement.** Method still needs finetuning to adapt to the static pattern; cannot be dropped into arbitrary checkpoints like dynamic methods.
4. **Vision eval is narrow.** Only FLUX.1-dev (one heavy diffusion transformer). A 1B/2B open-image model or a video-lite model would help the “general” claim.

### Questions
1. For Qasper: if you double the LoRA rank or increase training steps, does the gap shrink?
2. Can INTRA be combined with head-level specialization (e.g., some heads always Gather, some always Scatter) to reduce interleaving frequency without losing connectivity?
3. You mention token-wise loading “generalizes” FlashAttention. What is the max sparsity irregularity you can handle before shared-memory packing becomes the bottleneck?
4. For video: since you already have 2D Scatter/Gather, can you simply stack a temporal Gather every N layers to get 3D coverage, or does ISPD break because of block-size mismatch?

### Soundness
3

### Presentation
3

### Contribution
3
