# A3 : an Analytical Low-Rank Approximation Framework for Attention

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Large language models have demonstrated remarkable performance; however, their massive parameter counts make deployment highly expensive. Low-rank approximation offers a promising compression solution, yet existing approaches have two main limitations: (1) They focus on minimizing the output error of individual linear layers, without considering the architectural characteristics of Transformers, and (2) they decompose a large weight matrix into two small low-rank matrices. Consequently, these methods often fall short compared to other compression techniques like pruning and quantization, and introduce runtime overhead such as the extra GEMM kernel launches and memory operations for decomposed small matrices. To address these limitations, we propose $A^3$, a post-training low-rank approximation framework. $A^3$ splits a Transformer layer into three functional components, namely $\texttt{QK}$, $\texttt{OV}$, and $\texttt{MLP}$. For each component, $A^3$ rovides an analytical solution that reduces the hidden dimension size inside each component while minimizing the component's functional loss (i.e., error in attention scores, attention outputs, and MLP outputs). This approach directly reduces model sizes, KV cache sizes, and FLOPs without introducing any runtime overheads. In addition, it provides a new narrative in advancing the optimization problem from singular linear layer loss optimization toward improved end-to-end performance. Through extensive experiments, we show that $A^3$ maintains superior performance compared to SoTAs. For example, under the same reduction budget in computation and memory, our low-rank approximated LLaMA 3.1-70B achieves a perplexity of 4.69 on WikiText-2, outperforming the previous SoTA's 7.87 by 3.18. We also show versatile applications of $A^3$,  including its use in KV cache compression, integration with quantization, fine-tuning and mixed-rank assignments for further performance improvements.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes an analytical method for post-training low rank approximation of transformer language models. Experiments are conducted on LLaMA models of various sizes (7B to 70B parameters) with compression ratios of 10% and 20% for most results (with up to 60%).

### Strengths
Making language models compact and efficient is an important topic.

### Weaknesses
There are two major weaknesses: (1) the proposed method is highly incremental, and (2) the practical significance of the results is very limited.

* Method: Generally speaking, a post-training low-rank approximation method for transformer language models falls within the realm of minor engineering or optimization tweaks rather than representing a broadly impactful algorithmic contribution. Unless there is significant algorithmic innovation or strong empirical results, such an approach is insufficiently impactful.

* Experimental results: In the end, there is substantial performance degradation even with only a 10% compression ratio compared to the uncompressed model. As a result, while the method might be relevant for some very specific application scenarios, it does not stand out as a broadly useful or competitive approach.

For these reasons, while the paper may be suitable for an engineering-oriented venue focusing on language model compression, it falls well below the standard expected at general machine learning conferences.

### Questions
The reviewer has no further questions and considers it unlikely that this work will become acceptable after any rebuttal or discussion, given that the limitation lies in the core idea of the proposed method itself.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
A3 is an analytical low-rank approximation (LRA) framework that operates at the functional component level of Transformer layers—QK, OV, and MLP—instead of factorizing individual weight matrices. It provides closed-form reductions for QK and OV, and a CUR-style selection for MLP, with adaptations for GQA and RoPE. Claimed benefits: lower PPL at the same compression ratio, reduced KV-cache and FLOPs, and end-to-end speedups without extra kernel launches.

### Strengths
- **Component-aware objectives.** Clear decomposition into QK/OV/MLP with matched optimization goals; closed-form A3-QK (Thm. 2) and A3-OV (Thm. 3) are principled and easy to implement.
- **No extra GEMMs.** Unlike matrix factorization $W \approx AB$ that adds a matmul, A3 shrinks head dimensions and FFN width, preserving operator count while directly cutting FLOPs and KV-cache. End-to-end speedups are reported (e.g., Fig. 3).
- **Covers GQA and RoPE.** Joint-SVD for GQA and CUR over RoPE frequencies extend applicability beyond vanilla MHA.

### Weaknesses
- **Omission of SVD-LLM v2 in main tables.** The text lists SVD-LLM v2 as a baseline, but Tables 1–2 compare only to SVD-LLM. Please add head-to-head v2 results in the main tables or justify exclusion.
- **Limited compression-ratio coverage.** Core tables report only 10% and 20%. Provide 30–60% for all models to show scaling behavior, not just a single model/figure.
- **Invertibility assumptions.** Theoretical parts assume invertible/positive-definite correlation matrices $R$. Clarify when $R$ can be singular or ill-conditioned in practice (e.g., sample-poor regimes), what regularization is used (SVD pseudo-inverse, ridge, shrinkage), and how this affects accuracy.

### Questions
- **SVD-LLM v2.** You list v2 as a baseline. Why is it absent from Tables 1–2? Please provide direct v2 numbers under the same three-component setting or explain incompatibilities.
- **Separate $W_Q$ and $W_K$ in practice.** Eqs. (9)–(13) conceptually fuse $W_Q$ and $W_K$ during analysis, then yield reduced-dimensional projections. In real stacks that use split $W_Q/W_K$, how do you deploy the reduced head dimension without breaking fused-QK kernels, KV-cache layout, or tensor-parallel shards?

### Soundness
2

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
The paper proposes A3, a post‑training low‑rank approximation framework tailored to Transformer functional components rather than individual linear layers. A3 decomposes each layer into QK (queries-keys), OV (values-output), and MLP, and derives analytical solutions that reduce the shared hidden dimensions within each component: head dims for QK/OV and the MLP intermediate size. For MHA without RoPE, A3-QK and A3-OV have closed-form weighted SVD solutions that minimize pre‑softmax attention score error and per‑head attention output error, respectively. For MLP and for attention with RoPE, A3 uses CUR-style selection heuristics to identify important columns/rows (and RoPE frequency pairs). The method also adapts to GQA via joint SVD across grouped heads. Because it shrinks native dimensions rather than decomposing layers into two factors, A3 reduces FLOPs, KV cache, and parameters without adding runtime kernels. Experiments on LLaMA-2/3.1 (7B–70B), MPT, and Phi show large perplexity and downstream accuracy gains over SVD-LLM and other low-rank baselines at matched compression budgets, along with consistent throughput improvements.

### Strengths
- **Clear component-wise reformulation** (QK, OV, MLP) that aligns local objectives with Transformer functionality, leading to head/intermediate dimension reduction rather than two-factor decompositions.

- **Closed-form weighted SVD solutions** for QK/OV (MHA-NoPE) with principled use of activation autocorrelations; practical GQA joint-SVD extension.

- **Efficiency**: Avoids extra GEMM kernels and reduces KV cache and attention FLOPs; verified throughput gains on GPU.

- **Strong empirical results** across multiple model families and sizes (7B–70B) and tasks, often with large margins over SVD-LLM at 10–20% compression; especially notable PPL improvements on LLaMA-3.1-70B.

- **Sensible ablations** (component-wise, RoPE vs pruning baselines, simplified variants), and demonstrations of compatibility with quantization and mixed-rank allocation.

### Weaknesses
**Novelty & positioning** 
The key step beyond prior activation-aware low-rank methods (e.g., Eq. 8-style weighted SVD) seems to be the component-wise framing; A3-QK feels close to CLOVER but adds activation whitening. To make the novelty pop, I’d love to see fuller, apples-to-apples comparisons to CLOVER and Palu at scale (not just small ablations), plus a short what’s truly new vs. reinterpreted paragraph in the intro.

**Fairness of compression ratio.** 
It’s hard to tell whether reduction budget is matched across compute, memory, and KV cache for each baseline. Please spell out the budget definition and how it maps to SVD-LLM and others (e.g., FLOPs saved, KV footprint, kernel count) so the reader can judge fairness.

**Throughput evidence.**
Throughput is reported for a single model/hardware/batch. Since kernel efficiency can swing with head-dim choices (alignment, vectorization, fused kernels), a brief sensitivity sweep (head dim × batch × sequence length) or a second hardware config would improve support.

**Theory–practice gap.**
The QK objective drops the softmax nonlinearity and OV heads are treated independently; that can decouple local error from global attention error. Even a small diagnostic (e.g., correlation between local objective reduction and end-to-end perplexity) or a bound/intuition section would help reconcile this.

### Questions
- How exactly is the "compression ratio" defined and matched across methods? Do you equalize total FLOPs (including attention), KV cache, and parameters, or parameter count alone?

- For RoPE, how robust is the CUR frequency-pair selection across datasets and sequence lengths? Any analyses showing which frequencies are retained/dropped and their impact on long-context performance?

- What regularization/pseudoinverse strategy is used when autocorrelation matrices are ill-conditioned? Sensitivity to calibration set size/domain?

- Are you able to run more throughput experiments, e.g., with different batch sizes, sequence lengths, attention backends, and head dimensions (including hardware-friendly multiples)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new low-rank approximation method for transformer blocks. In contrast to prior work that minimizes the approximation error of individual linear layers, this method divides the transformer blocks into functional components (attention scores, attention outputs, and MLPs) and derives analytical solutions to approximate each of them. The authors show that such an approximation produces a smaller error and can be implemented more efficiently (with less GEMM operations). It also reduces the KV cache size and might be stacked with quantization to further reduce the model size.

### Strengths
1. **Practical solution.** The proposed method is easy to implement, it improves the KV cache size and the throughput in tokens/sec. Furthermore, it is compatible with common architecture changes such as RoPE and grouped-query attention. The authors evaluate their method on a variety of models and benchmarks.
2. **Significance.** The paper addresses the problem of reducing the size of LLM parameters, which is crucial for deploying these models.
3. **Clarity.** The paper is well-written and describes the proposed method in a clear way.

### Weaknesses
1. **Compression time.** It would be valuable to extend Appendix F with comments on how compression time scales with model dimensions, as well as with the time needed to compress the Llama-3.1-70B model.

2. **Comparison to other compression methods.** While improving low-rank approximation methods might be valuable on its own, the paper doesn't directly compare $A^3$ to state-of-the-art quantization methods and doesn't clarify whether using quantization + $A^3$ produces better results than simply quantizing the model more aggressively (e.g. using less memory for storing outliers in high precision in methods like [1]).

3. **Limited novelty.** It's worth noting that the paper extends the well-known method of minimizing layer's output error and doesn't suggest fundamentally new approaches to low-rank approximation - however, publishing these results is valuable for practitioners.

4. **Typos.** L933: "Sever specs" -> "Server specs"

[1] Dettmers, Tim, et al. "GPT-3.int8(): 8-bit matrix multiplication for transformers at scale." Advances in neural information processing systems 35 (2022): 30318-30332.

### Questions
1. When would you recommend using $A^3$ on top of/instead of existing quantization methods?
2. How does offline compression time scale with model dimensions? How long does it take to compress the Llama-3.1-70B model?

### Soundness
3

### Presentation
4

### Contribution
3
