# FastEdit: Low-Rank Structured Regularization for Efficient Model Editing

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4

## Abstract
When new knowledge emerges, it is crucial to efficiently update large language models (LLMs) to reflect the latest information. However, state-of-the-art methods widely adopted in the model editing community—such as MEMIT, EMMET, and AlphaEdit—suffer from prohibitively slow editing speeds, often taking over 15 hours to sequentially edit 5,000 facts on models like LLaMA-3-8B, making real-time updates impractical, especially as model scale increases. Moreover, they require extensive pre-computation to sample pre-edit knowledge—a step that can take over 24 hours—severely limiting their deployability. In this paper, we present \textbf{FastEdit}, a framework that leverages the intrinsic low-rank structure of FFN key spaces not only for speed but also for more effective editing. FastEdit regularizes only the low-rank primary semantic subspace—where most pre-edit knowledge resides—while leaving the remaining directions in the key space unregularized and freely editable. This design channels edits into the unregularized subspace, thereby better preserving pre-trained knowledge in the primary semantic subspace, and enables fast computation via the Sherman–Morrison–Woodbury identity. On LLaMA-3-8B, FastEdit completes 5,000 sequential edits within 4 hours and consistently achieves higher editing accuracy and stability. Moreover, it requires only a small number of pre-edit samples, drastically reducing preprocessing overhead. Our work shows that low-rank structure provides a principled way to balance editability, efficiency, and knowledge preservation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes **FastEdit**, a model-editing framework that introduces a **low-rank plus diagonal (LR+D) structured regularization** for efficient and stable knowledge updates in large language models (LLMs).  
By leveraging the **Sherman–Morrison–Woodbury (SMW) identity**, the method reduces the cubic-time matrix inversion in existing editing frameworks (e.g., MEMIT, PRUNE, AlphaEdit) to a low-rank \(O(dr^2)\) computation.  
A periodic spectral compression strategy is further introduced for sequential edits to maintain bounded rank and computational cost.  
Experiments on GPT2-XL, GPT-J, and LLaMA-3 show up to **10× faster editing** with comparable factual accuracy.

### Strengths
The paper addresses an important bottleneck — the inefficiency of model editing — and offers a structured, implementable solution. FastEdit achieves order-of-magnitude acceleration (5×–10×) and memory reduction (17GB vs. 22GB) across three large models without harming edit precision.The use of the LR+D covariance model and the Sherman–Morrison–Woodbury identity is mathematically sound and clearly derived. The closed-form update (Eq.7) is clean and easy to reproduce. The writing is clear, and the method is straightforward to implement, making it useful for practitioners seeking faster model editing.

### Weaknesses
1. **The claimed acceleration may be overstated.**  

   The reported “10× speedup” considers only the matrix inversion phase but not the **entire editing pipeline**, including the computation of  \(v\) and the LR+D covariance estimation step. When these additional costs are included, the overall acceleration is likely to be much smaller.  


2. **The observed improvement mainly stems from SMW algebraic simplification.**  
   The acceleration is primarily achieved through the **Sherman–Morrison–Woodbury (SMW)** identity, a standard algebraic transformation widely used in low-rank approximation and online inverse computations.  
   Hence, the performance gain should be interpreted as an **engineering optimization**, rather than a genuine algorithmic innovation. 


3. **The safety metrics (eₜ, sₜ) are not novel.**  
   Similar orthogonality-based interference measures were already discussed in previous works.  
   The metrics in FastEdit share nearly identical definitions, differing mostly in naming and visualization.  

4. **Compression advantage is marginal in batch editing scenarios.**  
   The claimed benefit of periodic spectral compression mainly appears in **sequential single editing**, where many edits are applied one-by-one.  
   However, in **batch editing** (e.g., editing 100 facts simultaneously as in AlphaEdit), both MEMIT and FastEdit may exhibit similar runtime reductions, suggesting that compression contributes little to efficiency in such cases.

### Questions
See Above

### Soundness
4

### Presentation
2

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
The paper presents FastEdit, a method to fix the speed bottleneck in model editing. State-of-the-art methods are slow due to large $O(d^3)$ matrix inversions. The authors' key insight is to assume the pre-edit knowledge representation follows a Low-Rank plus Diagonal (LR+D) structure. This assumption allows them to replace the standard regularizer with a structured one that can be inverted efficiently using the Sherman-Morrison-Woodbury (SMW) identity, dropping the computational complexity to $O(dr^2)$. They also use SVD-based compression for sequential editing and a fused covariance estimate to cut down on pre-computation time. Experiments show FastEdit is about 10x faster than baselines on LLaMA-8B and achieves comparable or better editing performance.

### Strengths
1. The paper tackles a critical, practical bottleneck. Current editing times are a major blocker for real-world use. A 10x speedup is a significant engineering contribution that makes real-time editing much more feasible.

2. The core technical idea is elegant. Using the expected LR+D structured regularizer to enable the SMW identity is a smart and principled way to achieve the speedup, moving beyond just brute-force computation.

3. The fused covariance estimation is another important practical win. The 24+ hour pre-computation of prior work is a huge hidden cost, and reducing it to minutes by combining a data-driven estimate with a structural prior is a big step for deployability.

4. The safety analysis in Section 5.3, with the $e_t$ and $s_t$ metrics, provides a nice geometric intuition for why some sequential editing methods fail over time, adding a good diagnostic tool to the paper.

### Weaknesses
1. The novelty feels a bit thin. This seems less like a new framework and more like applying a standard low-rank approximation to the covariance matrix in the MEMIT objective. Using SMW for this is a classic linear algebra trick.
    
2. The sequential editing comparison looks like a strawman. As described in Appendix C.2 the baselines are adapted to accumulate all past keys making their matrix inversion rank grow linearly. This guarantees they will be slow.
    
3. The periodic SVD compression for sequential editing is a lossy heuristic. There's no analysis of its impact on catastrophic forgetting. For instance does the model forget edit #1 after 2000 edits?
    
4. The results are a clear speed-accuracy trade-off. FastEdit is competitive but it doesn't win on accuracy. It underperforms PMET on Llama-3 Generality for example.

### Questions
1. Regarding Appendix C.2 am I understanding correctly that you made the baselines accumulate all past keys into the matrix for inversion? This seems to create an unfair comparison. Why not compare against a standard incremental SMW update for the baselines as well?
    
2. Did you test for catastrophic forgetting caused by your periodic SVD compression? I'm curious what the accuracy on the _first_ 100 edits is after all 2000 edits are finished.
    
3. How did you choose the crucial $r_0$ rank for the Llama-3 and GPT experiments? The appendix only shows a sensitivity analysis for GPT2-XL but this parameter seems central to the method's performance.

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
4

### Summary
This paper presents FastEdit, a fast and structured model editing framework for large language models. FastEdit exploits the empirical low-rank structure of FFN key representations within Transformers, introducing a regularization scheme that enables efficient closed-form updates via the Sherman-Morrison-Woodbury (SMW) identity. The method achieves a dramatic reduction in both computational and memory costs without sacrificing edit precision or generalization. Experimental results across several benchmarks and models (GPT2-XL, GPT-J, Llama-3) show that FastEdit supports rapid, scalable sequential editing while maintaining model robustness and editing safety.

### Strengths
- A central strength is FastEdit's dramatic acceleration of large-scale editing—Figure 1 highlights orders-of-magnitude time reduction for performing 2,000 sequential edits, reducing editing latency from many hours to under two for state-of-the-art LLMs. This is a genuine practical advance addressing a severe real-world bottleneck in model editing.
- Table 1 and Table 2 deliver comprehensive comparisons on CounterFact and ZsRE. FastEdit matches or outperforms strong baselines (e.g., MEMIT, AlphaEdit, PRUNE) on editing efficacy, specificity, and generalization—in many cases closing the efficiency gap without loss of edit success. The wider applicability is supported by experiments on three model architectures.
- The adaptation of all baselines for sequential editing is clearly described, supporting meaningful comparison and reproducibility.

### Weaknesses
- The SMW-based approach relies on good estimation of the LR+D structure from a (now-small) sample of pre-edit keys. While Section 4 and Appendix A provide justification for the low-rank model, the main paper does not thoroughly analyze what happens if the pre-edit data is excessively sparse/noisy, or if the singular value spectrum is not strongly decaying—the potential for bias or regularizer miscalibration is not deeply probed.
- Some of the notation describing periodic spectral compression and SVD-based rank truncation in Algorithm 1 (Appendix B) are terse and could be more explicitly connected to the mathematical formulation in the main sections. For instance, the reuse of symbols for compressed key matrices may confuse practitioners unfamiliar with the area.

### Questions
Same as Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets the practical efficiency bottlenecks of model editing in LLMs. It proposes FastEdit, which exploits an inherent low-rank-plus-diagonal (LR+D) structure in the edit update to accelerate computation. Specifically, the regularization term is rewritten using a structural prior $U U^\top + D$, which enables efficient inverses via the Sherman–Morrison–Woodbury identity. For sequential edits, the method maintains a low computational cost by periodically compressing the accumulated keys to keep the low rank. Experiments indicate that it can achieve faster editing while maintaining editing quality comparable to that of prior editors.

### Strengths
1. Although low-rank modeling and SMW are established, the paper integrates them in the model editing setting to deliver tangible speedups without changing the editing quality.
2. Leveraging the FFN down-projection SVD as a prior to estimate the key covariance is a clever way to reduce the number of samples and preprocessing time.

### Weaknesses
1. In many editors, the dominant per-edit cost stems from optimizing the value vectors $V$, and the inverse is often a smaller fraction. FastEdit primarily accelerates the inverse step.

2. As the number of edits grows, the rank of accumulated keys may increase. While periodic compression is proposed, the paper does not yet demonstrate that performance remains reliable at larger scales.

3. The method introduces several hyperparameters. The combined sensitivity and robustness of outcomes to these choices is insufficiently characterized.

### Questions
1. Please evaluate substantially more edits to test whether periodic compression continues to preserve efficacy.

2. How exactly is ''editing time'' measured—end-to-end (including data loading and forward), per-edit from the first optimization step, or only the inverse step? More details would help highlight the method's contribution.

3. How does FastEdit’s editing time compare to the parameter-preserving methods like GRACE? For sequential editing, SimIE should also be included, as it has demonstrated strong performance.

### Soundness
2

### Presentation
3

### Contribution
2
