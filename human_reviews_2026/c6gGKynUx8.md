# LUCID: Attention with Preconditioned Representations

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Softmax-based dot-product attention is a cornerstone of Transformer architectures, enabling remarkable capabilities such as in-context learning. However, as context lengths increase, a fundamental limitation of the softmax function emerges: it tends to diffuse probability mass, assigning non-trivial weights to irrelevant tokens. This dilutes focus and degrades precision, especially in long-sequence scenarios. We introduce \textit{LUCID Attention}, an architectural modification that applies a preconditioner to the attention probabilities. This preconditioner, derived from exponentiated key-key similarities, minimizes overlap between the keys in a Reproducing Kernel Hilbert Space, thus allowing the query to focus on important keys among large number of keys accurately. If a query $\mathbf{q}$ is highly similar to a key $\mathbf{k}$, LUCID outputs the corresponding value vector $\mathbf{v}$ with minimal blending from other tokens. This mechanism enables significantly sharper and more precise attention distributions. LUCID is designed as a drop-in replacement for existing attention mechanisms, retaining the same asymptotic complexity. We validate our approach by training $\sim$ 1 billion parameter language models, pre-trained on a 2K sequence length and then fine-tuned up to a 65K sequence length. Our results demonstrate improved next-token prediction loss and significant gains on long-context retrieval tasks. LUCID shows an average improvement of $\sim$20\% in single and multi-needle in a haystack benchmarks compared to standard attention.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce LUCID Attention, a modification of attention that uses a pre-conditioner on attention probabilities. This is derived from the key-key similarities, where the overlap between keys is reduced within a RKHS, allowing the query to focus more accurately on important keys and providing more precise attention distributions. Results on needle-in-a-haystack tasks shows a noticeable improvement, while next-token-prediction losses also sees improvements.

### Strengths
The proposed method is well motivated and appears to be broadly applicable to different attention variants. Performance on needle-in-a-haystack (NIAH) tasks shows that the proposed method offers meaningful raw performance gains compared to regular attention.

### Weaknesses
- The proposed method still appears to suffer from the same degradation in performance with sequence length or with the number of needles, as presented in the M-NIAH results. While it does appear that Lucid attention is effective up to 8192 sequence length on the S-NIAH setting, there is a question regarding whether or not this is explicitly meaningful.

- There is a lack of non-synthetic task performance. For example, LongBench (for long contexts) as well as tasks from the `lm-eval-harness` suite should be provided in order to understand explicitly whether or not the proposed method is useful/meaningful. Table 6 in the appendix provides some results but only compares against standard attention; I would suggest comparison with both attention based methods here (GLA, GSA, DeltaNet, etc.) as well as non-attention based models (Mamba, RWKV, RetNet, etc.) Furthermore, as the motivation of the method appears to be enabling models to focus on more important information, sparse attention methods should further be considered as baselines.

- Bench-marking efficiency against these methods would appear to be relevant. The preconditioning appears to involve a matrix inverse; this can be costly and furthermore it is not guaranteed to be stable. The authors should consider investigating the condition number of such matrices in order to verify that the current implementation will in fact be stable, especially if implemented in lower-precision settings as is standard for training transformers.

### Questions
See above.

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
LUCID Attention introduces a preconditioned variant of softmax attention that decorrelates keys in the Reproducing Kernel Hilbert Space (RKHS) using an exponentiated key–key similarity matrix, enabling sharper and more precise focus on relevant tokens. This approach preserves the standard $O(N^2d)$ complexity while guaranteeing precise value retrieval when queries equal keys and avoiding the vanishing-gradient problem seen in low-temperature softmax. Experiments on large-scale language models show consistent improvements in long-context retrieval tasks (e.g., “needle-in-a-haystack”)

### Strengths
The proposed method is novel and consistently outperform baselines across synthetic benchmarks.

### Weaknesses
1. Currently benchmark are mostly synthetic. Could authors compare their methods with baselines on real-world NLP long-context modeling benchmarks, e.g. LongBench [1] ?
2. How is the matrix inversion efficiently implemented? Could authors compare the wall time against softmax attention?
3. What is the motivation of assuming Q=K in the theoretical part? Why we need to find a P that multiplied to the attention score and produce an identity matrix?

[1]. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. ACL 2024.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LUCID Attention, a novel attention mechanism that applies a preconditioner derived from exponentiated key-key similarities to sharpen attention distributions and reduce noise from irrelevant tokens. The method aims to improve retrieval precision in long-context scenarios while retaining the same asymptotic complexity as standard softmax attention. The authors evaluate LUCID on needle-in-a-haystack tasks and show improvements over several baselines.

### Strengths
- The idea of using a key-key similarity preconditioner to decorrelate keys in a Reproducing Kernel Hilbert Space (RKHS) is novel and theoretically motivated.  
- The method is presented as a drop-in replacement for standard attention, requiring no additional parameters and maintaining \(\mathcal{O}(N^2 d)\) complexity.  
- Empirical results on SNIAH and MNIAH tasks demonstrate improved retrieval performance over standard attention and some existing variants.

### Weaknesses
1. **Limited Empirical Validation of Core Claims**  
   The paper claims that LUCID improves focus and reduces attentional noise, but no direct evidence is provided (e.g., visualization of attention maps or quantitative analysis of attention sparsity). Similarly, the claim that LUCID mitigates gradient vanishing is not empirically verified. It would be valuable to:
   - Visualize attention distributions for LUCID vs. standard attention on long-context examples.
   - Include gradient norm analysis during training to support the non-vanishing gradient claim.

2. **Baseline Comparisons Could Be More Comprehensive**  
   The current baselines (Standard, Diff Transformer, DeltaNet, PaTH) are reasonable but not exhaustive. Given that LUCID belongs to the family of methods that modify attention scoring, it would be beneficial to compare against:
   - Sparse attention mechanisms 
   - Gating-based approaches (e.g., Stick-Breaking Attention)
   - Methods with dynamic decay or routing mechanisms

   Such comparisons would better situate LUCID within the broader landscape of attention enhancements.

3. **Efficiency Analysis Is Incomplete**  
   While the authors state that LUCID has the same asymptotic complexity as standard attention, no wall-clock time or memory usage comparisons are provided. Given the additional triangular solve and kernel matrix computation, a more thorough efficiency analysis—especially for long sequences—would be helpful to assess practical utility.

4. **Orthogonality and Integration with Other Methods**  
   It is unclear whether LUCID is compatible with or complementary to other popular attention enhancements, such as:
   - Sparse attention (e.g., NSA)
   - Advanced positional encodings (e.g., ALiBi, RoPE variants)
   - Linear attention approximations

   The authors should discuss whether LUCID can be combined with such methods and whether it offers orthogonal benefits.

5. **Significance in the Context of Large-Scale Models**  
   Recent large-scale models (e.g., Llama, Qwen) with full softmax attention and modern positional encodings already perform well on needle-in-a-haystack tasks. The authors should contextualize their contribution more clearly:
   - Is LUCID primarily beneficial in resource-constrained or narrow-head settings?
   - How does it scale with model size and training data?

### Questions
See the weaknesses section for details

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes LUCID Attention, a new attention mechanism that modifies standard softmax attention through a preconditioner derived from exponentiated key-key similarities in the RKHS feature space. The method aims to decorrelate keys, mitigating “attentional noise” and producing sharper, more precise attention distributions—particularly beneficial for long-context retrieval tasks.

Unlike linear or sparse attention variants, LUCID retains the same $\mathcal{O}(𝑁2𝑑)$ complexity and serves as a drop-in replacement for standard attention. Empirical results on 1B-parameter language models demonstrate consistent gains in Needle-in-a-Haystack (NIAH) benchmarks and small but measurable improvements in downstream language understanding tasks.

### Strengths
1. The paper presents a well-motivated and theoretically grounded contribution. It provides a clear diagnosis of the limitations of softmax attention: its diffuse probability distributions and key correlations. This paper also introduces LUCID through a clean, RKHS-based derivation.

2. The proposed preconditioning step is mathematically elegant and interpretable. It functions as a decorrelation operator that enhances retrieval precision and training stability without adding extra parameters.

3. Empirical results demonstrate consistent and meaningful improvements. LUCID achieves strong gains on single- and multi-needle retrieval tasks, along with modest yet positive improvements on standard NLP benchmarks, indicating practical benefits for long-context modeling without increased computational complexity.

4. The method is implementation-friendly. LUCID can be seamlessly integrated as a drop-in replacement for standard attention within Transformer architectures, maintaining comparable runtime and memory requirements.

5. Comprehensive analysis strengthens the work’s credibility. Analytic evidence of non-vanishing gradients, together with detailed ablation studies (on key normalization, QK-Norm, and RoPE) and scalability experiments, collectively reinforce the soundness and robustness of the approach.

### Weaknesses
1. Evaluation focuses heavily on NIAH benchmarks and lacks large-scale or real-world long-document tasks (e.g., book summarization, multi-hop QA). Broader validation would strengthen claims of generalization.

2. The paper evaluates LUCID against relatively few baselines, leaving open questions about its comparative advantages over a broader range of recent attention mechanisms.

### Questions
It remains unclear whether the non-vanishing gradient property arises primarily from the proposed preconditioner or from the normalization/scaling mechanisms. In Line 264, the paper states that "$a$ is not one-hot." Could the authors elaborate on why this holds? Moreover, the statement “Because LUCID Attention doesn’t use temperature setting, the softmax output is not one-hot” seems incomplete. Please clarify the underlying mechanism by which the absence of temperature ensures non-saturation.

### Soundness
3

### Presentation
3

### Contribution
3
