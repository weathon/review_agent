## Human Reviewer 1

### Summary
The paper introduces cLUT (compressed Lookup Tables), a method for exact, fast, and
energy-efficient random variate generation from arbitrary discrete distributions. The
key idea is to quantize the target distribution into integer frequencies, represent
those frequencies in binary, and compress the resulting lookup table by aligning entries
along bit-planes that correspond to powers of two. Sampling then reduces to drawing a
row index from a truncated geometric distribution and a column index uniformly from a
fixed-width table. This produces exact samples using only bit-level randomness and a
single memory access per draw.  
The authors benchmark cLUT against classical exact samplers and demonstrate superior
performance on large discrete distributions in both speed and energy efficiency. They
further validate the method in an applied setting showing
that replacing NumPy’s RNG with cLUT yields equivalent posterior results at
substantially lower runtime and energy cost. Overall, the paper proposes an elegant
unification of algorithmic simplicity, theoretical exactness, and practical efficiency
for one of the oldest primitives in probabilistic computation.

### Strengths
- **Conceptual elegance and originality.**  
  The compression of frequency tables along binary bit-planes is a simple yet powerful
  idea. It yields an exact, entropy-optimal sampling procedure that directly exploits
  the binary nature of digital hardware. The resulting geometric–uniform two-step
  sampling scheme is mathematically sound.

- **Clarity and quality of exposition.**  
  The paper is very well written, with intuitive explanations and instructive figures
  (especially Figs. 1–2). The visual decomposition of frequencies into bit-planes makes
  the algorithm conceptually understandable. 

- **Empirical thoroughness.**  
  Benchmarks span both high-level (NumPy, PyTorch, JAX) and low-level (C)
  implementations, reporting not only runtime but also energy via RAPL counters. The
  TrueSkill case study convincingly demonstrates real-world impact, reducing both time
  and energy while preserving statistical equivalence of posterior estimates.

- **Practical significance.**  
  Random number generation is a foundational operation across ML and simulation. A
  method that is *exact*, *faster*, and *more energy-efficient* without sacrificing
  correctness has immediate relevance for probabilistic modeling, simulation-based
  inference, and on-device ML. The bit-level efficiency arguments also connect nicely to
  ongoing discussions of energy-aware ML.

- **Implementation simplicity.**  
  The algorithm relies only on two primitive RNG calls: a geometric distribution
  obtained from bit flips and a power-of-two uniform integer. Both are exact and
  hardware-friendly. This makes the method easy to embed in existing frameworks.

### Weaknesses
- **Limited discussion of integration and practical deployment.**  
  While the method is described cleanly, the paper provides little guidance on how cLUT
  would be incorporated into standard frameworks such as `torch.distributions` or
  `jax.random`. In practice, the preprocessing cost and table storage would need to be
  managed per distribution instance or cached globally; a brief discussion of this would
  clarify the usability pathway.

- **Evaluation metric for TrueSkill comparison.**  
  The comparison of posterior estimates between NumPy and cLUT uses t-tests on means.
  This only probes first-moment agreement; distributional similarity could be assessed
  more rigorously with two-sample metrics such as the **C2ST** score (which should
  approach 0.5 for indistinguishable posteriors).

- **Memory and break-even analysis.**  
  Figure 5 shows clear scaling of preprocessing cost, but some contextualization would
  help. For very large supports (e.g., $n>10^8$), achieving the theoretical break-even
  might require enormous sample counts. Discussing typical discretization sizes for
  common continuous distributions (Normal, Gamma, etc.) would make the results more
  interpretable for practitioners.

- **Minor editorial points.**  
  Line 235: stray capital ‘T’.  
  Line 236: slight mix-up in text; should read “is replaced by two a in the third row,
  which corresponds to a frequency of 2.”

These are all minor and easily addressed; none affect the core soundness or clarity of
the work.

### Questions
1. **Construction for general distributions.**  
   The cLUT method requires a discretized finite support to construct its compressed
   lookup tables. Could the authors clarify how this construction would proceed for
   *continuous or parameterized* distributions (e.g., those in `torch.distributions` or
   `jax.random`)?  
   In particular:  
   - How are truncation bounds and binning strategies chosen for unbounded or
     heavy-tailed distributions?  
   - How is quantization precision $b$ selected in practice to balance fidelity and
     memory footprint?  
   - Could this preprocessing be automated so that cLUT can serve as a backend for
     standard probabilistic libraries?

2. **Vectorization and hardware integration.**  
   The reported benchmarks compare cLUT to scalar or lightly vectorized C baselines.
   Many ML frameworks already implement heavily vectorized sampling kernels.  
   - Can the authors comment on how cLUT would integrate into such environments, would
     the same speedups hold once both systems are equally vectorized? For example: 
      - Is the algorithm amenable to SIMD or GPU parallelization (drawing rows and columns
     in bulk and performing vectorized gathers)?  
      - How does cLUT’s memory access pattern compare to typical GPU-friendly RNG kernels?

3. **Memory–compute trade-off and practical break-even.**  
   Figure 5 illustrates a break-even analysis in terms of energy and wall-clock time,
   showing that cLUT’s higher preprocessing cost is amortized after sufficient draws.  
   - Could the authors contextualize these numbers for typical workloads? For example,
     how many discretized bins are required to approximate a continuous distribution
     such as a Normal or Gamma to standard ML accuracy, and how many draws would that
     entail in realistic training or inference runs?  
4. **Code availability and reproducibility.**  
   Will the authors release an open-source implementation, preferably an anonymous
   repository for review, covering both CPU and GPU backends?  
   Ideally, this should include scripts for rebuilding benchmark tables, integration
   examples with PyTorch or JAX, and automated evaluation of time and energy metrics.
   Such an implementation would be highly valuable for assessing portability and
   adoption potential.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes a new (to the best of my knowledge) method to generate random variables from arbitrary distributions, based on lookup tables. The approach is rather simple and well justified, and various benchmarks show a good improvement in time and energy savings with respect to state-of-the-art method.

### Strengths
The paper is clear, the approach is sound and simple (in a good way) while being theoretically justified.
Sampling is not a huge bottleneck in most modern applications of machine learning, but it is always good to make things more efficient.

### Weaknesses
I would have appreciated a more in-depth discussion of limitations. It does not seem clear to me that this will completely replace existing approaches, as it is less "out-of-the-box" than those. 
Additionally, while you touch upon one Bayesian learning application, the analysis is quite light. It would be better to include a more in depth analysis for a more standard application (such as a more complicated distribution, eg sampled with MC methods for bayesian posteriors).

### Questions
See weaknesses.

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

## Human Reviewer 3

### Summary
This paper introduces cLUT (compressed Lookup Tables), a sampling method designed to generate samples from arbitrary discrete distributions efficiently. The authors claim that sampling, a core component of many machine learning algorithms is a significant computational and energy bottleneck.

The proposed method is based on two key contributions (1) a lossless compression scheme via a constructing a compressed lookup table and (2) a fast algorithm to sample efficiently given the compressed lookup table.

The authors provide a Python and C implementation of their algorithm. The results are demonstrate that the Python implementation of cLUT is 10-100x faster than standard Python samplers (NumPy, PyTorch, JAX), depending on the number of samples. The C implementation of cLUT is more energy-efficient and faster than state-of-the-art (SOTA) C-based samplers like ALDR, FLDR, and the Alias method. However, this efficiency comes at the cost of a higher one-time preprocessing step. The method's practical benefits are demonstrated in a modified TrueSkill application, where it significantly reduces execution time and energy consumption.

### Strengths
**Significance**: The paper addresses a fundamental, practical, and important problem: the high energy and computational cost of sampling in ML. The claimed 33-60% energy savings and 10-100x speedup over common Python libraries is impressive.

**Originality**: I am not an expert in the particular area of low-level implementation of sampling algorithms. But the core idea of combining a specific binary-expansion-based compression with a geometric-plus-uniform index sampling scheme appears to be novel.

**Quality of evaluation**: The authors wisely compare against two separate groups: (1) high-level, commonly used Python libraries and (2) low-level, high-performance SOTA C implementations and demonstrate improved performance in terms of sampling time and energy efficiency across the board. They also will publicly release the implementation. The authors are upfront about the method's primary trade-off of cLUT's higher pre-processing time.

**Clarity**: The paper is mostly well-written and is easy to follow. I have some suggestions for further improvements that are detialed in the next sections.

### Weaknesses
**Missing Algorithmic Details on Preprocessing**: The preprocessing step, which constructs the compressed table, is a critical part of the contribution but is not well-explained. The main text describes a "rectification" process to ensure all rows have a uniform width, which is essential for the sampling scheme. This section refers to Algorithm 2 in the appendix. However, Algorithm 2 is itself a high-level sketch. It calls a function distribute(z, f, r, c)  which seemingly performs all the work, but this function is undefined. The visual example in Figure 2 is helpful but insufficient to understand how this redistribution is performed algorithmically.

**Vagueness on Key Parameter $r$**: The performance, compression ratio $\rho = 2^r / (r+1)$, and bit efficiency are all critically dependent on the number of rows, $r$. The paper is vague on how $r$ is determined, stating it "depend[s] on the frequencies f". The formula provided in Algorithm 2 (Line 2) 33 is complex and presented without any derivation or intuition. The paper would be much stronger if it provided bounds or illustrative examples of what $r$ typically is for common distributions (e.g., uniform, Gaussian) at a given precision.

**Limited ML Application Context**: The introduction strongly motivates the work by citing its relevance to VAEs, contrastive learning, and diffusion models. However, the evaluation only uses an extended TrueSkill system. While this is a valid real-world test, it is a relatively niche application. To compellingly demonstrate the paper's significance to the ICLR community, it would be far more powerful to show the effectiveness of plugging cLUT into even a small-scale generative model (like a VAE or a simple diffusion model) during training or inference. As is, the contribution feels more like a general-purpose algorithm paper than a paper demonstrating a clear impact on a core ML problem.

### Questions
I am wondering how generalizable the proposed method is. Is it possible to plug cLUT into pytorch so that all the sampling operations uses cLUT instead of whatever the default sampling algorithm is? If so, what would the end-to-end effect of such update be on a typical training/inference task?

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

## Human Reviewer 4

### Summary
This paper proposes a new sampling strategy for random variate generation from arbitrary distributions. The core idea is to construct a lossless compressed lookup table and perform efficient sampling from it. The approach to generating the compressed lookup table appears to be computationally efficient and potentially innovative.

### Strengths
The paper is well-written, and the experiments are sound. The baselines are well-chosen, ranging from generic Python library routines commonly used in the machine learning community to state-of-the-art competing methods for this particular task. The paper’s emphasis on energy efficiency is commendable and highlights an often-overlooked but important aspect of algorithmic design.

### Weaknesses
The absence of a computation complexity analysis appears to be the paper’s main weakness, though the authors may have a valid reason for omitting it.

### Questions
__Q1.__ In the conclusion section, you mention potential advantages of implementing your method on GPU. Since one of the comparisons is against GPU-accelerated libraries while your experiments are conducted only on the CPU, could you elaborate on these points in more detail? specifically, how these advantages arise compared to other methods and how they could be effectively leveraged or exploited in practice?

__Q2.__ While many discrete sampling routines in Python packages use C++ backends, how do you ensure that Python wrapper overhead does not skew your runtime comparisons?

__Q3.__ In Table 1, why is the sampling time faster for $n \in [10^6, 10^7]$ in your proposed method compared to smaller $n$ ($n \in [10^4, 10^5]$), while the other methods have slower sampling times when $n$ is larger?

__Q4.__ It might be helpful to also show, in Figure 5, the peak memory usage excluding the preprocessing cases. Additionally, it would be nice to include other memory-related metric(s) such as average memory usage as well as the overall memory behavior during execution.

__Minor__

- In line 107, the real RAM assumption should be introduced more clearly, perhaps with one or two explanatory sentences or a short note in the appendix. Instead of only referring to Shamos (1978). At the minimum, please provide the abbreviation of RAM (Random Access Machine).
- In line 149, while $c$ and $b$ are quite self-explanatory, it might be helpful to add a brief sentence explaining what $r$ represents to give readers a quick intuition. In general, a short description of all notation used could greatly improve the clarity of the paper.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
1