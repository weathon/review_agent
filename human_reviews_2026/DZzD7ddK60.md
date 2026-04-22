# Segmented Operations using Matrix Multiplications

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Specialized computational units that perform small matrix multiplications as primitive operations are typically present in modern AI accelerators. However, these Matrix Multiplication Units (MMUs) are often underutilized for many fundamental deep learning operations besides dense matrix multiplications. Coincidentally, the lack of a rigorous theoretical model of computation for such architectures obstructs algorithmic design. In this work, we propose MMV-RAM, a computational model which judiciously extends the Vector-RAM model with an additional MMU. We provide a detailed theoretical analysis, and carefully balance the computational power between the matrix and vector units, guided by the circuit complexity lower bound that parity is not in AC[0]. Given MMV-RAM, we proceed to algorithm design, starting with two fundamental parallel operations: *segmented scan* and *sum*. By expressing them as compositions of elementary parallel primitives (e.g., seg. sum reduces to: scan, compress, and vector differentiation), we can  exploit MMUs to perform *speculative* blocked computations, ultimately leading to *provable theoretical speed-ups* against vector-only approaches. These results extend to other ubiquitous AI kernels, including dense matrix product, and sparse matrix-vector product. As a case study, we implemented the proposed algorithms on the Ascend 910B AI accelerator, which contains both matrix and vector cores. We evaluate these implementations on synthetic and real-world datasets from various applications, including Large Language Models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Introduces a new compute model called MMV-RAM, consisting scalar, vector & matrix units with memory. This extends the existing Vector-RAM model with a matrix unit. The vector and matrix units are distinguished by their circuit complexity class - the vector unit operations must be in AC[0] (unbounded fan-in, polynomial width, constant depth), while the matrix unit is restricted to linear transforms, but these are not in AC[0].

The work describes parallel algorithms for scan, segmented scan and segmented sum, with good computational complexity under this model. These can be used to support a sparse matrix-vector product (SpMV). Finally, a demonstration in software on deep learning hardware shows the benefit of these methods that use the matrix unit for segmented scan and SpMV.

### Strengths
The paper is well-written and the claims and results are clear. The question of which compute model to use for algorithm analysis is relevant and interesting to the community. Particular strengths:

 - Clarity of notation, examples and algorithms in general, as well as diagrams and plots.
 - Simplicity of the proposed model, MMV-RAM.
 - Section 4.2 on applications of segmented operations was helpful for understanding the impact of the main results.
 - Details of proofs, algorithms, and experimental datasets in the appendix are comprehensive and appreciated.
 - The practical algorithm benchmarks in software are appreciated.

### Weaknesses
I am not experienced in circuit complexity theory, and so I hope to be corrected on my main concern, regarding the bounds that are shown to justify the matrix unit. In Theorems 4.1 and 4.2, the step bound is given $\mathcal{O}(\log_s(n))$ with matrix unit, and $\Omega(\frac{\log(n)}{\log(\log(n))})$ without. Since $s$ is a model parameter that does not depend on $n$ (L184), these bounds seem to me to overlap, as if I treat $s$ as a constant, $\mathcal{O}(\log_s(n)) = \mathcal{O}(\frac{\log(n)}{\log(s)}) = \mathcal{O}(\log(n)) \supset \mathcal{O}(\frac{\log(n)}{\log(\log(n))})$. So, for any given $s$, I can't see how these bounds would justify MMV-RAM over Vector-RAM?

Specific concerns:

 1. The experiments are interesting, but from my perspective do little to justify the cost model. Specifically, I imagine an accelerator with two vector units, one fast & one slow, could exhibit identical behaviour to the Ascend with vector & matrix units. To understand the importance of separating vector and matrix units in the cost model, it might be more convincing to look at the hardware implications: for an example from another domain, Rouhani et al. (2023) demonstrate their low-precision format ideas using ASIC synthesis and area estimation - this is likely too far, but for this work, some analysis that allows for more exploration of design parameters rather than being constrained to a specific accelerator might be more convincing.
 1. The model assumes computing an $n \times s$ with $s \times s$ product in a single step, which seems quite far from hardware designs. This is addressed in L802, Appendix A.2, which describes tiling an $s \times s$ square product, but since we are concerned with parallel algorithms, I wonder if would be simpler to specify the matrix unit as multiplying $s \times s$ and $s \times s$ in a single step, as we can execute arbitrarily many (e.g. $\lceil \frac{n}{s} \rceil$) of these operations in parallel, in any case.

Minor concerns:

 - The paper is somewhat lacking in breadth/depth of insights gleaned from the MMV model. For example, an explicit comparison of any advantage of MMV over TCU for the unsegmented scan of Zouzias and McColl (2023), as well as the introduced SCD and SSCR algorithms would be useful. Or, some deeper understanding of the resource requirements for balancing matrix and vector units, beyond the observation that both are necessary in Section 3.1.
 - I understand scan/segment-scan to typically become memory-bound when operands need to come from a HBM/DDR memory system, and that vector units can typically keep up with the memory system in this case. Is this the case for the Ascend AI 910B accelerator?
   - Aside: it would be useful to quote the L3 cache and scratchpad sizes referenced in L431.
   - I am surprised how slow the vector-only implementation runs in Figure 5.1, compared with CPU. What resource utilisation is this achieving versus peak vector arithmetic throughput?

---

_Darvish Rouhani, B., Zhao, R., Elango, V., Shafipour, R., Hall, M., Mesmakhosroshahi, M., More, A., Melnick, L., Golub, M., Varatkar, G. and Shao, L., 2023, June. With shared microexponents, a little shifting goes a long way. In Proceedings of the 50th Annual International Symposium on Computer Architecture (pp. 1-13)._

_Zouzias, A. and McColl, W.F., 2023, August. A parallel scan algorithm in the tensor core unit model. In European Conference on Parallel Processing (pp. 489-502). Cham: Springer Nature Switzerland._

### Questions
See questions mentioned above in "weaknesses". Many thanks!

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents a well-defined rationale for using segmented operations to improve modularity and scalability.

### Strengths
Demonstrates 30–40% performance gains in efficiency and reduced latency compared to traditional methods.
The segmented model is straightforward and applicable to real systems, with clear diagrams and structured explanations.

### Weaknesses
The paper lacks a formal analysis of segmentation boundaries and complexity trade-offs.
Evaluation is limited to local or small-cluster setups; performance on large distributed systems remains untested.
Missing information about hardware specs, configuration, and code availability, making reproducibility difficult.
Results are presented clearly but lack significance testing (e.g., error bars, confidence intervals).
Resource usage implications of segmentation are not deeply explored.

### Questions
There are many works studying using matrix operations for AI networks. Can you show your novelty and advantage over them.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the problem of underutilization of MMUs in modern AI accelerators . While these units perform well at dense computations, they are often idle during irregular and sparse operations, which are also common in deep learning.

### Strengths
Theoretical Guarantees: The paper provides a formal theoretical analysis, proving that its algorithms achieve a step complexity of O(log_s​(n)). This is provably faster than any vector-only algorithm, which is lower-bounded.

Novelty - MMV-RAM model. The paper addresses a key gap left by the prior "TCU model" by formally including the Vector Unit (VCU). This makes it a more accurate theoretical representation of modern accelerators like TPUs, NVIDIA GPUs, and Ascend NPUs, which all have both matrix and vector units.

### Weaknesses
Doubt on Generalization, Requirement of Custom Hardware: The experimental speed-ups are demonstrated on a Huawei Ascend 910B using the proprietary AscendC programming framework. While the paper lists analogues (e.g., NVIDIA Tensor Cores), the results are not on commodity hardware, making them less generalizable.

Theoretical vs. Practical Complexity: The most work-efficient algorithm presented (Theorem 4.3) is admitted to be "rather involved" and requires "specialized circuitry that might not be available on existing hardware," making it "mainly of theoretical interest". The simpler, implemented algorithms (SCD/SSCR) have a quadratic work complexity in their initial analysis, which is not ideal.

Unfair Baseline Comparison: The SpMV experiments compare their Ascend NPU implementation against CPU libraries (MKL and Eigen). They do not (and state they cannot) compare against an optimized SpMV implementation for their own Ascend hardware. While the results are encouraging, comparing a next-gen NPU to a last-gen CPU architecture isn't a direct apples-to-apples performance win.

### Questions
SpMV experiments were "deliberately chosen" to fit in cache to study arithmetic intensity, not I/O complexity . In many large-scale LLM and graph applications, SpMV is famously memory-bandwidth bound. How do you project your algorithm's performance will change when data must be streamed from HBM, and is there a risk that the overhead of the speculative MMU computation will be negated by I/O bottlenecks?

experimental validation was a case study on the Ascend 910B accelerator. How readily could your algorithms be mapped to other common AI accelerators, such as NVIDIA Tensor Cores or Google TPUs? Do you foresee any significant barriers in the programming models (like CUDA) to implementing the "speculate and correct" logic with the fine-grained control you require?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
**Summary:**

This paper presents a new design for performing vector and matrix operations. It integrates an MMU unit alongside the VCU to handle core arithmetic computations in AI accelerators. The authors theoretically demonstrate the speedup in terms of the number of steps and total work required. They also propose an algorithm for scan and segmented-sum operations and evaluate its performance empirically.

**Strengths:**

1. The paper provides theoretical guarantees on the number of operations (steps) required for scan and segmented operations, which extend to more general computations.

2. The proposed MMV-RAM model is general and treats existing VCU and MMU architectures as black boxes, allowing future improvements in these components to directly benefit the model.

3. The authors develop efficient algorithms for segmented scan and related primitive operations.

4. Experimental results demonstrate a clear speedup, supporting the theoretical analysis.

**Weaknesses and Questions:**

*Trade-offs in Hardware:*

1. The design integrates an MMU alongside the VCU; what are the trade-offs involved? In particular, since *energy efficiency* is critical for AI accelerators, does this addition significantly increase power consumption? A discussion or experimental analysis on this aspect would strengthen the paper.

2. From a *hardware implementation* perspective, does incorporating an MMU introduce challenges in *physical circuit design* or substantially increase the silicon area? (Acknowledging that I am not an expert in this area.)

*Theoretical Analysis:*

3. The paper states that $T^\prime = \frac{\log n}{\log s}$ is an improvement over $T = \frac{\log n}{\log \log n}$. However, theoretically, when $s = O(1)$, we still have $T^\prime = \Omega(T)$. Can you explain more about this? I know you said for **an appropriate value of $s$** it is true. More clarification on this value is needed. In addition, a discussion of the *appropriate* values of $s$ in the experiments (for what values this yields the speedup and for which doesn't) is important.

*Experiments:*

4. The experiments should include a runtime analysis of MMV-RAM across varying matrix sizes $n$. It would also be valuable to examine how different choices of $s$ impact the overall performance.

5. The paper does not appear to include experiments on general matrix–matrix operations beyond SCAN and SpMV. It would be helpful to include results for matrix–matrix multiplication, unless I have overlooked them.

6. It would be valuable to include evaluations on more **end-to-end** tasks, such as Transformers or other deep neural networks, to assess whether the proposed design leads to practical end-to-end speedups. In particular, I recommend adding experiments related to the *attention head*, which is especially relevant due to its use of keys and queries and the potential for sparsity. It would be interesting to see whether the proposed algorithms achieve measurable acceleration in this setting. I know adding additional experiments might be painful, but the current experiments do not seem to cover more general tasks, unless you can convince me on that.

*Related work:*

7. Several prior works have achieved speedups using segmented-sum and scan-like operations, such as [1]. It would be useful to discuss how the proposed approach relates to or could be integrated with such methods. Including this in the related work section would help clarify the broader impact and applicability of the proposed design.

*Minor Issues:*

8. On line 283, there is an extra “is” in the sentence — it should read “parallel COMPRESS can be” instead of “parallel COMPRESS is can be.”

**References:**

[1] “An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks.”

### Strengths
Please see the 'Summary'.

### Weaknesses
Please see the 'Summary'.

### Questions
Please see the 'Summary'.

### Soundness
3

### Presentation
3

### Contribution
3
