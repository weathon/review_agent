# Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs

- Decision: Reject
- Scores: 0, 8, 2, 4

## Abstract
Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40\% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley's algorithm. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper provides a theoretical explanation showing that the Hadamard matrix is better than a random rotation matrix in reducing outliers in the LLM quantization.
It also proposes two techniques: gradual binary search (BGS) and dimension expansion. These two techniques improve the perplexity and benchmark accuracies and allow more flexibility in the layer dimension requirements.

### Strengths
In terms of originality, while the Hadamard matrix is well-known to be effective for LLM quantization in practice, this paper is one of the first to explain this phenomenon from a theoretical perspective. It also proposes two new methods (gradual binary search and dimension expansion) that can be integrated into the QuaRot framework.

In terms of quality, both theoretical findings and experimental results are presented and discussed in the paper.

In terms of clarity, the text of the paper is easy to understand overall.

In terms of significance, the proposed two methods show better results in both perplexity and benchmark accuracies.

### Weaknesses
1. This paper looks like an incremental work that consists of two independent methods: gradual binary search (BGS) and dimension expansion.
- BGS is not linked to any theory in the paper and is only based on an assumption: the optimization landscape is convex and has a single minimum. Also, this binary search is inefficient and not scalable, as it requires a very long runtime (12 hours on A100 for a 8B model).
- The dimension expansion is only weakly linked to the theoretical part of the paper. The theory states that the Hadamard matrix reduces outliers better than a random rotation matrix. However, expanding the dimensions to match the available sizes of Hadamard matrices, especially with zero padding, is a trivial engineering practice.

2. The theories in Section 3.2 are not sound.
- In Theorem 3.1, the outlier is not well-defined.
The form $x = \epsilon + \sum_{j=1}^{k} c e_{p_{j}}$ cannot properly describe a vector in $\mathbb{R}^n$ containing $k$ outliers.
For example, let $x = (100, 100, \dots, 100)^\top$.
We can consider it as a vector of either $k=n$ outliers ($\epsilon = [1, 1, \dots, 1]^\top$, $c=99$) or $k=0$ outliers ($\epsilon = [100, 100, \dots, 100]^\top$, $c=10000$).
According to the equation in Theorem 3.1, if $k=n$, Hadamard $H$ would have a larger $\mu$ than random rotation $Q$.
And if $k=0$, $H$ would have a smaller $\mu$ than $Q$.
This is a contradiction.
- In Theorem 3.1 and Lemma 3.2, I do not understand why a rotation matrix can be drawn from a unit sphere $S^{n-1} = \\{ x \in \mathbb{R}^{n} : \\| x \\|_{2} = 1 \\}$, as each sample on the sphere is a $n$-dimensional vector, not a $n \times n$ matrix.
- In Line 257, the term $O(\sqrt{\frac{k}{n}})$ is inconsistent with Lemma 3.1, where $k$ is not under the square root.
- In Definition 3.1, it is not necessary to define a new function using $\mu$. There is an existing concept called the L-infinity norm that does exactly this.

3. The GBS algorithm introduced in Section 4.1 is not clear. It does not describe the definition of the clipping ratio and how the ratio is applied during quantization. In particular, why is it always between 0 and 1 in the pseudocode? It is also not clear how this algorithm (pseudocode) is applied to weights, activations, and KV caches.

4. The citations violate the ICLR formatting instructions. Instead of using `\citet{}` inside a parenthesis, the authors should use `\citep{}` without the parenthesis. Note that with `\citep{}`, the year will appear behind a comma instead of inside another parenthesis. There are also a few citations that only have names but not years. Please check Section 4.1 of the "Formatting Instructions for ICLR 2026 Conference Submissions" document in the provided LaTeX template.

5. Other errors:
- The equation on Line 74 is wrong. The term $2^b$ should be $2^{b-1}$.
- On Line 94, while the GPTQ algorithm itself runs for any bitwidth, the original paper already showed good results on 4-bit. "GPTQ enables 8-bit quantization ..." is not a correct statement.
- On Line 170, the statement "both rotation matrices and Hadamard matrices are essential for the quantization ..." is inaccurate. Firstly, Hadamard matrices are also rotation matrices. Secondly, as the authors suggest, the Hadamard matrix is better than a random rotation matrix, and it should be the only essential one.
- On Line 339, using the inequality in Lemma 4.1, I get $d \le \frac{4096 \times (4-3)}{3} \approx 1365.33$. So the maximum $d$ is 1365 instead of 1366.
- In Tables 1 and 2 and other parts of the paper, only the Llama2 7B model uses the default datatype FP16 (float16); the default datatype for other models is BF16 (bfloat16).

6. The writing is poor. There are many typos.
- On Line 104, the quote symbols around ”sink tokens” should be “sink tokens” (please note the directions).
- On Line 203, "trough" should be "through".
- On Line 229, the articles "a the" are redundant.
- On Line 323, "figure" should be "Figure".
- On Lines 350-351, "We performs ours experiments" should be "We perform our experiments".
- On Line 470, I am not sure what "mix computation" is. Should this be the same as "mixed computation" in Section 7, Line 476?

7. As a general comment, I think Section 3.1 or at least its first paragraph belongs to the related work (Section 2).

### Questions
1. In the caption of Figure 1, what are the differences between QKV and Head projections, and what are the differences between Out and Embedding projections?

2. On Line 324, why can increased dimensionality enhance performance? I think the increased dimensionality requires more computation and slows down the performance.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a systematic solution for extremely-low-bit LLM quantization by combining three key ingredients: 1) theoretical proof that Hadamard rotations suppress activation outliers more effectively and efficiently than random orthogonal matrices; 2) Gradual Binary Search (GBS), a layer-wise greedy algorithm that optimizes the clipping ratio for each projection by minimizing perplexity, achieving 3-bit W/A/KV quantization with up to 40% higher average accuracy on Mistral/LLaMA/Qwen families and clearly outperforming prior methods; 3) dimension expansion via zero-padding and Paley’s construction to enable Hadamard transforms on non-power-of-two embeddings, further reducing outliers. The evaluation is thorough and the ablations convincing, making the work valuable for edge deployment.

### Strengths
1. Solid theory: First rigorous proof that Hadamard matrices suppress activation outliers more effectively than random rotations, providing a deterministic foundation for subsequent work.

2. Novel method: Introduces GBS, a layer-wise binary-search that optimizes clipping ratios directly for perplexity, circumventing the failure of traditional “quantization error” metrics at very low bits.

3. Significant performance: 3-bit W/A/KV quantization improves average accuracy by up to 40 % on Mistral/LLaMA/Qwen families, approaching FP16 baselines and clearly outperforming existing SOTA.

4. Broad applicability: Dimension expansion combined with Paley’s construction breaks the power-of-two barrier, enabling Hadamard-based quantization for non-standard architectures like Qwen.

5. Comprehensive evaluation: Extensive experiments across multiple models, tasks, and bit-widths, with ablations on gradual vs. one-shot search and different initializations, yield credible results and demonstrate practical deployment potential.

6. Overall great work.

### Weaknesses
1. Although the accompanying experiments demonstrate that the layer-wise greedy strategy of GBS performs reliably in practice, it still fails to provide solid evidence that the accumulation of all locally optimal clipping ratios constitutes a globally optimal solution.

2. The need to compute perplexity many times for every layer incurs significant overhead; while this can be regarded as an inherent drawback of the method, it remains an aspect that warrants future improvement.

### Questions
I would welcome a deeper analysis from the authors on how the layer-wise greedy nature of GBS relates to the global optimum; nevertheless, this issue does not diminish the contributions of the present work.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an approach to perform W/A/KV quantization built over QuaRot. The main contributions are performing clipping for reducing quantization error and dimension expansion to enable application of hadamard matrix to arbitrary dimensions. While it is no surprise that clipping helps in achieving better performance, it is not clear why authors went with binary search for finding clipping ratio. A common alternative is also to learn quantizer parameters to improve quantization but it is not clear which one would work better. Theoretical analysis in this work shows superiority of hadamard rotation over random orthogonal rotation, but in practice, both Quarot and SpinQuant use randomized hadamard rotation. Therefore there is a mismatch between theory and implementation. Finally, the dimension expansion approach adds computational cost. Another alternative would be to not perform dimension expansion and leverage random orthogonal rotations for quantization. The paper does not provide enough evidence to support dimension expansion approach.

### Strengths
1. The paper is well written and easy to follow.
2. The motivation for using clipping to improve quantization makes sense.
3. The results clearly outperform QuaRot.

### Weaknesses
1. The paper is missing the motivation behind using GBS for finding clipping ratio as opposed to learning quantization parameters.
2. The theory is based on hadamard rotation but Quarot and SpinQuant use randomized hadamard rotation.
3. The motivation for dimension expansion is not clear. Agreed that it enables using hadamard rotation, but it also adds computational cost. An alternative would be to resort to randomized rotation. 
4. The evaluation baselines are weak. More recent LLM W/A/KV quantization baselines should be shown : SpinQuant (https://arxiv.org/pdf/2405.16406), DuQuant (https://arxiv.org/abs/2406.01721) , ResQ (https://openreview.net/forum?id=4qIP1sXcR1), OSTQuant 9https://arxiv.org/abs/2501.13987).
5. Also it is valuable to expand the evaluation tasks to cover more domain : math understanding, long context, coding, reasoning, etc.
6. Hardware performance evaluations are missing.

### Questions
Please see weaknesses.

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
4

### Summary
This paper presents a method that combines Gradual Binary Search (GBS) with Hadamard-based rotations to improve post-training quantization for large language models. The approach focuses on optimizing clipping ratios per projection using perplexity as an objective, supported by a gradual quantization schedule that stabilizes optimization. The authors also introduce dimension expansion, allowing Hadamard transforms to work with non-power-of-two model dimensions.

Theoretical analysis shows that Hadamard matrices suppress outliers more effectively than random rotations, and experiments confirm this advantage. Extensive evaluations on multiple LLM families demonstrate strong improvements at both 4-bit and 3-bit precision, outperforming prior methods.

### Strengths
The experiments are broad and well-executed. The authors evaluate across several modern LLM architectures and show consistent gains with the proposed quantization method at both medium and low bit levels. The results are stable and show clear improvement over existing baselines.

The analysis connects theoretical intuition with empirical behavior, validating the effect of Hadamard rotations and exploring the trade-off between accuracy and compute when expanding dimensions. The inclusion of runtime and calibration ablations adds practical depth. Overall, the work demonstrates that the proposed approach generalizes effectively and holds up under realistic model and deployment settings.

The paper is technically clear and well-structured. It provides both theoretical grounding and implementation details, including concrete algorithms, equations, and appendices that make replication feasible. The link between the theoretical motivation (Hadamard outlier suppression) and its practical use in quantization is clearly articulated, helping readers understand not just that it works, but why. This balance between rigor and practicality strengthens the paper’s overall credibility.

### Weaknesses
While the paper presents clear theoretical and empirical gains, some important aspects of computational practicality remain underexplored.
First, the runtime and memory overhead of increasing the Hadamard matrix size (via dimension expansion) are not well quantified. As shown in Figure 2, performance continues to improve with expansion, but there are no concrete measurements of latency, FLOPs, or memory cost relative to baseline models. It remains unclear whether the added storage and compute required by expanded dimensions outweigh the efficiency gains from quantization and Hadamard rotation.

Second, the Gradual Binary Search (GBS) method—central to the paper’s contribution—is not sufficiently compared to simpler alternatives. There are no ablations on search hyperparameters (e.g., iteration count, dataset size, convergence threshold) beyond limited results in Appendix E, nor comparisons against heuristic or uniform clipping baselines. Moreover, the justification for using binary search assumes a unimodal perplexity landscape but provides little quantitative evidence to support this assumption.

Finally, the authors should provide comparison with ClipQ and sufficient literature review of clipping optimization methods.

### Questions
1. Could the authors quantify the runtime and memory overhead introduced by dimension expansion? For example, how much latency increase (tokens/s or wall-clock) and additional storage are observed per +512 or +1024 dimensions for LLaMA and Mistral?

2. How does the computational benefit vs. cost trade-off evolve when combining quantization with expanded Hadamard matrices? Does the accuracy gain outweigh the increase in compute?

3. Can the authors provide ablations comparing GBS to fixed or uniform clipping ratio settings (e.g., all projections share the same L), or a random-search/grid-search baseline?

4. Why was binary search chosen over alternative search methods (e.g., grid, ternary, gradient-based)? Does perplexity indeed form a convex landscape for each projection?

5. What is the sensitivity of GBS to search parameters such as dataset size, iteration limit, or threshold ϵ? Are results consistent across different seeds or calibration subsets?

6. Can the authors provide a comparison with ClipQ and describe the similarities and differences? 
ClipQ: Clipping Optimization for the Post-Training Quantization of Convolutional Neural Network

### Soundness
2

### Presentation
3

### Contribution
3
