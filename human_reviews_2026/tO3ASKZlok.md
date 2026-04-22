# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 4, 10, 6

## Abstract
Vector quantization, a problem rooted in Shannon's source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. We propose TurboQuant to address both mean-squared error (MSE) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates. Our data-oblivious algorithms, suitable for online applications, achieve near-optimal distortion rates (within a small constant factor) across all bit-widths and dimensions. TurboQuant achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates, and leveraging the near-independence property of distinct coordinates in high dimensions to simply apply optimal scalar quantizers per each coordinate. Recognizing that MSE-optimal quantizers introduce bias in inner product estimation, we propose a two-stage approach: applying an MSE quantizer followed by a 1-bit Quantized JL (QJL) transform on the residual, resulting in an unbiased inner product quantizer. We also provide a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer, demonstrating that TurboQuant closely matches these bounds, differing only by a small constant ($\approx 2.7$) factor. Experimental results validate our theoretical findings, showing that for KV cache quantization, we achieve absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel. Furthermore, in nearest neighbor search tasks, our method outperforms existing product quantization techniques in recall while reducing indexing time to virtually zero.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The author introduce a new method for online vector quantization, which does not assume data-specific tuning, which is especially useful for applications such as KV cache compression in LLMs. Their method consist of applying a random rotation to map the (unit-normalized) distribution to a representation with near-independent coordinates whose distribution converges to a known distribution in high dimensions. This allows them to pre-compute per-coordinate $b$-bits codebooks optimized for this distribution, that are suitable for online usage. They further combine this method with QJL to provide unbiased inner product, which is important for KV cache compression.

### Strengths
- The proposed method is clearly explain and simple to use.
- The proposed method is online, making it easily applicable to KV cache compression.
- The authors provide theoretical justifications and bounds.
- The proposed method has potential for broad application on LLM inference

### Weaknesses
While the presented method has undeniable strengths, they are undermined by weak experimental validations.
- For KV compression, all compared methods use a different number of bits, making comparison of the accuracy-compression ratio harder to read. Better analysis of where this method takes place on this trade-off is needed.
- KL compression is compared to only two other comparison methods, ignoring some other methods, including RotateKV which is cited by the authors. Comprehensive comparison is needed.
- Authors propose “TurboQuant”, which revolves around the random matrix multiplication followed by per-bit quantization. The QJL and two-tier channel-wise quantization strategy parts come from other work and are generally applicable to some other methods, but are used in the comparison. Analysis of the individual component introduce by the paper would make results more convincing.
- This method is only compared to existing ones on a single benchmark, and single model.

The near neighbor search experiments are the least convincing:
- The method is only compared to PQ, and another method that barely surpasses PQ. For this kind of data-dependant method already surpass these baselines, including codebook-based methods (such as OPQ, RQ, LSQ++) or neural-network based (such as QINCo2 or UNQ).
- An argument is made about the time needed to quantize the training set, but this has very limited impact. Real-world ANN settings are usually constrained mostly by the search speed and accuracy on CPU, and the limited memory, more than the 1-time quantization. This part lacks justification.

### Questions
The method is interesting, and its simplicity is a strong argument for applying it. I would ask the authors to clarify their results and clearly justify the quality of the quantization in either KV compression or ANN settings.

### Soundness
1

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
TurboQuant is proposed to minimize both mean-squared error and inner product distortion in vector quantization by using random rotation and optimal scalar quantizers, achieving near-optimal distortion rates across all bit-widths and dimensions, with a two-stage approach for unbiased inner product estimation, and experimental results validate its superiority over existing methods.

### Strengths
As detailed below, I suspect there may be issues with the  two major innovations regarding mean-squared error and inner product distortion.

### Weaknesses
The two major contributions claimed in the paper appear theoretically and logically untenable.

1) **$Q_{mse}$:** This paper does not impose restrictions on the distribution of vector x, nor does it specify whether the elements of x are independent. If the elements are not independent (commonly seen in natural data), then multiplying x by a random rotation matrix cannot guarantee mutual independence among the resulting vector's elements (contrary to the statement in Line 215), rendering simple scalar quantization methods (which quantize elements individually) inapplicable. Consequently, the paper's claimed optimal mse quantization would not hold.

2) **Unbiased $Q_{prod}$:** The multiplicative bias of $2/\pi$  (derived in Line 296) for the inner product $⟨x,y⟩$ is a constant and does not affect comparisons between different inner products. Therefore, I question the necessity of eliminating this bias by introducing complex quantization schemes for the residual vectors.

### Questions
Besides the two major concerns mentioned above, I have the following questions:

1) Line 210: Why is multiplication with a random rotation matrix employed? Is the purpose to achieve a beta distribution? Additionally, is this approach originally introduced in this paper?

2) The experiments are insufficient, as comparisons are made with only two quantization methods, and only a few bit-width cases are examined. 

3) The “online” property claimed in the title is not explicitly demonstrated in either the theoretical analysis or experimental results.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper proposes TurboQuant as method for low-distortion vector quantization. To minimize MSE, TurboQuant first randomly rotates the input vector and then uses k-means generated codebook for each dimension. For an unbiased estimation of inner product, TurboQuant combines the random rotation method for MSE with an unbiased 1-bit quantization method via a residual quantization approach. Theoretical results show that the quantization errors of TurboQuant are close to the optimization. Empirical results show that TurboQuant provides good performance for both KV cache compression and nearest neighbor search.

### Strengths
Thank you for the interesting paper! I think this paper is among the best vector papers I have read in this year, and I learned a lot from the paper.

S1: A comprehensive discussion for the applications of vector quantization is provided in the introduction.

S2: TurboQuant has a low quantization complexity and yet strong performance.

S3: The theoretical analysis is in-depth, and the theoretical results are strong.

S4: Combing different vector quantization methods via a residual quantization like approach, despite straightforward, is interesting and makes sense.

S5: The experiment results are strong and cover both KV cache compression and nearest neighbor search.

S6: The presentation is fluent. Although the paper is heavy on math, the author makes it easy to read even for readers that may not be familiar with this area.

### Weaknesses
The paper can be enhanced with more detailed discussions and experiments to compare TurboQuant with RaBitQ and variants (e.g., [1]). Note that I do not think RaBitQ affects the novelty of TurboQuant since the theorical results of TurboQuant are much stronger.

D1: RaBitQ is discussed in the appendix but the discussions are far from insufficient. The original RaBitQ quantization is slow due to the joint search for the optimal quantized bits over dimensions but a recent work [1] solves the problem. Moreover, [1] also uses PCA to significantly reduce the quantization error of RaBitQ. RaBitQ and variants are similar to TurboQuant in that they all use random projection; but they are also different in crucial aspects, (1) they search for quantized bits while TurboQuant uses k-means generated codebook independently for each dimension; (2) they use a re-normalization trick for unbiased inner product estimations while TurboQuant combines random rotation with another vector quantization method. A crucial question is that which design is better for the two purposes, and giving clear answers will be valuable for this area. Some ablation experiments can be added for these design choices, e.g., check the quantization error by using different design combinations. It will even better if some theorical analysis can be conducted.     

[1] SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation

### Questions
NA

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TurboQuant, an online vector quantization algorithm designed to speed up nearest-neighbor retrieval and vector search by continuously updating codebooks as new data arrives. The core idea is to maintain good quantization quality in a streaming setting by using lightweight updates rather than full re-training. The method provides theoretical guarantees on quantization error and memory usage and is evaluated on several ANN benchmarks to show improved trade-offs between accuracy, latency, and memory.

### Strengths
S1 Addresses a relevant and timely problem: efficient vector quantization in streaming/online settings, which is important for large-scale retrieval systems.

S2 Includes theoretical analysis that supports the stability and bounded error of the online updates.

S3 Empirical results show clear improvements over static or periodically-retrained quantization baselines in both speed and accuracy.

S4 Paper is generally well-written, and motivations for online updates vs. periodic retraining are clearly explained.

### Weaknesses
The paper is in general quite solid. My question is that the paper seems missing comparisons with stronger or more recent quantization methods, such as those using residual or product quantization variants with adaptive codebook updates, or learned quantizers from recent literature. It’s not entirely clear how TurboQuant performs relative to the current state of the art. Can you provide more literature review and comparisons?

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
3
