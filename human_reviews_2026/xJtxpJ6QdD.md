# Quantized Gradient Projection for Memory-Efficient Continual Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Real-world deployment of machine learning models requires the ability to continually learn from non-stationary data while preserving prior knowledge and user privacy. Therefore, storing knowledge acquired from past data in a resource- and privacy-friendly manner is a crucial consideration in determining their viability. We introduce Quantized Gradient Projection Memory (QGPM), a systematic framework for continual learning that compresses and preserves the previous gradient subspace. QGPM integrates three key components: (i) distribution-aware, basis-wise quantization to minimize storage overhead, (ii) a Quantization Error-Aware (QEA) gradient projection that selectively relaxes orthogonality to mitigate gradient drift caused by accumulated quantization noise, and (iii) an on-the-fly sparse sketching strategy that improves runtime memory and computational efficiency. Experiments across multiple benchmarks demonstrate that QGPM achieves state-of-the-art performance under fixed memory budgets, highlighting its effectiveness in scalable, privacy-preserving continual learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Summary and Strengths
This paper presents Quantized Gradient Projection Memory (QGPM) framework that combines basis-wise quantization, quantization-error-aware projection, and sparse sketching for memory-efficient continual learning.

### Strengths
The core idea of applying quantization to gradient subspace storage is an interesting and novel direction, as it tackles the important issue of memory accumulation in projection-based continual learning methods. The authors’ empirical evaluation is thorough, spanning several benchmarks (CIFAR-100, miniImageNet, 5-Datasets) and architectures (AlexNet, ResNet-18, ViT). The proposed method consistently achieves competitive accuracy under stringent memory budgets, demonstrating that quantizing for gradient projection can indeed mitigate forgetting while reducing storage costs.
Another appealing design choice is to retain overloaded (outlier) values in high precision instead of truncating them, which differs from conventional quantization and appears to contribute to robustness  (though, in my view, this intriguing design deserves deeper discussion). Overall, the paper is clearly written and includes extensive experiments supporting the empirical claims.

### Weaknesses
- Simplistic treatment of quantization - While the work proposes a quantization method, its treatment of quantization theory remains rather naive. The paper frequently refers to the scheme as “information-theoretically optimal,” yet the design is essentially a variant of scalar quantization. From rate–distortion theory, vector quantization is provably superior to scalar methods. There is little justification for why this scalar approach is adopted, and the omission of stochastic or vector quantization (or even a discussion thereof) weakens the claim of theoretical optimality.

- Limited insight from the quantization analysis -  The theoretical analysis (e.g., Theorem 3.1) focuses on quantizing an i.i.d. Gaussian source. However, the quantization of Gaussian sources is a well-understood topic, and textbook references (e.g., Lecture Notes on Information Theory by Polyanski and Wu) describe optimal scalar and vector quantizers for such distributions. As such, the purpose of the analysis and the insights drawn from it are not very clear. Clarifying the purpose and novelty of this analysis would improve the paper.

- Disconnect between theoretical analysis and key continual-learning metrics -  A large portion of the paper and appendices is devoted to theoretical derivations related to quantization distortion, yet the core continual-learning metrics, namely, learning performance and memory usage, are analyzed only empirically. Theoretical discussion does not connect quantization distortion to these global measures. If such an analysis is intractable, the paper should explain why; otherwise, incorporating at least approximate analytical links between quantization distortion and forgetting performance would significantly strengthen the contribution.

- Restricted and oversimplified neural network assumptions - Section 3.1 appears to base the analysis on linear layers without bias terms, which limits the generality of the conclusions. In realistic neural networks, nonlinear activations such as ReLU clip all negative activations to zero, fundamentally changing the distribution and thus the quantization distortion behavior. The authors should discuss how such nonlinearities might alter or invalidate the derived results and whether the approach extends beyond this simplified setting.



Overall,  the paper explores an appealing and practically relevant idea of quantizing gradient subspaces for memory-efficient continual learning, and backs it with solid experiments. However, the theoretical claims are overstated, and the quantization analysis lacks depth and connection to learning outcomes. Addressing these conceptual limitations, clarifying the notion of “information-theoretic optimality,” and broadening the analysis to more realistic neural architectures would be essential for the work to meet the standards of ICLR.

### Questions
See above under weaknesses

### Soundness
2

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
4

### Summary
The paper proposes new techniques to quantize the core bases stored for gradient projection method (GPM).The paper first show that the deviation between ideal and quantized orthogonal updates grows quadratically with the quantization error. The paper addresses this using 2 techniques: a) an enhanced quantization approach that improves over the NFk via centered inlier normal float Q aimed at reducing influence of outliers during Q, b) quantization error aware gradient projection to adaptively relax orthogonality constraints based on estimated deviation from desired update direction.

### Strengths
- The overall approach to enhance GPM with the new quantization approach as well as the use of sparse sketching for updates have sound reasoning and theoretical bounds as shown in the paper.
- Basis wise quantization makes sense and should have a higher precision and can help retain near orthogonality after quantization
- Experimental results and ablation studies have good coverage in terms of datasets, models and baselines.

### Weaknesses
The enhanced quantization over NFk makes sense that it is more robust to outlier values. However, there are a few concerns:
1. Distributional assumptions: codebook is derived from a standard normal. If actual data distribution as pointed out in the paper deviates (especially has skewness), the quantization can be suboptimal
2. This is amplified because of the codebook being fixed and not being adaptive to input distribution
3. Compressing tail could also increase error disproportionately for the NF4; in contrast, the CINF4 codebook construction may over emphasize tails

### Questions
What are the assumptions made on the data distribution? How does it impact the quantization performance when the distribution changes? (especially as the paper argues that the continual data has a heavy tail).

Why is the codebook kept static? What are the drawbacks?

### Soundness
3

### Presentation
2

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
This paper proposes Quantized Gradient Projection Memory, a projection-based continual learning framework that compresses stored gradient subspaces via a distribution-aware quantizer, stabilizes learning with a quantization error–aware projection rule, and reduces construction overhead using on-the-fly sparse sketching, achieving near–full precision accuracy with 4–8× memory savings on standard vision benchmarks under matched memory budgets. The work combines clear algorithmic design, theoretical analyses of quantization-induced drift and sketching guarantees, and thorough experiments that show 8-bit QGPM matches GPM while 4-bit QGPM remains competitive with substantial memory reductions, positioning QGPM as a strong, memory-efficient alternative to rehearsal under privacy constraints.

### Strengths
1.This work proposes a targeted quantization scheme CINF for GPM bases and a QEA projection rule that adapts orthogonality weights via cosine-error, addressing quantization-induced subspace distortion in a principled way.

2.This work provides theoretical insights into why NF scaling fails under heavy tails, shows quantization-induced drift accumulation, and gives sketching guarantees, with ablations that align with the theory and quantify bitwidth/outlier/QEA effects on ACC and BWT.

3.The results are highly valuable. The ability to achieve nearly full-precision performance with 4x-6x less memory makes GPM a much more viable competitor to rehearsal-based methods, especially in privacy-sensitive domains.

### Weaknesses
1.The core GPM framework still accumulates bases with each new task. While QGPM compresses these bases, the total memory still grows monotonically. The experiments are limited to 5, 10, or 20 tasks. It is unclear how QGPM's memory footprint scales in a true lifelong learning scenario (e.g., 50-100+ tasks) compared to a rehearsal method with a fixed memory buffer. A discussion of this scaling trade-off is missing.

2.The method introduces new hyperparameters that appear critical to performance, particularly the QEA scaling factor $\alpha$ and the CINF outlier proportion $p$. Table 4 shows that performance is highly dependent on $\alpha$. Table 9 indicates that $\alpha$ and $p$ are tuned for each dataset and bitwidth. This raises concerns about the practical tuning cost. A more in-depth sensitivity analysis or a more principled method for setting $\alpha$ (e.g., adapting it based on observed error statistics) would improve robustness.

3.CINF stores outliers in full precision. The paper mentions this contributes minor overhead, but this cost is never explicitly quantified. If the distributions are truly heavy-tailed, storing 1-3% (from Table 9) of vectors in FP32 could be a non-negligible memory cost that offsets the 4-bit/8-bit quantization gains. A brief analysis of this would be beneficial.

### Questions
1.Could you comment on the memory scaling of QGPM in a long-sequence setting (e.g., 50+ tasks)? While QGPM compresses bases, the number of bases still grows. Is there a crossover point where a fixed-size rehearsal buffer (like ER) becomes more memory-efficient than the growing QGPM?

2.The QEA factor $\alpha$ seems critical but requires tuning per setup. How sensitive is the method to this hyperparameter? Is there a more principled way to set $\alpha$ automatically, perhaps based on the observed mean/max quantization error, to reduce the tuning burden?

3.What percentage of the final QGPM memory footprint (e.g., in the 8QGPM column of Table 1) is consumed by the full-precision outliers stored by CINF? Is this cost consistently negligible across different models and bitwidths?

4.The sparse sketching component appears to be a strict win, reducing SVD time and intermediate memory with no performance loss (Table 12). Is this always the case? Does the $(1 \pm \epsilon)$-subspace embedding (Theorem 3.3)  ever introduce approximation errors that interact negatively with the quantization errors, or is it robust in all tested scenarios?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Quantized Gradient Projection Memory (QGPM), a memory-efficient extension of Gradient Projection Memory (GPM) for continual learning. Traditional GPM stores orthonormal subspace bases to protect previously learned knowledge by projecting gradients onto a constraint subspace. However, its memory cost scales with model size and the number of tasks.

QGPM compresses the stored subspace bases using vector quantization, then dequantizes them only during projection. The authors analyze the quantization-induced distortion and theoretically show that projection error grows with both the number of stored bases and quantization noise. To mitigate forgetting caused by this distortion, the paper introduces Quantization Error-Aware Projection, which adaptively relaxes the orthogonality constraint based on cosine-similarity loss between original and quantized bases.

Experiments across standard continual learning benchmarks demonstrate that QGPM significantly reduces memory consumption while retaining competitive performance against state-of-the-art CL methods.

### Strengths
1. Clear motivation for memory efficiency in continual learning by reducing subspace storage cost.

2. Theoretical analysis linking quantization noise to projection error, improving understanding of limitations.

3. Empirical results show that the proposed method keeps performance competitive while significantly reducing memory.

### Weaknesses
1. Computational overhead may increase due to repeated dequantization and projection steps, especially for large models.

2. Insufficient ablation study. It would be helpful to see more analysis on different quantization levels and their impact on performance.

### Questions
I don't have any questions.

### Soundness
3

### Presentation
3

### Contribution
3
