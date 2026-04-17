# HAPPI: Efficient KV cache compression with Hadamard PCA-based Power iteration

- Decision: Reject
- Scores: 2, 2, 4, 6, 6

## Abstract
Truncated Singular Value Decomposition (SVD) has recently attracted renewed attention for its effectiveness in model optimizations, such as LoRA initialization and KV-cache compression. However, exact SVD remains computationally expensive, while approximate methods like power iteration often introduce non-negligible errors. In this paper, we present Hadamard PCA-based Power Iteration (HaPPI), a new algorithm that significantly improves the accuracy of low-rank approximation while retaining efficiency. Compared to prior methods, HaPPI achieves the lowest approximation error at a practical computational cost. Building on this foundation, we further propose HaPPI-KV, which combines HaPPI with key whitening and residual quantization to deliver high compression ratios for key–value caches. By leveraging both the efficiency and precision of HaPPI, HaPPI-KV achieves state-of-the-art trade-offs between memory efficiency and model quality, highlighting the superiority of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies low-rank approximation for the purpose of LoRA initialization and compressing key-value caches in LLMs. Variations of the RSVD algorithm appear to form the baseline for methods in this space.

The paper has two main contributions:
1. Proposing a version of the RSVD algorithm that relies on a deterministic Hadamard transform. This is the HaPPi algorithm.
2. Proposing a KV-cache compression algorithm based on the HaPPi algorithm with a different data centering approach than prior methods, as well as a different place where quantization is used.

I will note that the first bullet point does not exactly match how the article describes the HaPPi method. I will return to this point later.

The paper is all empirics, without any theory.

### Strengths
I will review this paper from my perspective. I am familiar with the world of numerical linear algebra, and less familiar with the world of LLMs.

The paper has an overall clear exposition, and intriguing empirical performance. The experiments appear to have been run pretty seriously, and the description in the text of the methods is mostly clear.

The proposed method does appear to have some value in achieving better performance than prior methods. I should note that I'm not clear that the authors correctly identify why they get better performance. More on this later.

It's a little tough for me to really explain the significance of this work within the world of LLMs. I would guess that, should all of my concerns be assuaged, this paper would be helpful for the broader LLM community that relies on low-rank approximations.

The name HaPPi is very cute. I like it.

### Weaknesses
I will review this paper from my perspective. I am familiar with the world of numerical linear algebra, and less familiar with the world of LLMs.

From the perspective of numerical linear algebra, I have concerns about this paper. This is a story in parts.

## Part 1: Algorithm 1 is the RSVD algorithm _(at least in infinite precision)_

Algorithm 1 on page 3 is called "Power Iteration" and is attributed to [Kang et al. 2024]. If you expand what this algorithm is doing, it's equivalent to the RSVD algorithm proposed in [Halko et al. 2011].

> Showing my work: Let $\Omega \in \mathbb R^{n \times r}$ be the initial value that $P$ take on line 1 of the algorithm. Then, this algorithm returns $P = A^\top U$ and $Q = U$ where $U = \mathrm{orth}(A(A^\top A)^{L-1}\Omega)$, so that $QP^\top = UU^\top A$. This is the RSVD method.

This is important because on [Lines 158-178] says that Algorithm 1 significantly outperforms the RSVD method of [Halko et al. 2011]. Really, they say the outperform the pytorch `svd_lowrank` method that implements the RSVD. Either way, it should be important that Algorithm 1 is RSVD, and therefore that we are merely comparing two different implementations of the same method. This is all a precursor to the actually HaPPi method though.

## Part 2: HaPPi appears to be power method applied to a rotation of a matrix

Algorithm 2 in the paper is the titular HaPPi method. To find a rank-r approximation to a matrix A, it works in four stages:
1. Compute $A_H := AH$ where $H$ is the Hadamard transform
2. Run RSVD on $A_H^\top A_H$ for at most 2 iterations, resulting in a test matrix I will call $\Omega_0$
3. Run RSVD on $A_H$ for $L$ iterations, starting from the test matrix $\Omega_0$
4. Return $H$ times the output of line 3, unrotating it from the Hadamard space

If you have infinite precision in your floating point numbers, this algorithm is *EXACTLY EQUAL* to running RSVD on the matrix A for $L+3$ iterations.

> Showing my work: For simplicity, assume $L\geq 2$. Adopt the notation from the paper that $C_H = A_H^\top A_H$. Then $\Omega_0 = C_H \mathrm{orth}(C_H^3 \Omega)$, where $\Omega$ is the random (Gaussian) sketch matrix that the first RSVD algorithm is initialized with. We then find that bullet point 3 above results in the matrix $P = A_H^\top U$ and $Q = U$ where
$$U = \mathrm{orth}(A_H C_H^{L-1} \Omega_0) = \mathrm{orth}(A_H C_H^{L-1} C_H C_H^3 \Omega) = \mathrm{orth}(A_H C_H^{L+3} \Omega) = \mathrm{orth}(A (A^\top A)^{L+3} \Omega_H)$$
where $\Omega_H := H \Omega$ is distributed exactly at $\Omega$ if $\Omega$ is a Gaussian matrix. Hence, we output $H P = A U$ and $Q = U$, recovering the RSVD of $A$ exactly, just with $L+3$ iterations.

So, my conclusion is that any advantage that HaPPi might give you must be attributed to either A) running for 3 more iterations of subspace iteration when compared to other methods in the paper or B) running in finite precision is somehow better for the matrices the authors happen to be working with. Neither of these narrative are taken by the authors.

The authors declare [Lines 183-194] that the HaPPi method offers advantages over the standard RSVD method for reasons unrelated to the two above reasons. Instead they say that
1. [Lines 186-187] The Hadamard transform mitigates the effect of outliers on the RSVD method
2. [Lines 188-189] The initialization (bullet point 2 above; the part that produces $\Omega_0$) helps speed up the rest of the RSVD convergence.

I do not see how the first argument should hold. I agree with the second argument, but only because it's just because the preparation of $\Omega_0$ is just running RSVD for more iterations.

Seeing as the authors do not acknowledge any of these facts about their algorithm and do not analyze it as anything other than a rotated RSVD method, I'm pretty firmly against this method. Put more precisely, they offer no evidence which shows that their rotated RSVD method empirically outperforms other RSVD methods run with the same approximation rank $r$ and number of subspace iterations. 

My suspicion, which would need to be invalidated with very clear data, is that they see HaPPi outperform other methods because it has three extra steps of subspace iteration.

## Strange Errors

The paper has some other strange errors, less brutal than the one above but still important.
- [Lines 425-428] claim that Figure 6 shows information about the accuracy of certain methods. It's possible that they meant Table 2, but I also don't see the numbers they claim mention to in Table 2.
- [Lines 458-460] claim that the memory consumption of the methods used in the paper is grows linearly in the sequence length, as evidenced by Figure 6. Figure 6 clearly shows all the methods having superlinear memory consumption. It's not close to a line.

I've got other lesser qualms about the paper. They don't rise to the same standard for concern as the headline concerns above
1. Despite the use of randomized methods, no confidence intervals are given on their experimental results. This is especially since they are outperforming other methods by sub-percentage margins in accuracy (for instance see Table 1 on page 5)
2. Algorithm 3, "HaPPi with Key Whitening" on page 6, is a fairly odd algorithm. It seems to try to save time by running RSVD twice. The first time, they run only for 2 iterations on the whole data matrix. Then, they run RSVD on the (sorta) output of the previous RSVD for more iterations. It's not clear to me why they couldn't just use compute SVD factorization of the first RSVD result. _also I think line 6 of this method has a spurious transpose?_

### Questions
Given that HaPPi, in infinite precision, is exactly equivalent to RSVD run on the input matrix for 3 extra iterations, can you show that HaPPi truly outperforms RSVD run without the Hadamard transform for the exact same number of iterations?

Can you break up the SVD into the matrices $U \Sigma V^\top$? This is important on [Lines 280-290], where there are two different matrices $S$ in play.

Can you provide experimental evidence of HaPPi-KV being faster/better than just returning the RSVD of K from the get-go?

The Hadamard transform only exists for dimensions that are exact powers of 2. The usual plugin-replacement is the discrete cosine transform (DCT). Have you looked at using that transform instead, or do you always have a power-of-2 dimension?

## Other typos/recommended edits

Feel free to ignore anything here. It's just recommended edits.

1. [Algo 1 / whole paper] Cite [Halk et al] instead of [Kang et al] Rename to "RSVD" or "Subspace Iteration" or "Randomized QB Approximation"
1. [Algo 1] Reshape this to look more like standard RSVD pseudocode. See, eg, page 9 of [Halko et al]
1. [Algo 2] Write this as two calls to RSVD
1. [Fig 2] Add y-axis labels. Remove the vertical cap.
1. [Fig 2 / whole paper] Rename "SVD low-rank" to "pytorch svd_lowrank"
1. [whole paper] Use \citep instead of \cite throughout most of the paper. The lack of parentheses is really confusing.
1. [Algo 1, "Ensure" line] P is clearly a matrix, not a vector.
1. [142] this smushed parenthesis with HT in it is really unclear. Move cite to end of sentence, put (HT) right after "Transform"
1. [Sec 3.2] Mention FFT-based method for computing HT
1. [175] capitalize alg
1. [193] I'd not call this "iterative refinement" which has a kinda different vibe in the numerical linear algebra community in my opinion. I'd call these steps of subspace iteration.
1. [243] instead of building the matrix $C_H$ why not run RSVD as written on page 9 of [Halko et a], jumping back and forth between $A_H$ and $A_H^\top$
1. [equation (2)] use `\mathrm{HaPPi}` in your latex here to make it prettier
1. [468] remove "less"

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes HaPPI, a Hadamard PCA–based power-iteration algorithm designed to improve the accuracy of truncated SVD approximations while maintaining practical computational cost. The key idea is to first apply a Hadamard transform to mitigate activation outliers, then initialize power iteration using principal components of the transformed covariance matrix. Compared to existing randomized or power-iteration–based decompositions, HaPPI achieves consistently lower reconstruction error with modest FLOP overhead. The authors further introduce HaPPI-KV, a KV-cache compression pipeline that combines HaPPI with key-whitening and residual quantization. Extensive experiments across multiple LLMs and benchmarks (e.g., GSM8K, BBH, LongBench, RULER) demonstrate that HaPPI-KV achieves state-of-the-art trade-offs between cache memory reduction, inference quality, and latency. Additional evaluations on LoRA initialization and ablation analyses highlight the practical advantages of higher-precision low-rank approximations. Overall, the method targets scalable model optimization under resource constraints and presents promising empirical results.

### Strengths
S1 (Practical importance of KV-cache compression): The paper tackles a practically significant challenge in large-scale inference: KV-cache memory overhead under long-context decoding, without relying on retraining or model modification. Integrating approximate SVD, whitening, and residual quantization into a unified pipeline demonstrates strong engineering awareness and directly addresses real deployment bottlenecks.

S2 (Creative combination of known components): Although its components (Hadamard transforms, PCA-based initialization, power-iteration refinement) are individually known, their combination for on-the-fly KV compression is a creative design choice. The method is compatible with current GPU kernels, incurs modest computational overhead, and can be plugged into existing inference stacks for models such as Llama3, Mistral, and Qwen2.

S3 (Comprehensive experimental validation): The authors conduct comprehensive evaluation across diverse LLMs and benchmarks (e.g., GSM8K, AQuA, BBH, LongBench, RULER). The inclusion of LoRA initialization experiments and detailed ablation studies strengthens the claim that the proposed method improves low-rank approximation quality in realistic downstream scenarios and is not narrowly tuned to a particular workload.

### Weaknesses
W1 (Lack of theoretical justification): The paper never explains why applying an orthogonal Hadamard transform should improve power iteration convergence or reduce approximation error. Since such transforms preserve the spectrum, the improvement mechanism must be numerical or empirical, but this is not analyzed. Without this reasoning, the claimed advantage of HaPPI lacks theoretical grounding.

W2 (Experimental inconsistencies): Several experimental details are inconsistent or misleading.
• Table 2 mis-highlights HaPPI-KV as best in cells where GEAR performs better (e.g., the Multi and Code columns for Llama3-8B).
• Figure 6 does not contain the accuracy results cited in Sec. 7.3 (“91.24% at 4k tokens”), suggesting missing or mismatched figures.

W3 (Insufficient efficiency analysis): Despite claims of “comparable efficiency,” HaPPI shows higher FLOPs and latency than baseline power iteration (0.35 vs. 0.12 GFLOPs, in Fig. 2). The ablation table verifies that combining components helps, but does not isolate why—it never tests alternative orthogonal transforms or reports multi-run variability. The paper would benefit from clearer breakdowns of runtime cost and per-component contribution.

W4 (Limited conceptual novelty): The core ideas appear incremental rather than a theoretical breakthrough. Gains largely reflect a synthesis of existing techniques, and the inconsistent results make the work read more as an engineering refinement than a conceptual advance.

### Questions
Q1 (Impact of orthogonal transforms): Since the Hadamard transform is orthogonal, what theoretical reason allows it to change the convergence speed or approximation quality of power iteration?

Q2 (Task-dependent performance variation): Why does HaPPI-KV underperform baselines on several tasks (e.g., AQuA, MultiDoc QA)? Are those differences statistically significant? Could you provide additional evaluation results to more clearly demonstrate HaPPI-KV’s accuracy gains?

Q3 (Efficiency claims vs. measured latency): In Fig. 6, latency is clearly higher for HaPPI than GEAR—why do you claim efficiency parity?

Q4 (Component-wise contribution): Can you provide an error decomposition showing which component (Hadamard, PCA, whitening) contributes most to MSE reduction?

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
In this paper the authors propose a novel modification of the (blocked) power iteration method for approximate truncated SVD, called HaPPI, to improve its applications in LLMs. One of the main ideas is to use a Hadamard transform to mitigate outliers. This comes at a cost of increased compression time, indicating new trade-offs between speed and accuracy.
The main application of HaPPI is it to compress KV caches in LLMs, with a novel algorithm called HaPPI-KV. 
Experimental results indicate that HaPPI-KV can improve the accuracy on Llama3, Mistral, and Qwen2. Reducing the effect of outliers provides both lower error in the LORA approximation and better quantization. Overall HaPPI-KV shows promising results for compression rate and latency for the models tested.

### Strengths
1) KV cache size is an important problem for LLMs and compression is a crucial direction to mitigate it. 
2) The results showed in the paper are promising and could lead to great progress on the field.
3) The authors propose new ways of dealing with outliers and this brings benefits to both LLM compression error and quantization.
4) The paper is mostly well-written and concise.

### Weaknesses
The main weaknesses that I can mention are the following.
1) **Mathematical analysis**: The main algorithm is lacking some basic mathematical analysis. It would be helpful for the reader if the authors dedicate a section to mathematically justify why the algorithm works. It is hard to infer only from the algorithm description.
2) **Literature overview**: Algorithm 1 is essentially Block-power iteration (aka "simultaneous iteration"), which has been studied for decades, and it is closely related to the celebrated Block-Krylov PCA (see refs [1,2,3,4] for some recent analyses). Besides low-rank approximations, there have been major advances in terms of **full-SVD** computation. Recent works have essentially shown that the worst-case complexity of full-SVD is the same as matrix multiplication (up to logarithmic factors, see refs [5,6]). It is important to mention these related works (and potentially others that I miss).
3) **Randomness**: Hadamard transforms have recently drawn a lot of attention for quantization. However, most of the recent approaches use randomness. The main use of randomness is to avoid worst-case instances, due to the "Heisenberg-principle" (a flat signal can have outliers in the frequency domain). The idea is simple: just multiply each column of the Hadamard transform with a random sign, with probability 1/2. See for example refs [7,8,9,10]. This paper uses a deterministic Hadamard transform. This is not a bad thing (on the contrary, removing randomness is desirable), but the two variants should be compared. 

#### Minor weaknesses

4) **Performance measurements**: Measuring the overhead in FLOPS provides limited insight on compression time, and the latency has been measured only for HaPPI-KV as a whole. It would be helpful to see a breakdown of the HaPPI-KV runtime, highlighting the impact of HaPPI.
5) **Table/Figure descriptions**: In some tables/plots it is unclear what is being presented. It would be helpful to write some more details on what each legend represents, what are the "bold-face" numbers in the tables, etc.
6) **Introduction**: From the writing perspective the paper is well structured, but the introduction seems to be just a longer version of the abstract. A better introduction/background section would help the reader to understand the context of the paper better.


#### References

- [1] Musco, Cameron, and Christopher Musco. "Randomized block krylov methods for stronger and faster approximate singular value decomposition." Advances in neural information processing systems 28 (2015).
- [2] Allen-Zhu, Zeyuan, and Yuanzhi Li. "LazySVD: Even faster SVD decomposition yet without agonizing pain." Advances in neural information processing systems 29 (2016).
- [3] Sobczyk, Aleksandros, Marko Mladenovic, and Mathieu Luisier. "Invariant subspaces and PCA in nearly matrix multiplication time." Advances in Neural Information Processing Systems 37 (2024): 19013-19086.
- [4] Meyer, Raphael, Cameron Musco, and Christopher Musco. "On the unreasonable effectiveness of single vector krylov methods for low-rank approximation." Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). Society for Industrial and Applied Mathematics, 2024.
- [5] Shah, Rikhav. "Hermitian diagonalization in linear precision." Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). Society for Industrial and Applied Mathematics, 2025.
- [6] Sobczyk, Aleksandros. "Deterministic Complexity Analysis of Hermitian Eigenproblems." 52nd International Colloquium on Automata, Languages, and Programming (ICALP 2025). Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2025.
- [7] Ailon, Nir, and Bernard Chazelle. "The fast Johnson–Lindenstrauss transform and approximate nearest neighbors." SIAM Journal on computing 39.1 (2009): 302-322.
- [8] Tropp, Joel A. "Improved analysis of the subsampled randomized Hadamard transform." Advances in Adaptive Data Analysis 3.01n02 (2011): 115-126.
- [9] Tseng, Albert, et al. "QuIP $# $: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." International Conference on Machine Learning. PMLR, 2024.
- [10] Ashkboos, Saleh, et al. "Quarot: Outlier-free 4-bit inference in rotated llms." Advances in Neural Information Processing Systems 37 (2024): 100213-100240.

### Questions
I have the following questions that would help me better understand the paper and to make a more informed final recomendation:
1) The goal of using the Hadamard transormation was to avoid outliers, so why do you need to apply also key-whitening?
2) It would be interesting to see the breakdown of how much memory is saved with the LoRA approximation and how much by having better quantization due to the removal of outliers.
3) How does the method compare computationally to prior work, in particular, to the additional references that I mentioned above? (exact SVD, Block-Krylov PCA, Lazy-SVD, other?). Would it benefit to replace the Power Iteration of HaPPI with a Block-Krylov iteration?
4) How does HaPPI compare to randomized Hadamard transform variants? Can you use randomness in HaPPI?
5) What do the bold-face numbers represent in Table 2? It is a bit confusing, for example, that in some cases the bold-face value is not the highest one in the corresponding column.
6) Line 2 of algorithm 1 is redundant. Line 3 of Algorithm 2 seems also redundant. Is it?
7) In Eq. (2), it is confusing that S is used both for the optimization variable, but also for the singular valuyes of $K^\top K$. Could you explain these steps in more detail, by using different matrices?

### Additional Feedback
I can mention the following additional points which could help the authors improve the manuscript. These points are not "necessary" to make my final recommendation.
- Some figures can be placed closer to where they are referenced. Especially Fig. 1, Fig. 2, and Fig. 3 show evaluation results, that belong to the analysis sections.
- Section 3.1 should be named "Singular **Value** Decomposition" (Singular Vector Decomposition is not wrong, but it is not the "established" acronym).
- In Fig. 3, it would be helpful to have a formula for MSE
- Between lines 185-187, I think you need to apply randomness to guarantee the outlier elimination (with high probability).
- A mathematical analysis (at least a basic one) of the algorithms, e.g., to justify their correctness, would greatly strengthen the presentation.
- Step 1 in Algorithm 3 seems redundant (is it?). I think you can directly call HaPPI(K,...), but I am not 100 percent sure.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors of this manuscript propose HaPPI, an approximate Truncated SVD algorithm that uses a Hadamard PCA-based Power Iteration to improve the accuracy of low-rank approximations while retaining efficiency. Building on this, they propose HaPPI-KV, a KV cache compression method that combines the HaPPI algorithm with key whitening and residual quantization. The authors claim this combined approach achieves state-of-the-art trade-offs in memory savings and model quality, outperforming prior methods like GEAR.

### Strengths
1- The paper is well-written and easy to follow.

2- The problem this work addresses, improving the efficiency and accuracy of SVD approximations, is very important in practice due to its large applications in different fields. Specially if the authors open source their code, it can be very impactful in the field.

3- The accuracy results reported for HaPPI-KV are very good, showing consistent and significant improvements over the GEAR baseline across multiple models and benchmarks.

4- The validation of the core HaPPI algorithm on LoRA initialization (Section 5.3) is a valuable addition, demonstrating that the improved approximation quality translates to better performance on downstream tasks.

### Weaknesses
1- The theoretical justifications for the improvements in HaPPI are limited. While the paper provides intuition (Hadamard for outliers, PCA for convergence), it lacks a formal analysis of approximation error bounds or convergence guarantees compared to standard power iteration or randomized SVD. Also, an analysis of the applicability of HaPPI on larger ranks (128, 256, 512) can be helpful.

2- The major contribution of this work is the HaPPI SVD approximation method, but its analysis feels narrow. The evaluation is largely focused on KV cache tensors. More analysis on matrices with different properties and distributions (e.g., various weight matrices from real models, synthetic data with controlled rank and outlier distributions) is essential to fully analyze the benefits and limitations of this method.

3- The timing results reported (Figure 6) are only for the compression function itself and only compare HaPPI-KV and GEAR. To understand the true practical overhead, a comparison against an uncompressed FP16 or BF16 benchmark (which would have zero compression latency) is needed to evaluate the impact on end-to-end inference time.

4- A detailed time decomposition of the inference process is missing. Showing the SVD time vs. the key whitening, quantization, and the rest of the model's computation can give more insight into the practical application and bottlenecks of HaPPI-KV.

5- The order of SVD and quantization in HaPPI-KV (SVD followed by residual quantization) is not well justified beyond a single ablation study result (Table 5). A deeper analysis of why this order is superior (e.g., analyzing the distribution of the residual) is needed, especially since the baseline (GEAR) quantizes first.

### Questions
1- The ablation study (Table 5) suggests applying HaPPI before quantization is better. How would quantizing the full tensor before applying the HaPPI low-rank approximation affect model quality? Could the authors provide a deeper analysis of why quantizing the residual is more effective?

2- How does the end-to-end inference latency of a model using HaPPI-KV perform in comparison to an uncompressed FP16 or BF16 benchmark, not just the latency of the compression function?

3- Could the authors provide a time decomposition of the inference latency when using HaPPI-KV? Specifically, what percentage of the total inference time is spent on the HaPPI SVD approximation versus the key whitening and residual quantization steps and other parts of the model?

4- How does the HaPPI algorithm's accuracy (MSE) and speed compare against other approximation methods (like `torch.svd_lowrank`) when applied to a wider variety of matrices, such as the main FFN weight matrices of a model, not just the KV cache?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new SVD algorithm that improves upon the conventional power iteration method by making the following changes:
- Applies a Hadamard Transform prior to power iteration. 
- Uses the covariance matrix of the transformed tensor for initializing the power iteration. 

The paper then applies this SVD method to KV cache compression, with additional improvements specific to KV cache such as key whitening and residual quantization.

### Strengths
- A novel SVD algorithm well tailored for capturing the low-rank structure of the KV cache. 
- Strong empirical validation compared to a sufficiently wide range of other KV cache compression methods.

### Weaknesses
**[W1]** In Table 2, showing the compression ratios of different methods and configurations would strengthen the claim. This can also be presented as an accuracy vs. memory usage plot. 

**[W2]** The plotted figures are too small and therefore hard to read, especially the font sizes for the legends and axis titles. 

Minor comments that did not affect the score:
- L469: Double-check the number for GEAR’s latency measurement.

### Questions
**[Q1]** Would the strong performance hold for smaller and larger model scales? 

**[Q2]** How would the method perform for reasoning models such as the Qwen3 model family on even longer generation tasks, up to the scale of ~10k tokens? Demonstrating this would highlight the method’s robustness under long-generation scenarios, which are not captured by current CoT tasks or long-context retrieval tasks.

### Soundness
3

### Presentation
2

### Contribution
3
