# Achieving low-bit Muon through subspace preservation and grid quantization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Training Large Language Models (LLMs) faces severe memory constraints due to the increasing size of model parameters and optimizer states. The Muon optimizer, which is based on matrix orthogonalization, has recently demonstrated significant potential and offers considerable memory advantages over AdamW by utilizing only the first moment. However, how to apply memory-reduction techniques to further compress the optimizer states of Muon remains underexplored. Directly applying existing methods may encounter significant difficulties due to the orthogonalization process. In this work, we investigate the low-bit compression of Muon and systematically analyze the quantization error exacerbated by orthogonalization. We identify that the error primarily originates from the top singular subspace and the outlier patterns of moment matrix appearing across both dimensions. To address this, we propose 4-bit-Muon-GRASP (GRid And Subspace Preserving), which compresses the Muon optimizer states to 4 bits using grid quantization, while preserving the top singular subspace with minimal overhead. We evaluate 4-bit-Muon-GRASP through pre-training on LLaMA-130M, 350M, and 1.1B architectures and fine-tuning on 7B models for various reasoning tasks. Extensive experiment results show that our 4-bit-Muon-GRASP achieves accuracy comparable to full-precision counterparts while reducing training memory consumption by up to 28\%. The source code is publicly available at ~\url{https://github.com/wuhuaijin/lowbit-Muon}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of reducing the memory footprint of the Muon optimizer, which is already a memory-efficient alternative to AdamW as it only stores the first moment. The authors investigate applying 4-bit quantization to Muon's optimizer states, a non-trivial task as they demonstrate that Muon's core orthogonalization step (the Newton-Schulz iteration) significantly amplifies quantization errors.

The paper identifies that this error amplification originates primarily from the top singular subspace of the moment matrix. To solve this, the authors propose 4-bit-Muon-GRASP (GRid And Subspace Preserving).

Experiments on LLaMA pre-training (up to 1.1B) and fine-tuning of 7B models show that 4-bit-Muon-GRASP achieves accuracy comparable to its full-precision counterpart, while reducing total training memory consumption by up to 28% compared to fp32-Muon.

### Strengths
* The paper identifies the Newton-Schulz (NS) iteration as an amplifier for quantization error, and successfully pinpoints the source of the error amplification to the top singular subspace.
* The proposed 4-bit-Muon-GRASP method is a logical and novel response to the identified problem. It directly treats the "sensitive" top subspace with higher precision (8-bit) while aggressively compressing the "stable" residual (4-bit).
* The introduction of grid quantization to handle outliers in both row and column dimensions is a simple but effective technique that outperforms standard group quantization.
* The paper demonstrates that 4-bit-Muon-GRASP achieves its goal, matching the performance of the full-precision fp32-Muon optimizer on both pre-training (130M, 350M, 1.1B models) and challenging fine-tuning tasks (7B models).

### Weaknesses
1.  **Limited Pre-training Scale:** The pre-training experiments are limited to models up to 1.1B parameters. While fine-tuning is performed on 7B models, the most critical need for optimizer memory savings is during large-scale pre-training. The memory gains are also less substantial at smaller scales. Figure 6 compellingly shows projected memory savings for 3B and 5B models, but lacks the corresponding training loss curves to prove that GRASP holds up at that scale. The authors acknowledge this limitation.
2.  **Baselines for 4-bit Muon:** The main low-bit baseline, "4-bit-Muon-base", is a naive implementation. The paper's argument would be stronger if it discussed why other, more sophisticated 4-bit quantization techniques would *also* fail. Is the NS-iteration-based amplification a fundamental blocker for *all* previous quantization methods?
3.  **Computational Overhead of Subspace Preservation:** The method adds a Power Iteration step at every iteration to compute the top singular vectors. While Table 3 shows the total step time overhead is "minimal", a more detailed breakdown of the cost of this step versus the rest of the optimizer logic (including the NS iterations) would be beneficial.
4.  **Clarity of Algorithm 1:** Algorithm 1 is slightly difficult to parse. For example, the `PowerIter` function also uses `Orthogonalize(P)` (line 3) which is implemented via QR decomposition, but the main text refers to this as "column normalization", which is a bit ambiguous.

### Questions
1.  The main experiments use a rank of 1/16 for the top subspace. The ablation in Fig. 7a shows that rank matters, and 1/16 is a clear compromise between 1/8 and 1/32. How sensitive is this hyperparameter? Does it need to be re-tuned for different model sizes or architectures, or is 1/16 generally robust?
2.  The ablation in Fig. 7c shows that discarding the residual singular space leads to poor performance. This is a key finding. Does this imply that the primary insight is that Muon (due to its orthogonalization) *cannot* be approximated by a simple low-rank matrix, and *must* retain the full-rank information, even if the residual part is heavily quantized?
3.  Regarding the computational overhead, how does the cost of the single Power Iteration step compare to the cost of the 5-step NS iteration? A more granular profiling would help in understanding the trade-offs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes an approach to quantizer momentum state in Muon optimizer to 4-bit. The approach involves preserving the singular values of momentum in 8-bit while the residual components are quantized to 4-bit. The idea of subspace preserving mixed precision quantization is intuitive and the grid based quantization scheme seem to work better than group based quantization.
To my knowledge, this is the first work to perform 4-bit quantization of Muon optimizer so there aren't any related baselines to compare with. The evaluations in this work are thorough and the ablations are useful to motivate the approach.

### Strengths
1. The approach seems to work compared to baseline approach of 4-bit Muon quantization without subspace preservation.

### Weaknesses
1. Need to show results on multiple seeds.
2. Code is not provided.

### Questions
1. Do the authors have any insight on the performance of quantized 4-bit Muon v/s 4-bit AdamW optimizer? 
2. Is muon optimizer more difficult to quantize than adamw optimizer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a variant of Muon, where the optimization state M is quantized into a low-bit version. By observing that the singular vectors with large singular values will have large quantization errors after NS, the authors propose to use 8-bit quantization for the low rank matrix associated with the singular vectors with large singular values and to use 4-bit quantization for the residual part. To further decrease quantization error, the authors propose to use grid quantization. The experimental results show that the proposed low-bit Muon achieves similar performance to the 32-bit version, while costing less memory.

### Strengths
1. The authors identify that after NS iteration, the relative error of the matrix associated with the singular vectors with large singular values will be amplified a lot. 

2. Based on the observation, the authors propose an algorithm that uses different precision for different classes of singular vectors.

3. To further decrease the quantization error, the authors propose grid quantization method that scales the matrix according to the maximum absolute value in each row and each column.

4. The experimental results show a similar performance between the proposed algorithm and the 32-bit Muon.

### Weaknesses
1. Why do the authors use a single iteration of the power method? Is it enough for the algorithm to identify the top singular vector? Will it affect the performance?

2. The overall time seems to be more than the 32-bit version. How does the time compare to the CPU-offload muon, which costs 0 memory on GPU but costs more time than the 32-bit version, especially for large models and large k (the rank of Q).

3. The selection of rank will affect the performance even for 350M model. How to choose the rank for even a large model?

### Questions
See weaknesses.

### Soundness
2

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
4

### Summary
The authors propose 4-bit-Muon-GRASP, a novel approach for reducing storage requirements for Muon’s momentum state. This is important for practitioners training in memory-constrained scenarios. The authors find that directly quantizing Muon’s momentum to 4 bits leads to large errors relative to an FP32 baseline after Newton Shulz (NS) orthogonalization is applied. The authors show that these errors are mainly caused by quantizing the top singular space. Therefore, they break the quantization problem into 2 parts to reduce errors caused by NS: (1) they quantize the top singular subspace to 8 bits, and (2) they quantize the remainder to 2 bits. In their experiments spanning language model pre-training and finetuning, the authors achieve this naturally through power iteration. In their experiments, the authors show that 4-bit-Muon-GRASP nearly matched the convergence of the FP32 baseline.

### Strengths
- The paper is easy to understand and follows a coherent story. 
- The experiments provide a good evaluation of the proposed method and are well-executed. 
- The authors provide the timing and memory complexity of their method.
- The approach is intuitive.

### Weaknesses
- Due to grid quantization’s reliance on both the column and row scales, the dequantization step is not compatible with optimizer state sharding(zero-1/fsdp) out of the box, despite the combination of these methods being relevant for low-memory training. 
- While 8bit-Muon is included in Figure 6 (memory usage), I miss a comparison to it in most other performance-related experiments. It seems like 8-bit Muon is a much simpler approach than 4-bit-Muon-GRASP, so it would be beneficial for practitioners to understand how it performs.
- I miss a description of how hyperparameters were chosen? Were the optimal values an interior point of all the values searched? This is crucial to have sound results.

### Questions
- Why are the singular values all below 1 for the NS(real) in Figure 1 (d)? Have you swapped the labels by accident? If not, the plot reads like the quantized NS is closer to UV^T (singular values 1) than real NS.
- I’m not sure exactly what rank refers to in Table 1.  I assume you are referring to the number of dimensions in the top singular subspace, but the writing does not make this clear.
- In a distributed optimization setting, [1] finds that Muon trajectories can be quantized to 2 bits. It could be interesting to discuss how this relates to your work.
- What does official choice mean in Figure 2?


**Nit-picks/suggestions:**
- Figure 2:  The y-axis could show “Quantization Error RE(A,B)” without needing a title
- line 244 missing citation
- equation 8, tensors should be in mxk and nxk

[1][MuLoCo: Muon is a practical inner optimizer for DiLoCo]

### Soundness
3

### Presentation
3

### Contribution
3
