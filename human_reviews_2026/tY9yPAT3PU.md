# Boosting Entropy with Bell Box Quantization

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Quantization-Aware Pre-Training (QAPT) is an effective technique to reduce the compute and memory overhead of 
Deep Neural Networks while improving their energy efficiency on edge devices. Existing QAPT methods produce models stored in compute-efficient data types (e.g. integers) that are not information theoretically optimal (ITO). On the other hand, existing ITO data types (e.g. Quantile/NormalFloat Quantization) are not compute-efficient. 
We propose BBQ, the first ITO quantization method that is also compute-efficient. BBQ builds on our key insight that since learning is domain-agnostic, the output of a quantizer does not need to reside in the same domain as its input. BBQ performs ITO quantization in its input domain, and returns its output in a compute-efficient domain where ITO data types are mapped to compute-efficient data types. Without sacrificing compute efficiency, BBQ outperforms prior SOTA QAPT methods by a perplexity reduction of up to 2 points for 4-bit models, up to 4 points for 3-bit models, up to 5 points for 2-bit models, and up to 18 points for 1-bit models. Code is available at https://github.com/1733116199/bbq.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BBQ, a novel quantization method for low-bit training of large language models. The authors improve the quantization process via adaptive scaling factors and demonstrate its performance on the LLaMA model under 1-4 bit quantization, comparing it with QuEST/LSQ methods.

### Strengths
(1) Propose the BBQ quantization method, which optimizes low-bit quantization through adaptive scaling factors  
(2) Systematically evaluate quantization performance across different bit widths (1-4 bit) on the LLaMA model

### Weaknesses
(1) Limited experimental scale and diversity: The method is tested only on LLaMA, without validation on other mainstream architectures (e.g., GPT, Bloom), making it hard to confirm its generalizability  
(2) Insufficient theoretical analysis: No in-depth explanation is provided for why BBQ performs better at certain bit widths (e.g., 3-bit), relying solely on experimental observations without theoretical support

### Questions
(1) The authors mention that "LSQ does not support 1-bit quantization and LSQ diverges when n+ e= 300M". Does BBQ also face similar limitations? Will BBQ remain effective for models with larger parameter scales (e.g., >300M)?  
(2) The paper states that "each experiment for n+ e= 300M is conducted on one Nvidia A100 80GB and lasts for 3.5 days". For larger-scale models (e.g., 1B+ parameters), will the computational cost of BBQ increase significantly? Is it still practically viable?

### Soundness
2

### Presentation
2

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
This paper presents Bell Box Quantization (BBQ), a novel quantization method that aims to achieve information-theoretically optimal (ITO) quantization while remaining compute-efficient for modern hardware. The method introduces a seven-step quantization pipeline involving the Hadamard Transform, RMS Normalization, Gaussian CDF-based Probability Integral Transform, and precision-scaled quantization operations. BBQ is applied in Quantization-Aware Pre-Training (QAPT) settings, targeting low-precision training (1–4 bits) for large-scale models such as LLaMA. Experimental results show that BBQ outperforms existing quantization schemes like LSQ and QuEST in both entropy (capacity utilization) and perplexity (model quality), while maintaining hardware efficiency. The paper further demonstrates that entropy serves as a strong proxy for quantized model quality.

The paper presents a technically sound and conceptually novel quantization framework that advances the state of low-bit quantization for large-scale models. However, its current limitations in adaptability and architecture diversity need to be addressed in future work.

Recommendation: weak accept (with added generalization experiments and ablation analysis).

### Strengths
* The introduction of ITO-based quantization combined with hardware-compatible output domains is a well-motivated and original contribution to quantized training research.
* The step-by-step quantization process (Hadamard Transform → Gaussian CDF → Uniform Quantization → Scaling) is well explained and systematically justified.
* Across multiple model sizes and bit-widths, BBQ consistently achieves lower perplexity and higher entropy compared to LSQ and QuEST, validating the proposed method.
* Using entropy as a measurable proxy for model capacity utilization is a meaningful contribution that provides interpretability beyond standard performance metrics.
* The profiling experiments demonstrate that BBQ can achieve real-world speedups on hardware supporting FP4, showing its deployment potential for large models.

### Weaknesses
- section 2.3, "learning is domain agnostic" : the two example to support this is very simplistic assumption. For eg. rotated, cropped and color jitter image is still image and in the same domain, images transformed to frequency domain is still an image but encoded in different format. In both cases, domain is still image processing. Autoencoder is a bit closer but they can't be used to train cross domain models. Authors are suggested to add more concrete examples and possibly quantification of this assumption
Limited scope of evaluation: Experiments are conducted exclusively on LLaMA-based architectures; it remains unclear whether BBQ generalizes to convolutional or vision transformer models.
* Since BBQ operates in separate input/output domains and does not minimize reconstruction error, it may not perform well in fine-tuning or post-training quantization scenarios — limiting its versatility.
* Lack of ablation analysis: The impact of individual steps (e.g., Hadamard Transform, Φ smoothness, or γ initialization) is not quantitatively analyzed, leaving unclear which components are most critical.
* Implementation details underexplored: Although latency improvements are discussed, comparisons with more hardware-optimized kernels/different GPU generations  or newer architectures like NPUs could provide stronger evidence of scalability.
* No comparison to information-theoretic quantizers beyond LSQ/QuEST: Including recent ITO-based methods (e.g., [NF4](https://chatgpt.com/c/690279a0-9b50-8320-a3bc-93decef608ce#:~:text=neural%20network%20weights.-,arXiv,%2B2,-FP4%20(4%2Dbit), [FP4](https://arxiv.org/html/2501.17116v1?utm_source=chatgpt.com), or adaptive quantizers) would strengthen the empirical claim of superiority.

### Questions
- section 3.4 : Uniform quantization and unsigned to signed conversion : Does it make any difference for ReLU based models where output activations are always positive and unsigned values are preferred for better accuracy. ? I support this sentence "we subsequently convert the positive on data to symmetric data by subtracting 2b−1 and the hyperparameter zero point z," is to take care of ReLU activation outputs ? 
*  How well does BBQ perform on architectures beyond transformers, such as CNNs or Vision transformers?
* Can the authors provide results isolating the contribution of each step (Hadamard Transform, Φ, RMS normalization, etc.) to the final accuracy and entropy gain?
* Does BBQ introduce any gradient instability during early training, especially in extremely low-precision (1–2 bit) setups?
* Could the authors discuss whether the binary search for Φ−1 values introduces any measurable latency overhead in hardware implementations?

### Soundness
3

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
5

### Summary
This paper introduces BBQ (Bell Box Quantization), an information-theoretically optimal (ITO) quantization method for quantization-aware pre-training (QAPT) of large language models. BBQ builds on prior work, such as QuEST and NormalFloat, by introducing a Gaussian CDF on the normalised Hadamard transform before applying a learnable bit-dependent scaling, resulting in a quantizer that is both ITO- and compute-efficient. The key idea is that quantization can occur in one domain while producing outputs in a different, hardware-efficient domain, thus preserving information content without losing computational advantages.

The authors conduct experiments on LLaMA models of various sizes, demonstrating improvements in quantized weight entropy and C4 perplexity over QuEST and LSQ. The proposed quantizer also supports efficient implementations on emerging low-precision formats, such as MXFP4.

### Strengths
The authors correctly note that existing QAPT methods are limited in their representation capacity, as they implicitly constrain the entropy of the quantised weights. They ground this hypothesis clearly and use it to motivate the development of the first information-theoretically optimal (ITO) quantizer for QAPT. The resulting method, BBQ, is conceptually straightforward yet effective, consistently improving over strong baselines such as QuEST and LSQ across multiple bit widths.

A key strength of the paper is its demonstration that introducing the Gaussian CDF transformation to achieve ITO quantization does not incur additional computational overhead, owing to an efficient implementation that maintains compatibility with low-precision arithmetic. 
It is also interesting to see how the authors explore and integrate emerging numerical formats, such as MXFP, highlighting the method’s forward-looking relevance to upcoming GPU architectures.

### Weaknesses
In general, the premise of the paper is sound but requires significant work in presentation, ablations and experimental results to qualify for the conference: 

* The experimentation section is lacking. I would like to see zero-shot results and one more family of models, rather than just increasing the size of the same architecture. Please include a comparison to other SOTA methods, such as ParetoQ.

* From a methodological standpoint, the contribution feels incremental relative to QuEST: most of the quantization pipeline (Hadamard transform, normalisation, uniform rounding) is directly inherited, and the main novelty — applying a Gaussian CDF and learning a scaling factor — would benefit from deeper ablations against QuEST’s corresponding components. This would make it more straightforward to determine how much of the gain comes from the CDF transformation itself.

* More results and information are required regarding inference speed-up (see questions below)

* The presentation of the paper has issues (see details in the question below).

### Questions
* Table 1: How is MXFP4 integrated into the method, given that MXFP4 is a dynamic quantisation scheme with an E8M0 scale and FP4 data?
* line 281: A smoother function can be: any experiments or reference to support this statement?
* Table 3: Please provide a description of the heads in the caption
* Table 3: It would be easier to read the table if the precision were presented as the difference from the full-precision
* The argument that quantised weight entropy is a good proxy for the final ppl is fundamental to the results section. However, there are no ablation studies or experiments to validate this hypothesis.
* Please add legends to all figures with training curves, cause they are hard to read.
* Section 4.2 requires more work: the latency overhead of the quantization function should be given as a fraction of the total latency. I would recommend showing the median latency over several runs for the full LLama models and using profiling to illustrate the proportion of the quantization kernel. In addition, it would be nice to quantify the total speed-up of the method compared to NormalFloat and then the full-precision baseline in a table
* A competing QAT method: Learnable Companding Quantization for Accurate Low-bit Neural Networks by Kohei Yamamoto introduces a similar concept where the transformation function is learnt. Although it is an older paper, it follows a similar logic. Have yοu considered comparing to that method too? They have awe-inspiring results for ResNets 
* Appendix A2: From the main text, I understood that EMA of the reciprocal of $\sigma$ is part of the method. What do you compare against here? 
* Appendix A3: I don’t think that adding all these validation losses adds much value to the paper. Expanding your results section to include more architectures and zero-shot results would be more impactful

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes to improve LLM quantized training by transforming the input to enable desired properties (entropy) in the network layers.  The quantization method aims to utilize all of the quantized levels equally often. The input quantization consists of seven steps and includes Hadamard transform, RMS normalization, probability integral transform, and so on. Even though no formal proof of equal utilization has been provided, empirical evidence shows that entropy increases compared to SOTA. Empirical evidence also indicates that perplexity is decreased, resulting in improved performance.

### Strengths
1. The paper proposes to transform the input for improved performance. This idea is novel and does lead to better-performing LLMs.

### Weaknesses
1. It is not known if just transforming the input would also lead to deeper layers, also preserving the "utilize all of the quantized levels equally often" property. Though the transformer layer norm would bias it to do just that. So it's not that big of a problem.

### Questions
1. Overall, the paper is acceptable to me. I am curious to know if there is a straightforward way to achieve something similar with non-transformer models, such as CNNs.

### Soundness
3

### Presentation
3

### Contribution
3
