# Highly Efficient and Effective LLMs with Multi-Boolean Architectures

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Weight binarization has emerged as a promising strategy to reduce the complexity of large language models (LLMs). Existing approaches fall into post-training binarization, which is simple but causes severe performance loss, and training-aware methods, which depend on full-precision latent weights, adding complexity and limiting efficiency. We propose a novel framework that represents LLMs with multi-kernel Boolean parameters and, for the first time, enables direct finetuning LMMs in the Boolean domain, eliminating the need for latent weights. This enhances representational capacity and dramatically reduces complexity during both finetuning and inference. Extensive experiments across diverse LLMs show our method outperforms recent ultra low-bit quantization and binarization techniques.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MBOK, a binarization framework that trains LLMs directly in the Boolean domain with multiple Boolean kernels, avoiding latent weights. It achieves near-FP16 performance with lower memory and computation compared to existing binarization and quantization methods.

### Strengths
1. The paper addresses an important problem in LLM binarization and makes a convincing case for reducing the reliance on latent full-precision weights.
2. The proposed approach is technically sound and well-motivated, with clear formulations and ablation studies that support the design choices.
3. Experimental evaluation is fairly comprehensive, showing competitive or even near-FP16 performance with much lower memory and computation cost.
4. The presentation is clear, with tables and figures that make it easy to compare against strong baselines.

### Weaknesses
1. The paper claims Pareto-frontier results, but only compares against 2\3-bit quantization without presenting a full Pareto curve, which makes the claim less convincing.
2. Although the method is said to optimize directly in the Boolean domain, it is unclear whether training/finetuning actually reduces memory consumption, or if fake quantization is still used; concrete evidence of memory savings during optimization is missing.
3. The comparison with BinaryMoS seems unfair: three Boolean kernels likely consume more memory than three MoS experts, yet the paper does not account for this discrepancy.
4. No results are reported on real inference acceleration or end-to-end memory reduction; claims about efficiency remain theoretical.
5. Large-scale validation is missing — while OPT and LLaMA-13B are evaluated, experiments on larger models (e.g., 65B) are absent, leaving scalability uncertain.
6. The distinction between the proposed multi-Boolean kernels and existing multi-binary-base methods (e.g., BitStack, QBB, DB-LLM) is not sufficiently clarified, weakening the novelty claim.
7. The initialization strategy (SVID) is largely borrowed from OneBit, which reduces the originality of that component.

### Questions
Please refer to the weakness section.

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
This work proposes a multi-boolean-kernel QAT method for LLMs. Extending BOLD’s single-kernel design, it integrates successive SVID and adaptive mixed-precision, outperforming prior binarized QAT methods. It also enables direct boolean domain LLM fine-tuning to save training resources.

### Strengths
1. This work is supported by solid theoretical foundations, with a detailed explanation of the computational process of multi-boolean-kernels during training.
2. The approach of training only the last kernel is insightful, and the authors have provided detailed experimental comparisons and explanations to validate it.
3. Experimental results demonstrate that MBOK outperforms previous binarized QAT methods in both performance and resource efficiency.

### Weaknesses
1. The novelty of this work is limited, as its core contribution is merely a natural extension of BOLD, expanding from a single kernel to multiple kernels.
2. The authors claim that MBOK lies on the Pareto frontier. However, the evidence provided in Table 3 only compares it with low-bit PTQ methods such as RTN and GPTQ. Given that MBOK is a QAT method, a fairer evaluation would involve comparisons with advanced low-bit QAT methods like SpinQuant and EfficientQAT.
3. SpinQuant with W4A8 quantization can achieve better results (as shown in Table 1 of their paper) than MBOK with 4 kernels (i.e., W4A16). If compared under the same activation bit-width, SpinQuant may hold even greater advantages, which would undermine the scalability of multi-kernel designs (the core contribution of this work).
4. Key ablation studies are missing. Specifically, analyses are needed on how initialization via successive SVID impacts final distillation performance, and on the effectiveness of the kernel allocation strategy (i.e., mixed precision) compared to uniform precision under the same kernel budget. Including such ablation results would strengthen the work’s contribution and novelty.
5. Several typos and errors require correction. For instance, in Table 1, BiLLM also employs higher-bit salient weights. In Table 3, "QPTQ" should be corrected to "OPTQ".

### Questions
1. Is there any comparison of actual training time to demonstrate that MBOK is more efficient than other binarized QAT methods?
2. For other questions, please refer to the weaknesses.

I would be willing to reconsider my rating if the authors can address these concerns.

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
This paper proposes MBOK (Multiple Boolean Kernels), a novel framework for training and fine-tuning large language models entirely in the Boolean domain, without relying on floating-point latent weights. The key idea is to represent each weight matrix as a sum of multiple Boolean kernels, each with distinct binary weights and scaling factors, thereby improving representational capacity while keeping computation highly efficient. The authors further introduce a successive SVID-based extraction method to transfer knowledge from full-precision models and a dual-level knowledge distillation strategy for refinement. Extensive experiments on the OPT and LLaMA families show that MBOK achieves performance close to FP16 with only 1 to 2 bits per weight, surpassing state-of-the-art binarization and ultra-low bit quantization methods in both perplexity and zero-shot accuracy, while greatly reducing memory and computational costs.

### Strengths
The paper presents a creative sound approach to Boolean-domain training for large language models. The proposed MBOK framework is conceptually clear and well-motivated, trying to address key limitations of existing binarization methods by removing reliance on floating-point latent weights. The methodology is thoughtfully designed, combining multiple Boolean kernels, knowledge distillation, and adaptive kernel allocation. Experiments are extensive and carefully executed, demonstrating consistent improvements over strong baselines in both efficiency and accuracy.

### Weaknesses
While the proposed Boolean-domain framework is novel, the evaluation is somewhat limited. The experiments mainly focus on older model families such as OPT and LLaMA1/2, leaving uncertainty about the method’s effectiveness on newer architectures (e.g., Qwen3). Moreover, the discussion on training stability, convergence behavior, and scalability to very large models (70B and above) is also relatively brief, making it difficult to assess the robustness of the approach under real deployment conditions. To achieve extremely high training efficiency, the authors adopted a strategy of fine-tuning only the last Boolean kernel (Section 6.1.2). While experiments show this is the most efficient strategy, this approach may limit the model's ultimate performance.

### Questions
1. Could the authors evaluate the proposed MBOK framework on more recent and competitive open-source models, such as Qwen3, LLaMA3.2, to demonstrate its generalization across architectures?
2. Can the authors provide additional analysis or experiments on the training stability and convergence characteristics of Boolean-domain optimization, especially for larger models (e.g., 70B parameters and beyond)?
3. The authors have shown that fine-tuning only the last Boolean kernel achieves the best efficiency-performance trade-off. Could the paper include a deeper analysis explaining why joint optimization of multiple kernels leads to degraded performance, and whether this trend holds consistently across larger models or more complex tasks?
This is an interesting work, and if you can address my questions, I would be happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript proposes a new framework, MBOK, for efficiently compressing and fine-tuning LLMs entirely in the Boolean domain. Unlike prior binarization or quantization methods that rely on full-precision latent weights and gradient approximations, MBOK introduces a multi-kernel Boolean structure based on BOLD, and further integrates strategies such as adaptive bit-budget allocation and knowledge distillation. Experimental results show that the proposed method achieves competitive performance compared to other contemporary approaches.

### Strengths
1. The manuscript is the first to eliminate the dependency on latent weights in LLMs through the use of BOLD, and achieves strong compression and training results by combining multi-level weight decomposition, global bit allocation, and knowledge distillation strategies.

2. The discussion of the technical approach is thorough and detailed, with extensive comparisons to similar compression methods such as BinaryMoS and BitStack.

3. MBOK demonstrates excellent empirical performance, outperforming other methods in both efficiency and accuracy even under existing hardware conditions.

### Weaknesses
1. The innovation is somewhat ambiguous. Techniques such as BOLD, SVID, knowledge distillation, and bit allocation have all appeared in prior works. For example, approaches similar to SVID and bit allocation were already explored in BitStack.

2. The paper’s structure and organization are somewhat confusing. It is unclear why bit allocation is presented at a separate hierarchical level compared to SVID and knowledge distillation. Conceptually, bit allocation and SVID seem to belong to the architectural design of MBOK, while knowledge distillation is more of a training strategy, which should logically be applied after bit allocation rather than before it.

### Questions
1. How are the benefits and stability of knowledge distillation demonstrated or quantified?

2. Why does BitNet collapse with extremely high perplexity in the reported results, and are there comparisons on broader benchmarks such as ARC-Easy or MMLU?

3. In Figure 9, how can the model allocate more than 20 kernels under an average bit budget of 3.5 bits?

4. When fine-tuning only the last kernel, what are the approximate scaling values of each kernel at convergence? Does the last kernel ever become significantly more dominant than earlier ones?

5. Can the proposed approach be extended to activation quantization, or is it limited to weight compression only?

### Soundness
3

### Presentation
3

### Contribution
3
