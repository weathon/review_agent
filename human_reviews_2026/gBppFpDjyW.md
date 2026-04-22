# Communication Efficient LLM Pre-Training with SparseLoCo

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Communication-efficient distributed training algorithms have received considerable interest recently due to their benefits for training Large Language Models (LLMs) in bandwidth-constrained settings, such as across datacenters and over the internet. Despite reducing communication frequency, these methods still typically require communicating a full copy of the model's gradients—resulting in a communication bottleneck even for cross-datacenter links. Furthermore, they can slightly degrade performance compared to a naive AdamW DDP baseline. While quantization is often applied to reduce the pseudo-gradient's size, in the context of LLM pre-training, existing approaches have been unable to additionally leverage sparsification and have obtained limited quantization. In this work, we introduce SparseLoCo, a communication-efficient training algorithm for LLMs that effectively leverages error feedback with Top-k sparsification and 2-bit quantization to reach extreme sparsity as low as 1–3%  while outperforming full-precision DiLoCo. Our key observations are that outer momentum can be locally approximated by an error feedback accumulator combined with aggressive sparsity, and that sparse aggregation can actually improve model performance. We empirically demonstrate in a range of communication-constrained LLM training settings that SparseLoCo provides significant benefits in both performance and communication cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a communication-efficient training algorithm, called SparseLoCo. By combining ideas of error feedback and compression, it can reach a low sparsity while outperforming DiLoCo baseline. Through comprehensive experiments, the authors further established the choice of sparsity hyperparameters, communication benefits, and scalability.

### Strengths
- The paper has clear motivation (Section 4.1) and comprehensive experiments, with a detailed hyperparameter search table.
- The proposed method, SparseLoCo outperforms DiLoCo when having a smaller pseudo-grad size.

### Weaknesses
- Both error feedback and gradient compression are already existing approaches, which raises concerns about novelty. It would be preferable to provide motivation and experimental justification for why this specific gradient compression scheme was selected.
- The paper fails to offer a reasonable explanation for why the method works. For example, on why SparseLoCo can outperform DiLoCo, the author refers to recent model-merging work in multi-task fine-tuning. But the setting in this work is pre-training, which should suffer from less accurate gradient information. And SparseLoCo introduces a new sparsity hyperparameter search space, which may limit its usage.

### Questions
1. Could you provide a possible explanation on why there's an optimal sparsity location in Figure 1? Is it related to the quantization scheme used? Is the optimal value is affected by the model size as well?
2. Comparing results from Table 3 and 5, does the improvement diminish as the model size grows larger?
3. Is it possible to provide ablation on the `Sparse` part of SparseLoco? How does the top-k contributes to the training overall?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose SparseLoCo, which combines error feedback with TOP-k sparsification and 2-bit quantization to reduce communication volume to 1–3% of the original gradient, while maintaining performance superior to full-precision DiLoCo.

### Strengths
The authors achieve an exceptionally high gradient compression ratio, which reduces communication overhead, and conduct a detailed comparison with DiLoCo, demonstrating that their method maintains performance even under high sparsity.

### Weaknesses
- While the paper presents interesting ideas, its novelty compared to DeMO could be more clearly highlighted. It would be helpful if the authors more explicitly elaborated on the specific advancements beyond DeMO.
- The current writing could be further polished to improve overall readability and flow.

### Questions
- In subsection 3.3, the authors describe how SparseLoCo divides tensors into chunks and applies Top-k to each. Could you kindly provide more details on how this process is integrated into the framework?
- The terms "Density" and "Communication Density" are used without clear definitions.
- It would be valuable to know more about the concrete impact on training time—are there any quantitative results available?
- There seems to be a minor inconsistency: in the sentence "Concretely, the aggregation step in Algorithm 1 Line 14," "Line 14" may refer to "line 13" instead.
- Adding a more detailed comparison with DeMO, such as including its results in Figure 1, would strengthen the experimental evaluation.
- Could the authors provide a more comprehensive explanation of the theoretical convergence guarantees?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates communication-efficient training algorithms and proposes a method named SparseLoCo. The algorithm incorporates the error-feedback technique, specifically by communicating the top-k error updates across machines. Experiments conducted on the LLaMA model demonstrate the efficiency of the proposed approach.

### Strengths
The paper is well-written and the method is clearly presented. The experiments convincingly validate that the proposed approach effectively reduces communication overhead.

### Weaknesses
I have major concerns regarding the novelty and experimental validation of this work, which I find to be incremental.

1. Limited Novelty: The core technique of error-feedback is a well-established standard for achieving communication efficiency. The paper does not convincingly demonstrate a significant algorithmic advancement beyond this.

2. Insufficient Evidence for Acceleration: The experiments fail to prove that the method accelerates standard LLM training in practical settings (e.g., on 8 or 32 GPU platforms). Crucially, there is no plot showing the training loss against wall-clock time, which is essential for validating actual speedup.

3. Lack of LLM-Specific Innovation: While focused on LLM training, the method appears to be a generic application of sparsification and error-feedback, with no special design or adaptation for Transformer architectures.

Given the incremental nature of the contribution, I would expect a more compelling validation, such as a high-quality code package and solid experiments demonstrating clear wall-clock time acceleration in standard LLM training benchmarks.

### Questions
1. How does the proposed method compare to the standard baseline in terms of competitive wall-clock time for model convergence on a common scale, such as an 8-GPU platform?

2. Does the algorithm incorporate any domain-specific optimizations tailored for the Transformer architecture, or is it a generic approach?

3. To better demonstrate generality, could the authors show the method's performance on other model families, such as ResNet or GPT-2?

### Soundness
2

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
3

### Summary
This paper introduces SparseLoCo, a communication-efficient algorithm designed for distributed LLM pre-training in bandwidth-constrained environments. The method addresses the limitations of some communication algorithms, which still suffer from transmitting large, full-precision pseudo-gradients. SparseLoCo innovatively combines DiLoCo's multi-step local updates with aggressive compression, integrating TOP-k sparsification (achieving densities as low as 1-3%) and 2-bit quantization. The core insight is that DiLoCo's global outer momentum can be effectively replaced by a local error feedback (OuterEF) accumulator, which naturally manages the local pseudo-gradient updates. This unification allows SparseLoCo to drastically reduce communication volume while simultaneously outperforming the full-precision DiLoCo baseline in terms of final model performance and communication-loss trade-offs.

### Strengths
1. The paper presents a novel and highly practical solution (SparseLoCo) that effectively unifies two distinct lines of communication-efficient training: infrequent communication (like DiLoCo) and aggressive gradient compression (sparsification + quantization). This is a significant contribution for training in highly bandwidth-constrained settings, such as cross-datacenter or internet-based collaboration.

2. The core insight that DiLoCo's global outer momentum can be successfully replaced by a local error feedback (OuterEF) mechanism is a strong and non-obvious contribution. This discovery is what enables the integration of aggressive sparsification, as the EF buffer naturally manages the compression error—a problem that previously hindered attempts to combine DiLoCo with sparsification.

3. The method is backed by compelling empirical evidence. SparseLoCo not only matches but outperforms the full-precision DiLoCo baseline in terms of final loss and achieves a superior position on the communication-loss Pareto frontier. The ability to do this with extreme compression levels (e.g., 1-3% density and 2-bit quantization) is impressive and demonstrates the practical viability of the approach.

### Weaknesses
1. The paper would benefit from a brief theoretical analysis clarifying the role of the local inner updates ($H$ steps) in the optimization process. The authors compare SparseLoCo to DeMo, which (despite also updating non-dominant information locally) provides a theoretical justification that convergence can be achieved even with minimal global information, as long as it represents the dominant components of the momentum. SparseLoCo explicitly designs a local inner loop before processing and communicating dominant components. Therefore, the impact of these inner loops on global convergence, and how they interact with the OuterEF and sparse aggregation, is a key aspect that deserves further theoretical exploration. Some brief analysis may help to understand the effect of $H$.

2. The experimental setting for Table 3 is not clearly described in the text, making it difficult to ascertain the model scale and training configuration used for these downstream benchmarks. Furthermore, to sufficiently demonstrate generalizability, the evaluation would be more convincing if it included a wider array of tasks. Given that the paper reports zero-shot accuracy, a discussion on the method's applicability and performance in the fine-tuning regime would also be a valuable and relevant addition, but this is not explored.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
