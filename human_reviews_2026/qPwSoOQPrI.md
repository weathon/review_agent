# Leave it to the Specialist: Repair Sparse LLMs with Sparse Fine-Tuning via Sparsity Evolution

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Large language models (LLMs) have achieved remarkable success across various tasks but face deployment challenges due to their massive computational demands. While post-training pruning methods like SparseGPT and Wanda can effectively reduce the model size, but struggle to maintain model performance at high sparsity levels, limiting their utility for downstream tasks. Existing fine-tuning methods, such as full fine-tuning and LoRA, fail to preserve sparsity as they require updating the whole dense matrices, not well-suited for sparse LLMs. In this paper, we propose \textbf{Sparsity Evolution Fine-Tuning (SEFT)}, a novel method designed specifically for sparse LLMs. SEFT dynamically evolves the sparse topology of pruned models during fine-tuning, while preserving the overall sparsity throughout the process. The strengths of SEFT lie in its ability to perform task-specific adaptation through a weight drop-and-grow strategy, enabling the pruned model to self-adapt its sparse connectivity pattern based on the target dataset. Furthermore, a sensitivity-driven pruning criterion is employed to ensure that the desired sparsity level is consistently maintained throughout fine-tuning. Our experiments on various LLMs, including LLaMA families, DeepSeek, and Mistral, across a diverse set of benchmarks demonstrate that SEFT achieves stronger performance while offering superior memory and computation efficiency compared to existing baselines. The code is provided in the supplementary material and will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Sparsity Evolution Fine-Tuning (SEFT), a new method tailored for already-pruned large language models. It dynamically evolves the sparse topology during finetuning via a weight drop-and-grow strategy coupled with a sensitivity-driven pruning criterion, so task-relevant connections are restored while the overall sparsity budget is strictly preserved. Extensive experiments on multiple LLaMA, DeepSeek, and Mistral models across diverse benchmarks demonstrate that SEFT consistently outperforms prior baselines in both accuracy and memory-computation efficiency.

### Strengths
1. The proposed drop–grow–adapt procedure intuitively tackles the challenges of sparse fine-tuning: unlike LoRA, it allows high-rank weight updates while strictly preserving high sparsity, and its reduced memory footprint further boosts its potential for widespread adoption.
2. The manuscript is clearly written and easy to follow.

### Weaknesses
W1. Continuously updating the sparsity masks adds extra computation—especially because full dense gradients are computed during the sparse-topology-evolution phase—yet the paper does not report the actual training-time overhead versus LoRA, making it difficult to gauge the true computational cost.

W2. Although sparsification is promising, in practice weight quantization is often more useful, and state-of-the-art methods can reach 1–2-bit precision with minimal accuracy loss (roughly comparable to a 90 % sparsity level). Taking this into account, investigating whether the method can be combined with quantization would be an important consideration for practical applicability.

### Questions
Please clarify the concerns raised in W1 and W2.

Minor
• Typo at L885: “The results, presented in Table 7, show that …”.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors experiment on LLaMA, DeepSeek, and Mistral model families, pruned with both SparseGPT and Wanda. The results demonstrate that SEFT outperforms baselines like LoRA* (prune-after-tuning), SPP, and SQFT across various benchmarks (Commonsense Reasoning, MMLU, GSM8K), with a particularly strong advantage at high sparsity levels (e.g., 70%-80%). Furthermore, SEFT shows significant efficiency gains, including up to 2x memory savings during training and up to 2.5x inference speedup on a CPU.

### Strengths
SEFT cleverly adapts the idea of Dynamic Sparse Training (DST) to the novel scenario of "fine-tuning an already-sparse model." The "Drop-Grow-Adapt" three-step cycle shown in Figure 2 is clear and logically sound. The "Grow" step, in particular, which allows the model to "resurrect" pruned weights, is key to its task-adaptive capability.

he analyses in the appendix are thorough. For example, Appendix E.1 (Table 7) proves the necessity of "allowing resurrection of pruned weights," and Appendix E.2 (Fig 5a) demonstrates the superiority of the "sensitivity-based" pruning criterion over "magnitude-based." This greatly enhances the credibility of the method's design.

### Weaknesses
SEFT introduces new hyperparameters that require tuning, most notably the sparse topology "update frequency $k$" and the "drop rate." The appendix (F.3, F.4) shows that the optimal update frequency $k$ is task-dependent (e.g., 60 for C4 vs. 10 for Commonsense Reasoning). This slightly weakens the claim of a universally simple method, as this tuning adds back some of the experimental burden that fine-tuning aims to reduce.


The "Sparsity Adaptation" step (Sec 3.2) seems to re-prune the entire model at every $k$ steps using a sensitivity-based score ($|\theta_i \nabla_{\theta_i} L|$). Is this score computed on the fly, or are gradients accumulated? This step seems computationally intensive, and its cost relative to the "Drop" and "Grow" steps is not fully clarified.

### Questions
Could the authors please quantify the training wall-clock time of SEFT compared to the baselines (LoRA*, SPP, SQFT)? How significant is the computational overhead from the "Grow" and "Adapt" steps, which require dense gradient information?

In the "Sparsity Adaptation" step (Sec 3.2), is the sensitivity score $s_i = |\theta_i \nabla_{\theta_i} L|$ computed using the instantaneous gradient from the current batch, or is it an accumulated value (e.g., from an optimizer like Adam)? If it's instantaneous, how stable is this metric? If it's accumulated, what is the memory cost?

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
4

### Summary
This paper proposes Sparsity Evolution Fine-Tuning (SEFT), a new method for fine-tuning sparse large language models (LLMs). While traditional fine-tuning approaches like LoRA fail to maintain sparsity and post-training pruning methods (e.g., SparseGPT, Wanda) often cause substantial performance degradation at high sparsity, SEFT addresses these issues by dynamically evolving the sparse topology during fine-tuning while keeping the overall sparsity fixed.

### Strengths
- The paper is clearly presented and well-organized, with only a few minor typos.
- The proposed SEFT method is conceptually simple, well-motivated, and demonstrates strong empirical performance.
- The authors provide comprehensive ablation studies, including analyses of sparsity’s effect on performance, inference speed, and memory efficiency.
- Experiments are conducted across multiple datasets and diverse LLM families, which strengthens the empirical validation of the method.

### Weaknesses
- Limited Related Work Discussion:
The discussion of related work primarily focuses on low-rank methods (e.g., LoRA) but overlooks several recent sparse fine-tuning and pruning approaches that are methodologically closer to SEFT. Although Appendix A.2 and A.3 cover PEFT and dynamic sparse training, it would strengthen the paper to include recent sparse PEFT methods (e.g., [1,2,3,4]) for a more comprehensive contextualization.
- Novelty Clarification:
The paper lacks a clear explanation of how SEFT differs from similar sparse fine-tuning frameworks, particularly SpIEL [5], which also incorporate grow-and-drop phases and gradient-based parameter selection. A more explicit discussion comparing SEFT’s innovations—such as its sparsity adaptation phase or index update mechanism—with these prior works would clarify its novelty. Besides, similar to SMT[3], SEFT also introduce reconstruction indices method to maintain and train sparse model.  SEFT also have similar parameter selection apprach regards to gradient magnitude similar to SpIEL[5] and SMT[3]. More discussion about the similarity will be appreciated. 
- Missing Baseline Comparisons:
SEFT is conceptually similar to several sparse fine-tuning methods (e.g., SMT [3], SpIEL [5]), yet these are not included as baselines in the experiments. Adding such comparisons would provide a more balanced and convincing evaluation of SEFT’s performance and efficiency.
- Minor Typos:
Inconsistent spelling: “fine-tuning” vs. “finetuning” → use one form (“fine-tuning”).
“enables sparse LLMs recover” → “enables sparse LLMs to recover.”
“delta vector δ dynamically explore” → “explores.”
“Specifically models are fine-tuned…” → “Specifically, models are fine-tuned…”


[1] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

[2] Parameter-Efficient Fine-Tuning without Introducing New Latency

[3] Sparse Matrix in Large Language Model Fine-tuning

[4] Training Neural Networks with Fixed Sparse Masks

[5] Scaling Sparse Fine-Tuning to Large Language Models

### Questions
- Memory Cost:
SEFT maintains reconstruction indices to record sparsity patterns. Could the authors provide quantitative estimates of this additional memory overhead? Sparse matrices typically require extra storage for index data, potentially offsetting efficiency gains. Further, does maintaining and reconstructing these indices introduce additional copy operations or memory fragmentation?

- Training Speed:
Sparse matrix operations often involve index resolution and memory mapping, which may reduce cache efficiency and increase latency during the indices read operations. While the paper evaluates inference latency (main text and Appendix B), it lacks discussion of training time. Including an analysis of training speed and a comparison to dense fine-tuning methods would help clarify whether SEFT introduces measurable overhead during training.

This paper presents a promising and well-executed approach to fine-tuning sparse LLMs. However, its contribution would be strengthened by a deeper comparison to related sparse fine-tuning works, explicit clarification of novel aspects, and quantitative discussion of memory and training-time overhead.
If these concerns are addressed in the rebuttal, I would be willing to reconsider my current rating.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Sparsity Evolution Fine-Tuning (SEFT), a sparse fine-tuning method for large language models (LLMs) that dynamically adjusts the sparsity topology during training. SEFT employs a drop-and-grow mechanism to evolve sparse connections, removing weights with the smallest update magnitudes and reactivating previously pruned ones with the largest current gradients. This allows the sparsity pattern to adapt throughout fine-tuning while maintaining a fixed sparsity ratio. Experiments on models such as Llama and Mistral demonstrate that SEFT achieves better performance and higher efficiency than existing sparse fine-tuning baselines.

### Strengths
1. Efficient fine-tuning for large-scale LLMs is a pressing and valuable research direction. Leveraging sparsity to reduce training costs offers clear benefits for resource-constrained settings.
2. The motivation is sound and design is reasonable. Allowing the model to reconstruct its sparse topology and reactivate pruned weights provides flexibility for adapting to downstream tasks, addressing the rigidity of static pruning schemes.
3. Comprehensive experiments and evaluation settings. The experiments cover multiple open-source LLM architectures and two different pruning schemes as initialization. The analysis includes both performance and efficiency metrics, with additional ablations and sensitivity studies in the appendix, lending credibility to the conclusions.
4. Clear writing and presentation. The paper is well structured, and Figure 1 offers an intuitive comparison between SEFT and LoRA, clearly illustrating the mechanism.

### Weaknesses
1. Missing comparisons to recent baselines. The paper omits comparisons with recent methods such as SparseLoRA [1] and S^$2$FT [2], which are directly relevant. Even a conceptual or mechanism-level discussion would help position SEFT more clearly within the current literature.
2. Lack of statistical significance analysis. While SEFT often achieves the best average results (e.g., Table 2), the margins are small (~1%). Given the stochastic nature of LLM fine-tuning, reporting averages and standard deviations over multiple seeds or conducting significance tests would strengthen claims such as “SEFT consistently outperforms baselines”.
3. Insufficient discussion of computational overhead of “drop-and-grow”. The drop-and-grow updates rely on dense gradient computation (Appendix I), yet the main text lacks quantitative analysis of this cost. Although the proposed update scheme mitigates overhead, it may still limit SEFT’s deployability in low-resource environments.

---

**References**

[1] Khaki, S., Li, X., Guo, J., Zhu, L., Plataniotis, K.N., Yazdanbakhsh, A., Keutzer, K., Han, S. &amp; Liu, Z.. (2025). SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity. Proceedings of the 42nd International Conference on Machine Learning, in Proceedings of Machine Learning Research 267:29768-29783.

[2] Yang, X., Leng, J., Guo, G., Zhao, J., Nakada, R., Zhang, L., ... & Chen, B. (2024). S $^{2} $ FT: Efficient, scalable and generalizable LLM fine-tuning by structured sparsity. Advances in Neural Information Processing Systems, 37, 59912-59947.

### Questions
Could the authors quantify the time cost of the topology evolution step relative to the total training process? Would accounting for this overhead alter the comparative efficiency results between SEFT and other methods?

### Soundness
2

### Presentation
2

### Contribution
2
