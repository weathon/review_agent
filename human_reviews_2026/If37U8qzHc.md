# Towards Reversible Model Merging For Low-rank Weights

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Model merging aims to combine multiple fine-tuned models into a single set of weights that performs well across all source tasks. While prior work has shown that merging can approximate the performance of individual fine-tuned models for each task, it largely overlooks scenarios where models are compressed into low-rank representations$\textemdash$ either through low-rank adaptation (LoRA) or post-training singular value decomposition (SVD). We first demonstrate that applying conventional merging methods to low-rank weights leads to severe performance degradation in the merged model. Motivated by this phenomenon, we propose a fundamentally different approach: instead of collapsing all adapters into one set of weights, we construct a compact basis (e.g., an equivalent of holding two or more models) from which original task-specific models can be recovered via linear combination. This reframes merging as generating a reconstruction-capable model space rather than producing a single merged model. Crucially, this allows us to ''revert'' to each individual model when needed, recognizing that no merged model can consistently outperform one specialized for its task. Building on this insight, we introduce our method, Reversible Model Merging (RMM), an efficient, data-free, and flexible method that provides a closed-form solution for selecting the optimal basis of model weights and task-specific coefficients for linear combination. Extensive experiments across diverse datasets and model scales demonstrate that RMM consistently outperforms existing merging approaches, preserving the performance of low-rank compressed models by a significant margin.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper relaxes the model merging problem into a compression–reconstruction setting. Based on this setting, this paper proposes a simple reversible model merging method that provides a closed-form solution to the optimization-based compression objective. The experiments demonstrate that the proposed method can maintain relatively good performance while significantly reducing the number of stored parameters.

### Strengths
1. The proposed method is simple and easy to implement.
2. The paper is clearly written and easy to follow.

### Weaknesses
1. This paper relaxes the model merging problem by assuming that the task index is known during inference. Compared with the conventional setting of model merging, this setting is overly simplified and reduces the practical applicability of the method.
2. The paper lacks evaluations on the latest models.
3. There is no comparison with model merging approaches based on LoRA, such as [1].
4. The experimental results appear underwhelming — with 69% of the parameters retained, the performance drops by more than 10%, which seems difficult to justify.

[1] Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering, ICLR 2025.

### Questions
1. A more appropriate baseline could be a comparison with quantized LoRA methods, which also aim to reduce storage requirements.

### Soundness
1

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
4

### Summary
This paper proposes Reversible Model Merging (RMM), a new method for combining multiple low-rank models (e.g., LoRA adapters). Instead of creating a single, often poorly performing, merged model, RMM learns a compact shared basis from which individual task-specific models can be accurately reconstructed on demand. This approach offers a tunable trade-off between storage and performance, and empirical results show it significantly outperforms existing merging techniques on compressed models.

### Strengths
* Effectively addresses the poor performance of standard merging techniques on low-rank models by reframing the problem as model reconstruction rather than aggregation.
* Demonstrates significant performance gains over established baselines across multiple model architectures (RoBERTa, ViT), tasks, and compression ranks.
* Offers an efficient solution for managing numerous task-specific models, with storage costs growing sublinearly with the number of tasks.

### Weaknesses
*   **Insufficient Discussion of Practical Overheads:** The primary trade-off of RMM is performance vs. storage, but another key factor is computational overhead at inference time. The paper does not quantify the latency introduced by the reconstruction step, which must be performed for each task. This could be a non-trivial cost, especially in latency-sensitive applications. Furthermore, the framework relies on an "oracle router" to select the correct task index `i`, and the practical feasibility and computational cost of this routing mechanism are not discussed.
*   **Lack of Connection to Intrinsic Dimensionality:** The core idea of RMM—finding a low-dimensional basis that can represent multiple task-specific updates—is closely related to the concept of intrinsic dimensionality in fine-tuning, e.g., [1]. Research has shown that the updates required for fine-tuning often lie in a very low-dimensional subspace. A discussion of how RMM relates to this concept and potentially a comparison to methods that explicitly estimate and use this intrinsic dimensionality could provide deeper insight into why RMM is effective and place it in a broader theoretical context.
*   **Limited Baselines for "Separate Merging":** The paper argues against "Combined merging" on the grounds of storage inefficiency and proceeds to compare RMM only against baselines using the "Separate merging" strategy. While the efficiency argument is valid, the empirical case for RMM would be stronger if it included a performance comparison against a "Combined merging" baseline (even if just for a smaller model or layer). This would help disentangle how much of the baselines' failure is due to the inherent flaws of their merging logic versus the limitations imposed by the separate merging of low-rank factors.
*   **Clarity and Precision of Mathematical Formulation:** While the overall method is clear, the notation in Section 5 could be more precise. The formulation for RMM is presented generically using `x_i` and `X` without explicit indices for the layer or the specific row/column being processed. Since the algorithm is applied independently to each row of matrix `A` and each column of matrix `B` for every layer, incorporating this into the notation would make the description less abstract and easier to map directly to the implementation.

[1] Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Reversible Model Merging (RMM), a model merging method which creates a compact shared basis where individual merged models can be reconstructed. The experiments show that RMM outperforms several merging methods while being more storage efficient.

### Strengths
1. Novel perspective of relaxing the merging problem to allow maintaining more than just a single merged model.
2. Diverse task suites and model types in the experiments.

### Weaknesses
1. Limited comparison. Many modern SVD-based merging methods show good performance merging LoRA checkpoints (e.g. KnOTS [1], ISO [2], DRM [3]), as well as other non-SVD methods (e.g. PEFT’s cat [4]), though they aren’t compared or at least discussed.
2. Practical relevancy of merging low-rank compressed checkpoints is not clearly illustrated, reference to previous works supporting this is also lacking.
3. Natural baselines such as “combined” merging the checkpoints then compress, are not discussed or compared.

[1] https://arxiv.org/abs/2410.19735
[2] https://arxiv.org/abs/2502.04959v3
[3] https://arxiv.org/abs/2505.23117
[4] https://huggingface.co/blog/peft_merging

### Questions
1. I'm not quite convince regarding the relevancy or importance of “reversible.” If the individual checkpoints are already compressed, they should already be suitable for storage. Or if merged checkpoints need to be reversible, why merge them in the first place?
2. Usefulness of the storage reduction is debatable. As the merging is done post-training, the merged weight delta can simply be summed into the base weight, no extra parameters are to be kept.
3. Why are the tasks inconsistent between models? (QNLI, MRPC, SST-2, MNLI, QQP, RTE. CoLA, STS-B for RoBERTa vs. QNLI, MRPC, SST-2, MNLI, QQP,  BoolQ for OPT). What are the rationale for picking these subsets?


Writing
1. “an important limitation remains largely overlooked: even the most sophisticated merging strategies consistently fail to match the performance of the original individual finetuned models.” is not necessarily appropriate. As it’s not overlooked, but more of a goal the community is making progress toward.
2.  “This failure stems from two key factors: the limited expressive capacity inherent in low-rank representations, and misalignment of task-specific subspaces, each optimized independently, leading to severe interference when combined” needs further elaboration or citation.

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
3

### Summary
This paper proposes a new framework for merging multiple fine-tuned models that have been compressed using low-rank methods such as LoRA or post-training SVD. Traditional model merging techniques often fail when applied to low-rank models, leading to severe performance degradation due to limited representational capacity and task interference. To address this, the authors introduce Reversible Model Merging (RMM), which redefines merging as constructing a compact, reconstruction-capable model space rather than producing a single merged model. RMM learns a small set of shared basis components and task-specific coefficients through a closed-form SVD-based optimization, allowing each original model to be approximately reconstructed when needed. Experiments on multiple NLP and vision benchmarks show that RMM consistently outperforms existing merging methods like Task Arithmetic, TIES, and DARE, achieving much higher accuracy while maintaining efficient storage. The approach offers a tunable trade-off between performance and memory by adjusting the number of basis components, scaling efficiently to large multi-task or federated learning settings.

### Strengths
### Novel framing of the merging problem:
The paper redefines model merging as constructing a reversible and reconstruction-capable subspace rather than producing a single merged model. This is a clear conceptual shift from prior approaches that treat merging as a one-shot averaging process.

### Theoretical formulation with closed-form solution:
Unlike heuristic or sign-based methods (e.g., TIES, DARE), RMM provides a principled SVD-based framework that explicitly minimizes reconstruction error, offering mathematical interpretability and data-free optimization.

### 	Compatibility with low-rank compression:
The method operates directly on low-rank representations (LoRA, PT-SVD), addressing a practical gap where previous merging techniques fail due to limited expressivity and subspace misalignment.

### Flexible trade-off between performance and storage:
The introduction of the basis size p as a tunable hyperparameter enables a controllable balance between reconstruction fidelity and memory cost—an important design for scalable multi-task or federated learning scenarios.

### Weaknesses
### Lack of clarity in reconstruction generalization:
The paper does not clearly analyze whether the learned basis generalizes to unseen tasks or domains. RMM’s reversibility is validated only for known task models, which may restrict its applicability to dynamic or continual learning settings.

### Scalability validation remains limited in model scale:
While Section 6.3 demonstrates sublinear storage scaling across tasks, the evaluation is confined to relatively small models such as RoBERTa-base and OPT-1.3B. It remains unclear whether RMM maintains the same efficiency and reconstruction fidelity on larger-scale architectures. Validation on more recent and larger models (e.g., Qwen2.5-3B/7B or Qwen3-8B) would better demonstrate the robustness and scalability of the proposed method in realistic multi-billion-parameter settings.

### Lack of evaluation on modern LLM benchmarks:
The evaluation focuses primarily on classical NLP tasks (e.g., GLUE) and small-scale models. However, compared to recent works such as DARE, which include results on AlpacaEval, GSM8K, and HumanEval, this paper lacks validation on contemporary instruction-following or reasoning benchmarks that better reflect current LLM capabilities. Including such benchmarks would strengthen the empirical evidence for RMM’s effectiveness in modern large-scale settings.

### Questions
Tables 1 and 2 present the relative storage cost, but have you also measured the actual GPU memory consumption during inference or model reconstruction? Including such results could provide stronger evidence of RMM’s practical efficiency.

### Soundness
3

### Presentation
3

### Contribution
3
