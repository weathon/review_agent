=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
##Summary

DarwinLM introduces an evolutionary search approach for non-uniform structured pruning of LLMs, where a parent model is iteratively mutated by shifting sparsity between layers, and the fittest offspring survive. A central innovation is **training-aware selection (TAS)**: lightweight multi-step fine-tuning during the search process predicts which pruned configurations will recover well after full post-compression training. The method is validated on Llama-2-7B, Llama-3.1-8B, Qwen-2.5-14B-Instruct, and the MoE model Qwen3-30B-A3B, achieving state-of-the-art one-shot accuracy and matching prior methods with 5× less fine-tuning data.

## Strengths

- **Training-aware selection is a genuine and well-validated innovation.** Figure 2 provides direct empirical evidence that lightweight fine-tuning on 10K–200K tokens correlates with full-scale fine-tuning performance (2M tokens), and Table 5 shows that TAS-selected models recover better after continued training than non-TAS-selected models with the same one-shot accuracy. This addresses a real gap in prior pruning methods that optimize only for immediate post-pruning quality.

- **Strong one-shot results with large margins.** On Llama-3.1-8B pruned to 4.6B, DarwinLM achieves 51.6 average zero-shot accuracy versus ZipLM's 45.7 (with 6B parameters) and Minitron's 36.7 (Table 2). These are not marginal gains, and they hold across Llama-2-7B (57.2 vs. ZipLM's 54.5) and Qwen-2.5-14B (59.4 vs. 56.9 uniform).

- **Compelling data efficiency.** DarwinLM (10B tokens) outperforms ShearedLlama (50B tokens) on Llama-2-7B (62.8 vs. 62.6 average), and also outperforms a re-trained ShearedLlama at the same 10B token budget (62.8 vs. 61.9 in Table 1). This is a practically significant result.

- **First structured pruning exploration for MoE architectures.** The extension to Qwen3-30B-A3B demonstrates that non-uniform sparsity outperforms uniform pruning at equivalent parameter counts (68.8 vs. 67.9 at 20B-A2B; 63.3 vs. 62.5 at 16B-A2B in Table 3), opening a new direction for MoE compression.

- **Practical deployment validation.** Table 4 provides actual throughput and memory measurements on L40s, confirming ~2× speedup and memory reduction. Table 7 further validates under the vLLM serving framework, showing DarwinLM achieves higher throughput and lower latency than ShearedLlama at similar parameter counts.

## Weaknesses

### Major:

- **Search computational cost lacks transparency, and the "8 hours on 4 consumer GPUs" claim needs substantiation.** With 200 generations, 16 offspring per generation, and a 4-step selection process training on [10K, 50K, 100K, 200K] tokens with [8, 4, 2, 1] survivors, the total training token count during search alone is approximately: 200 × (16×10K + 8×50K + 4×100K + 2×200K) ≈ 272M tokens. On 4 L40 GPUs with 2.7B-parameter models, this implies sustained training throughput of roughly 2,000+ tokens/sec/GPU, which is aggressive for full backward-pass training at this model size. The paper does not clarify whether search-phase "training" uses full gradient updates, forward-only passes, or some approximation. If the training is substantially cheaper than standard fine-tuning, this should be explicitly stated; if not, the claimed wall-clock time needs detailed justification (e.g., parallelization strategy, effective batch size, gradient accumulation). This matters because it directly affects the claimed efficiency of the overall pipeline.

- **The MoE extension is significantly limited by its search space constraints and shows only marginal gains.** Section 3.3 states that "we employ the evolutionary search within each expert MLP, and therefore keep uniform sparsity across MoE blocks." This means all experts within the same MoE layer share identical sparsity patterns—dramatically reducing the search space compared to the dense setting where every subblock is independent. The resulting improvements over uniform pruning are modest (0.9–1.1% average accuracy gains in Table 3). While the paper claims to be the "first work to explore non-uniform structured pruning in MoE architectures," the current formulation barely extends beyond per-layer non-uniformity (which is the minimum any non-uniform method provides). The restriction should be explicitly acknowledged as a limitation, and the "first" claim should be tempered accordingly.

- **The KL-divergence fitness proxy is only validated on one model.** Figure 2 demonstrates the correlation between KL-divergence after small-scale training and full-scale training performance for Llama-2-7B only. Given that the method is applied to three distinct dense model families and MoE architectures, and given that the entire search mechanism depends on this proxy being predictive, validating it on at least one additional model would substantially strengthen the claim that TAS is robust across architectures. Without this, there is a risk that the proxy's effectiveness is model-specific.

- **The multi-step TAS design is not properly ablated in isolation.** Table 5 ablates TAS on/off but does not isolate whether the multi-step progressive design (4 steps with increasing token budgets) is necessary versus a single-step selection with the same total token budget. The multi-step design adds significant complexity to the pipeline, and it is unclear whether its progressive nature is essential or if a simpler single-step approach would suffice. This is a key design choice that deserves dedicated ablation.

### Minor:

- **Proprietary fine-tuning data for MoE results.** Section 4.1 states that "For Qwen3-30B-A3B model, we also use our proprietary high-quality dataset to finetune the compressed model," but Table 3 labels the fine-tuned result simply as "10.0B" without specifying the data source. This creates ambiguity about whether the 10B-token MoE fine-tuning result is reproducible. Since the one-shot MoE results (which use only open data) already demonstrate the method's value, this could be clarified easily.

- **No analysis of why specific layers tolerate aggressive pruning.** Table 10 reveals striking heterogeneity—e.g., attention heads dropping to 0 in some layers while others retain 28 heads—but offers no insight into what makes certain layers more or less sensitive. Understanding whether sensitivity correlates with position, activation norms, gradient magnitudes, or other measurable properties would significantly elevate the contribution from an engineering success to a scientific insight, and would help practitioners anticipate pruning outcomes without running the full search.

- **No failure mode or limitation discussion.** The paper does not discuss tasks where DarwinLM underperforms. For instance, in Table 1, DarwinLM (10B) scores 45.0 on ARC-C versus ShearedLlama (50B) at 45.1; on WinoGrande, ZipLM (4.0B params) scores 58.3 vs. DarwinLM's 55.8. An honest characterization of when and why the method struggles would strengthen credibility.

### Trivial:

- The claim of "compression guarantees" in the Introduction (line 13) could be read as implying a theoretical performance guarantee, when the paper actually guarantees only that the search respects a sparsity/speed constraint. This is a minor wording issue.

## Nice-to-Haves

- **DarwinLM with 50B-token fine-tuning** to show the performance ceiling and separate search quality from data efficiency. Table 9 partially addresses this for MoE but not for the main dense model comparisons.

- **Broader generation benchmark evaluation** beyond GSM-8K (Table 17), which shows very low absolute scores (~3%). Instruction-following or chat benchmarks (e.g., MT-Bench, AlpacaEval) would better substantiate the deployability claims.

- **Cross-model sparsity pattern analysis** to test whether structural insights from one model family transfer to another, which would increase the scientific value of the discovered architectures.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Compression guarantees" is misleading (Harsh Critic):** In context, the paper uses this to mean the search guarantees adherence to a target sparsity/speed constraint, which is a legitimate and standard usage. It does not claim a theoretical bound on accuracy loss. Removed as a substantive weakness.

- **"One-shot" definition ambiguity (Positive Reviewer):** The term "one-shot pruning" is standard in the literature and means no iterative pruning-retraining cycles. The OBS weight correction is part of the pruning step itself, not a retraining phase. This is not a genuine confusion point. Removed.

- **Limited ablation studies (Harsh Critic, citing external reviews):** The paper contains extensive ablations in the appendix—search metric (Table 20), number of offspring (Table 21), number of sparsity levels (Table 22), finetuning tokens (Table 23), pruning methods (Table 24), and TAS (Table 25). The claim that "only one ablation study" exists is factually incorrect. Removed.

- **No error bars / statistical significance (Harsh Critic):** For LLM-scale experiments with 10B-token fine-tuning runs, multi-seed evaluation is prohibitively expensive and not the norm in the field. The one-shot results are deterministic given fixed calibration data. This is a nice-to-have at best. Moved to nice-to-have.

- **Database step-size contradiction (Harsh Critic):** The critic claims a contradiction between "step size for attention differs from that of MLPs" and "consistent across all levels and modules." However, Section 3.1 explicitly states that mutations only swap sparsity within the same module type ("we sample whether to mutate MLPs or attention modules, which means the mutation only happens in the same blocks"). Since mutations are type-restricted and step sizes are uniform within each type, the speedup constraint is preserved. The wording could be clearer, but there is no logical contradiction. Downgraded to trivial.

- **Unfair comparison with baselines using different training data (Harsh Critic, citing external reviews):** The paper already re-trains ShearedLlama with the same 10B-token budget on the same data (Fineweb-Edu) for a controlled comparison (Table 1, ShearedLLaMA 10B†). The Minitron/Flextron comparison is limited by their closed datasets, which the paper acknowledges. This is partially addressed. Moved to minor/trivial.

- **Limited novelty of individual components (Harsh Critic, citing external reviews):** While individual components (OBS, evolutionary search, KL-divergence) are not novel, the specific integration—training-aware selection within an evolutionary search for structured pruning—is genuinely novel. Novelty in combination is a valid form of contribution when the combination produces results that individual components cannot. This is more appropriately reflected in the overall assessment than as a standalone weakness.

- **Parser artifacts in tables (Harsh Critic):** Per instructions, formatting artifacts are parser issues, not paper problems. Removed.

- **Missing related works:** Per hard rules, removed.

## Novel Insights

The most interesting emergent observation from the reviews and paper is the **decoupling between one-shot quality and trainability**. Table 5 shows that DarwinLM with and without TAS achieves nearly identical one-shot accuracy (55.1 vs. 54.5), but after training on just 1B tokens, the gap widens to 58.8 vs. 58.1. This suggests that two pruned architectures with equivalent immediate performance can have meaningfully different recovery trajectories—a finding with implications beyond this specific method. It implies that pruning methods optimizing only for post-pruning accuracy (the vast majority) may be systematically selecting suboptimal architectures for downstream use, and that the community should adopt training-aware evaluation as a standard practice.

## Suggestions

- Provide an explicit compute budget breakdown: total GPU-hours for database generation + search + fine-tuning, with a per-phase FLOP estimate, so readers can verify the efficiency claims against baselines.

- Add a single ablation comparing multi-step TAS (4 steps, [10K, 50K, 100K, 200K] tokens) versus single-step TAS (1 step, 360K tokens = same total budget) to justify the progressive design.

- Extend Figure 2 to at least one additional model (e.g., Llama-3.1-8B) to validate that KL-divergence as a fitness proxy generalizes across architectures.

- Acknowledge the MoE uniform-expert constraint explicitly as a limitation and discuss whether per-expert non-uniform pruning is feasible (e.g., by expanding the database to include per-expert sparsity levels).

- Add a brief paragraph analyzing the discovered sparsity patterns (e.g., in Table 10) in terms of layer depth, activation statistics, or other measurable properties, to move from "the search found this" to "here is why the search found this."

## Axis Assessment

- **Novelty:** Moderate-to-high. Training-aware selection during evolutionary search is a genuine and well-motivated innovation. Individual components are not novel, but the integration produces results that prior combinations could not. The MoE extension is novel in scope but limited in execution.

- **Technical soundness:** Moderate. The core algorithm is sound and well-described, but the search computational cost claim lacks sufficient detail for verification, and the fitness proxy is validated on only one model. The multi-step TAS design lacks isolated ablation.

- **Empirical support:** Strong for dense models (consistent gains across three model families, multiple baselines, and controlled comparisons). Weaker for MoE (marginal gains, limited baselines, constrained search space).

- **Significance:** High. Achieving SOTA structured pruning with 5× less training data has immediate practical impact. The training-aware selection insight could influence how the community evaluates pruning methods broadly.

- **Clarity:** Good overall. The method is well-structured and the writing is clear. The step-size discussion in Section 3.2 could be clearer, and some experimental details (MoE fine-tuning data source, search compute breakdown) need clarification.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 4.0, 6.0]
Average score: 5.5
Binary outcome: Reject
