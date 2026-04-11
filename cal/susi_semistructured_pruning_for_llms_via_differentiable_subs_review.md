=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary
This paper introduces SUSI, a semi-structured pruning method for LLMs that learns N:M sparsity masks via Weighted Reservoir Sampling and the differentiable Gumbel-Top-K trick. The core innovation reduces the parameter complexity of learnable mask methods from combinatorial to linear in the group size M. Experiments on OPT models (125M–1.3B) show SUSI achieves competitive or superior perplexity and zero-shot accuracy compared to strong baselines while using fewer trainable parameters.

## Strengths
- **Novel and efficient algorithmic formulation:** The application of Weighted Reservoir Sampling (WRS) and Gumbel-Top-K to reformulate mask learning as differentiable subset sampling reduces the number of trainable parameters from \(O(\binom{M}{N})\) to \(O(M)\). This is a clear and significant improvement over prior learnable-mask approaches like MaskLLM, especially for larger sparsity patterns (e.g., 2:8, 4:8), where SUSI remains tractable while MaskLLM becomes infeasible (Figure 3, Tables 6-7).
- **Comprehensive and rigorous empirical evaluation:** The paper provides extensive experiments across three OPT model sizes, multiple sparsity patterns (2:4, 2:8, 4:8), and a suite of benchmarks (WikiText-2 perplexity and six zero-shot tasks). Results consistently show SUSI outperforms strong baselines (SparseGPT, Wanda, MaskLLM) in perplexity while matching or exceeding average accuracy (Tables 1, 2, 6, 7). Ablation studies, robustness analysis (high mask overlap across seeds), and data-efficiency plots further substantiate the method’s design choices and stability.
- **Strong reproducibility:** The paper includes a detailed reproducibility statement, public code, full hyperparameters, and clear descriptions of datasets and evaluation metrics, aligning well with conference standards.

## Weaknesses
### Major:
- **Lack of hardware efficiency validation:** The paper motivates semi-structured pruning for “hardware-optimized inference” and “efficient deployment,” but provides **no measurements of actual inference speedup, latency reduction, or throughput improvement** on hardware that supports N:M sparsity (e.g., NVIDIA Ampere GPUs). Without such metrics, the practical acceleration claims remain unsubstantiated, undermining a key contribution. (Section 4, Abstract)
- **Limited model and task diversity:** Evaluation is confined to the OPT model family (125M–1.3B), which is relatively outdated. The brief extension to Qwen2.5 and Llama3.2 in Appendix A.8 shows notable performance gaps, indicating the method’s generalization to modern, diverse architectures remains unproven. This limits the significance of claims about being a “practical solution for compressing LLMs.” (Section 4.1, Appendix A.8)

### Minor:
- **Incomplete comparison landscape:** While SUSI is compared against established baselines (SparseGPT, Wanda, MaskLLM), more recent efficient pruning or sparse training methods (e.g., AST (Huang et al., 2025), cited but not benchmarked) are absent. This makes it difficult to assess SUSI’s standing relative to the latest state-of-the-art.
- **Theoretical justification could be deeper:** Theorem 1 establishes equivalence between the WRS-based variational objective and the exact distribution, but the paper lacks an intuitive or theoretical analysis of *why* subset sampling leads to better-performing masks compared to the full categorical parameterization of MaskLLM. The benefits appear empirical rather than principled.

### Trivial:
- **Formatting artifacts:** Minor OCR issues (e.g., garbled layer names in Figure 5) do not hinder understanding.

## Nice-to-Haves
- **Ablation on calibration data size:** Systematically varying the amount of calibration data (e.g., from 1B tokens down to 10M) would help characterize the method’s robustness and data efficiency more thoroughly.
- **Statistical reporting across runs:** Reporting standard deviations or confidence intervals for downstream task accuracy (beyond mask overlap) would provide a clearer picture of performance variability.
- **Analysis of pruned weight characteristics:** Investigating what types of weights (e.g., magnitude, outlier dimensions) SUSI selects compared to heuristic methods could yield insights into why the method works.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength:** “The paper is well-written” – Removed as generic.
- **Weakness:** “Evidential: The core claim of ‘minimal computational cost’ is not substantiated” – Partially kept as a major weakness regarding missing hardware measurements, but the original phrasing overstated the issue; the paper does demonstrate parameter and data efficiency gains (Figure 3).
- **Weakness:** “Incomplete ablation on calibration data sensitivity” – Weakened to a nice-to-have, as the paper includes ablations on power term and annealing; varying calibration data size is not a standard requirement for this type of work.
- **Weakness:** “Out-of-distribution generalization not addressed” – Removed as scope creep; the paper evaluates on standard benchmarks, and OOD robustness is not a stated goal.
- **Weakness:** “The claim of ‘minimal computational cost’ is not supported” – Redundant with major weakness on hardware validation; the paper does show reduced parameter count and data efficiency.
- **Weakness:** “Reproducibility details are incomplete” – Removed as the paper provides code, hyperparameters, and dataset details, meeting typical reproducibility standards.

## Suggestions
- **Include hardware performance metrics:** To solidify the practical contribution, report actual inference speedup (e.g., tokens/sec) and/or memory footprint reduction on hardware with native N:M support (e.g., NVIDIA A100). If direct measurements are infeasible, explicitly discuss this as a limitation and temper acceleration claims.
- **Expand evaluation scope:** Strengthen claims of generality by including experiments on at least one more recent and diverse model family (e.g., Llama 3 7B or Mistral 7B) and a broader set of tasks (e.g., reasoning benchmarks like MMLU). The extension in Appendix A.8 should be moved to the main paper and analyzed in depth.
- **Deepen comparative analysis:** Compare SUSI against 1–2 very recent (2025) semi-structured or learnable pruning methods (e.g., AST) to better position its contributions relative to the current state-of-the-art.

**Overall Assessment:** The paper presents a novel and technically sound algorithmic contribution that meaningfully improves the parameter efficiency of learnable semi-structured pruning. The empirical evaluation is thorough within the OPT family and demonstrates consistent performance gains. However, the absence of hardware acceleration measurements and the limited model diversity significantly weaken the practical significance and generalizability claims expected for a top-tier conference. With revisions addressing these gaps, the paper could become a strong candidate for acceptance.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
