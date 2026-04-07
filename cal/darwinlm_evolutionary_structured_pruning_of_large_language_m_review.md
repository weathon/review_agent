=== CALIBRATION EXAMPLE 67 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "DarwinLM: Evolutionary Structured Pruning of Large Language Models" accurately reflects the contribution. However, the abstract contains at least two questionable claims that require scrutiny.

First, the claim that DarwinLM "surpasses ShearedLlama while requiring 5× less training data" is technically accurate, but the comparison is confounded by data quality: DarwinLM trains on Fineweb-Edu (filtered to ≥0.9 quality score, yielding an 80B-token curated corpus), while ShearedLlama trains on RedPajama. The abstract does not flag this data-quality confounder.

Second, the abstract claims this is "the first work to explore non-uniform structured pruning in MoE architectures." But Section 3.3 explicitly states: "we employ the evolutionary search within each expert MLP, and therefore **keep uniform sparsity across MoE blocks**." The non-uniformity is only between the MLP components of individual experts, while the sparsity allocation across different MoE layers is kept uniform. This substantially weakens the "non-uniform" MoE claim. He et al. (2024) and Li et al. (2025) are cited but the comparison of what is and is not "non-uniform" is not clearly argued.

---

### Introduction & Motivation

The introduction is well-organized and the problem is clearly motivated. The distinction between structured and unstructured pruning, and the rationale for non-uniform allocation, are both clearly stated.

The contributions list is appropriately focused. However, the claim in the intro that pruning Llama-3.1-8B to half its size yields "5.9% higher average zero-shot accuracy relative to the best prior method (ZipLM)" obscures an important asymmetry: in Table 2, ZipLM is evaluated at **6B parameters** while DarwinLM operates at **4.6B** — a 30% difference in remaining model size. ZipLM is operating at a much milder compression ratio. Presenting this as a direct superiority comparison is misleading and should be clearly addressed.

---

### Method / Approach

**Evolutionary Search Design (Section 3.1):** The search algorithm is a (1+λ)-ES variant where only a single parent is maintained. This design choice is neither justified nor discussed. Maintaining a diverse population is a well-established technique in evolutionary computation to avoid local optima; the authors' claim that "the fittest model so far is most likely to produce even fitter offspring" is an assumption, not an argument, and no ablation studies test this design choice. This is a non-trivial methodological decision that should be addressed.

**Training-Aware Selection (TAS):** The core innovation. The motivation in Figure 2 is built on 16 offspring trained up to only 2M tokens to establish the correlation with "full training." But the actual post-training budget in experiments is 10B tokens — 5,000× larger. The paper's only justification for TAS transferability across this large scale gap is the Figure 2 scatter plots at 2M tokens. These plots show reasonable but noisy correlations, and the paper does not report correlation coefficients. Whether the surrogate at 200K tokens genuinely predicts quality after 10B tokens is the central empirical question and is never directly validated.

**Mutation Operator:** The "level switch mutation" is intuitively sound, but there is no ablation comparing it to alternatives (e.g., random sparsity resampling, crossover between two good candidates). The constraint that "the mutation only happens in the same blocks" (i.e., MLP↔MLP, Attention↔Attention) is a design choice with no ablation support.

**Second-Order Pruning (Section 3.2):** The formulation in Equations (3)–(4) is standard OBS-based structured pruning, and the adaptation is clearly presented. The decision to avoid pruning K and V matrices in GQA models (Llama-3.1, Qwen-2.5) is reasonable but the impact on inference speedup is not characterized. GQA K/V matrices are already small, so this choice likely has minimal accuracy impact, but no ablation is shown.

**MoE Extension (Section 3.3):** The decision to omit attention pruning for MoE models is justified by "majority of parameters are located in the expert MLPs." This is qualitatively reasonable for models like Qwen3-30B-A3B but should be supported by a parameter count breakdown. More importantly, the claim that DarwinLM performs "non-uniform" MoE pruning is inconsistent with the stated design: uniform sparsity is enforced across MoE blocks, and only within-expert MLP dimensions are non-uniformly pruned. The paper should be more precise about what dimension of non-uniformity is exploited.

---

### Experiments & Results

**Unfair Baseline Comparisons (Tables 1 and 2):** This is the most significant methodological concern.

- *Table 1*: DarwinLM (2.7B) is compared against ZipLM (4.0B), a 48% larger model. DarwinLM's "superior" performance at 57.2 vs. ZipLM's 54.5 should be qualified by this size disparity.
- *Table 2*: DarwinLM (4.6B) is compared against ZipLM (6B), again 30% larger. The introduction's "5.9% higher accuracy" claim applies to this comparison.
- The paper is correct that ZipLM's dynamic programming formulation targets a specific compute budget rather than a fixed parameter count, but the evaluation protocol should acknowledge and control for this.

**Data Quality Confounder:** DarwinLM uses the Fineweb-Edu dataset filtered to quality score ≥0.9. The authors do helpfully include "ShearedLlama (10B†)" trained on the same Fineweb-Edu data (Table 1), which is a fair comparison. This achieves 61.9 vs. DarwinLM's 62.8 — a much smaller gap (0.9 points) than the headline comparison (62.8 vs. 62.6 for ShearedLlama 50B). The authors should acknowledge that the advantage over ShearedLlama is partly due to better training data, not purely search quality.

**Proprietary MoE Fine-Tuning Data (Section 4.1):** The paper states "For Qwen3-30B-A3B model, we also use our proprietary high-quality dataset to finetune the compressed model." The MoE fine-tuned result of 69.7 average (Table 3) is therefore **not reproducible** and should not be presented as a central result without clear disclosure. This is explicitly noted in the Reproducibility Statement but should be flagged more prominently in the main text.

**Misleading Comparison with From-Scratch Models (Table 9):** The paper compares DarwinLM-16A2B (initialized from a strong 30B pretrained model, then fine-tuned for 10B tokens) against DeepSeek-MoE-base 16A2B trained from scratch on 2T tokens and DeepSeek-V2-Lite 16A2B trained on 5.7T tokens. This comparison is not meaningful: DarwinLM's starting point (Qwen3-30B-A3B) has already distilled enormous knowledge from pretraining; it is trivially expected to outperform a model trained from scratch at a much smaller scale with fewer total FLOP. The table should be presented as a practical utility comparison, not as evidence of superior learning efficiency.

**TAS Ablation (Tables 5 and 25):** The full ablation (Table 25) shows DarwinLM with TAS achieves 55.1 vs. 54.5 without TAS in one-shot (a 0.6-point difference over 8 tasks). After 1B tokens of training, the gap is 58.8 vs. 58.1 — still less than 1 point on average. Given that TAS is the paper's primary methodological novelty, this modest advantage is somewhat underwhelming. The ablation is only done at 1B tokens, not at the full 10B token scale where results are primarily reported — a significant omission. Whether TAS actually improves the final 10B token result is not directly ablated.

**Statistical Significance:** Differences of 0.2–0.9 average accuracy points (e.g., 62.8 vs. 62.6 in Table 1) are reported without variance estimates or multiple runs. Given the variance of zero-shot evaluations on these benchmarks, such small differences are not reliably significant.

**Missing Ablation — Search vs. Non-Uniform Allocation:** The paper does not include a baseline that performs evolutionary search *without* the training-aware selection component and is fine-tuned at the full 10B token budget. Table 5 only compares at 1B tokens. Without this, it's impossible to disentangle whether TAS or simply better non-uniform allocation (any search method) drives the improvement over ShearedLlama.

**Speedup Analysis (Table 4):** The paper reports 1.98× throughput improvement for 2.7B vs. 7B, which is sub-linear. This is attributed to "fixed inference overheads," but no breakdown is given. For the 4.6B model (8B→4.6B, a 42% reduction), the speedup is only 1.35×. This sublinear scaling raises practical questions about the real-world utility of the method for moderate compression ratios.

---

### Writing & Clarity

Section 3.1 uses the term "population" to describe a strategy that maintains only a single parent — this is confusing for readers familiar with evolutionary algorithms. The actual strategy is a (1+λ)-ES, which should be named as such.

The implementation detail that the search uses "10× L40 GPU workstation" (Section 4.1) is inconsistent with the introduction's claim that "the pruning and search complete in 8 hours on 4 consumer-grade GPUs." This discrepancy must be resolved.

The description of the MoE approach in Section 3.3 is very brief (one paragraph) despite being listed as a major contribution. The relationship between "evolutionary search within each expert MLP" and the overall level-switch mutation framework is not clearly explained.

---

### Limitations & Broader Impact

The Limitations section is entirely absent from the main paper. The Ethics Statement covers broader impacts at a surface level but does not discuss:

- The method's dependence on the quality of the layer database (including the database size, which can be substantial for large models — not analyzed anywhere)
- The assumption that KL-divergence at a few hundred thousand tokens reliably predicts quality at 10B tokens (this is the key unvalidated assumption of TAS)
- Scenarios where the evolutionary search fails to improve over uniform pruning (e.g., Table 5 shows a tiny gap; Table 3's MoE results show marginal improvements)
- The method's overhead in terms of GPU-hours for the full pipeline (search + fine-tuning), which the paper under-reports

The reproducibility statement's admission that MoE fine-tuning data is proprietary is important and should be more prominent.

---

### Overall Assessment

DarwinLM is a technically sound and empirically broad paper on non-uniform structured pruning for LLMs, combining second-order pruning, evolutionary search, and training-aware offspring selection. The one-shot results are strong and the integration of lightweight fine-tuning into the search process is well-motivated. However, the paper has several significant weaknesses at ICLR's bar. The central innovation — training-aware selection (TAS) — shows only marginal gains (< 1 point average accuracy) in the ablation study, and the ablation is not conducted at the full 10B-token training scale. The headline comparisons against ZipLM are not parameter-matched (DarwinLM is 30–48% smaller yet presented as superior), and the advantage over ShearedLlama is partly attributable to a higher-quality training dataset. The MoE contribution is limited by proprietary fine-tuning data, the "non-uniform" framing is overstated given that MoE-block-level sparsity remains uniform, and the comparison with from-scratch DeepSeek models in Table 9 is misleading. These issues do not invalidate the paper, but they substantially temper the claimed contributions and require the authors to reframe comparisons more carefully, add the missing 10B-token TAS ablation, and address the parameter-count mismatch with ZipLM. In its current form, the paper falls short of ICLR acceptance without these revisions.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces DarwinLM, an evolutionary search framework for training-aware structured pruning of Large Language Models (LLMs). The method employs a fitness function based on KL-divergence augmented by multi-step lightweight fine-tuning to predict post-compression performance during the search. Experiments demonstrate state-of-the-art accuracy on Llama, Qwen, and MoE architectures while significantly reducing post-training data requirements compared to prior structured pruning methods.

### Strengths
1.  **Superior Empirical Performance:** The paper presents strong experimental results across multiple model scales (Llama-2-7B to Qwen-3-30B). Table 1 and Table 2 demonstrate that DarwinLM consistently outperforms SOTA baselines (ShearedLlama, ZipLM, Minitron) in average accuracy, particularly when comparing models with similar parameter counts but different training budgets (e.g., DarwinLM uses 10B tokens vs. 50B for ShearedLlama).
2.  **Training-Aware Search Design:** Unlike methods that optimize purely for one-shot performance, DarwinLM incorporates a multi-step fine-tuning selection process (Section 3.1). Figure 2 provides empirical evidence that early-stage lightweight training correlates well with full fine-tuning performance, allowing the selection of a substructure that generalizes better after recovery fine-tuning.
3.  **Extension to MoE Architectures:** The method successfully extends to Mixture-of-Experts models (Qwen-3-30B-A3B), achieving the reported first non-uniform structured pruning results for this architecture type (Table 3). This demonstrates the flexibility of the evolutionary framework beyond dense Transformers.
4.  **Hardware Relevance:** The paper provides concrete memory and throughput measurements on real hardware (L40s), confirming that structured sparsity translates to practical inference speedups (~2x throughput, ~2.4x memory reduction for Llama-2) without requiring specialized sparse execution hardware.

### Weaknesses
1.  **Computational Cost of Search:** While inference is faster, the training-aware search process is computationally intensive. Section 4.1 states the search takes 8 hours on 4 consumer-grade GPUs plus a fine-tuning cluster for 10B tokens. Compared to lighter methods like ZipLM (which uses dynamic programming), the evolutionary overhead may be prohibitive for rapid iteration or resource-constrained settings.
2.  **Population Diversity Limitations:** Section 3.1 notes that the search maintains a "single model as our population" (single parent + mutated offspring) and relies on mutation. This limits exploration of the search space compared to broader population-based evolutionary strategies. The algorithm might converge to local optima regarding sparsity allocation more easily than methods exploring wider topologies.
3.  **Fitness Metric Sensitivity:** The reliance on KL-divergence on a small calibration dataset (4096 tokens) as a primary fitness proxy is standard but potentially fragile. While Table 20 compares KL-divergence vs. PPL in the Appendix, there is limited analysis on how sensitive the search is to the specific calibration dataset composition, which could bias sparsity allocation towards certain task distributions.
4.  **Comparison Fairness:** Some comparisons, such as against ShearedLlama, favor DarwinLM partly due to different fine-tuning data budgets. While this highlights data efficiency, a more direct comparison under *identical* fine-tuning budgets (beyond the re-run in Section 4.2) would strengthen the claim of inherent structural superiority versus retraining efficacy.

### Novelty & Significance
The work is **novel** in its specific combination of evolutionary search with training-aware fitness evaluation specifically tailored for structured pruning of *modern* LLMs (post-2020 architectures). While evolutionary pruning exists (e.g., EvoPress), the application of multi-step lightweight fine-tuning to guide this search in the context of LLM recovery is a distinct contribution. The significance lies in the practical reduction of downstream fine-tuning costs required to restore model accuracy after aggressive pruning, effectively lowering the barrier to deploying compressed LLMs. The extension to MoE models further broadens its applicability in the current hardware landscape.

### Suggestions for Improvement
1.  **Cost-Benefit Analysis:** Provide a more detailed analysis of the "time-to-accuracy" trade-off. Specifically, quantify the total compute cost (search + fine-tuning) vs. accuracy to demonstrate the return on investment compared to methods with cheaper search phases but potentially lower one-shot performance.
2.  **Population Strategy:** Investigate and ablate the impact of using a population size >1 during the evolutionary search. A larger population (e.g., 10-20 parents) could improve exploration and might reveal if the current "single parent" constraint is unnecessarily limiting.
3.  **Calibration Sensitivity:** Include an ablation study on the calibration dataset size or distribution (e.g., random vs. task-specific data). This would solidify the claim that the fitness metric is robust and not overfitting to a specific calibration setup.
4.  **MoE Specifics:** Clarify the constraints on MoE pruning further. Specifically, discuss why attention modules are omitted for MoE but included for dense models (Section 3.3), and analyze if this choice impacts speedup potential differently for MoE inference compared to dense inference.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **MoE fine-tuning results are incomplete** — Table 3 shows one-shot MoE results but claims about 10B token fine-tuning recovery lack comparison to existing MoE pruning baselines, undermining the claim this is "first work to explore non-uniform structured pruning in MoE architectures."

2. **Fair comparison with Minitron/Flextron is absent** — The paper dismisses comparison due to "closed fine-tuning dataset" but claims state-of-the-art; without reproducing their method on open data or providing equivalent training conditions, the SOTA claim is not verifiable.

3. **Llama-3.1-70B lacks fine-tuning validation** — Table 11 reports only one-shot results for the 70B model while smaller models claim fine-tuning recovery; this inconsistency undermines the scalability claim.

4. **Search cost analysis missing** — The paper claims efficiency (8 hours on 4 GPUs) but provides no breakdown of search vs. fine-tuning cost compared to baselines like ZipLM or ShearedLlama, making efficiency claims difficult to verify.

5. **No inference latency on real deployment hardware** — Table 4 reports throughput on L40s but no comparison to ShearedLlama under identical serving conditions (vLLM results in Appendix are insufficient), weakening the "end-to-end speed improvements" claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Training-aware selection correlation analysis** — Figure 2 shows motivation but no quantitative analysis of whether KL-divergence after 200K tokens actually predicts final 10B token performance across different models; without this, TAS is not justified.

2. **Sparsity distribution interpretation missing** — Table 10 shows searched sparsity patterns but no analysis of why certain layers are pruned more aggressively; without this, the method appears as black-box search rather than principled compression.

3. **Search stability analysis absent** — No discussion of whether evolutionary search converges consistently across random seeds; if results vary significantly, the claimed improvements may be search artifacts.

4. **Failure case analysis missing** — No discussion of when DarwinLM underperforms (e.g., WinoGrande in Table 1); without acknowledging limitations, the method's reliability is unclear.

5. **KL-divergence vs. perplexity justification weak** — Table 20 shows similar performance but no analysis of why KL-divergence was chosen or when it might fail; this undermines the fitness function design.

### Visualizations & Case Studies
1. **Layer-wise sparsity heatmap** — Visualize which layers receive high vs. low sparsity across different models to reveal whether the search finds consistent patterns or model-specific solutions.

2. **Fine-tuning recovery curves** — Plot accuracy vs. training tokens for DarwinLM vs. ShearedLlama to show whether the claimed 5× data efficiency holds throughout training or only at convergence.

3. **Search convergence plot** — Show fitness improvement over generations to demonstrate the evolutionary search actually finds better solutions rather than random walk.

### Obvious Next Steps
1. **Complete MoE fine-tuning comparison** — The MoE contribution is a key novelty claim but lacks fine-tuned comparison to uniform MoE pruning; this should have been in the main results.

2. **Reproduce Minitron/Flextron on open data** — If closed datasets prevent comparison, authors should have reproduced these methods on Fineweb-Edu to enable fair evaluation.

3. **Add standard deviation across seeds** — ICLR expects reproducibility; single-run results for evolutionary search are insufficient to claim reliable improvements.

# Final Consolidated Review
## Summary

DarwinLM introduces an evolutionary search framework for non-uniform structured pruning of LLMs, incorporating training-aware selection (TAS) that uses lightweight fine-tuning within the search process to predict post-compression recovery potential. The method constructs a layer-wise sparsity database using second-order pruning, then searches over sparsity configurations via mutation operations while selecting offspring based on fine-tuning performance at progressively larger token budgets. Experiments demonstrate strong results on Llama-2-7B, Llama-3.1-8B, Qwen-2.5-14B-Instruct, and MoE architectures (Qwen-3-30B-A3B), with the method achieving competitive accuracy using significantly less fine-tuning data than prior work.

## Strengths

- **Strong empirical performance with demonstrated data efficiency**: The method achieves 62.8 average accuracy on Llama-2-7B after 10B tokens of fine-tuning, surpassing ShearedLlama trained on 50B tokens (62.6). The fair comparison using identical training data (ShearedLlama 10B† at 61.9 vs DarwinLM at 62.8) isolates the structural search contribution from data quality effects, demonstrating genuine improvement from the search methodology.

- **Training-aware selection is well-motivated and empirically grounded**: Figure 2 provides evidence that KL-divergence after lightweight fine-tuning (10K-200K tokens) correlates with performance after 2M tokens, justifying the multi-step selection process. The ablation in Table 25 shows DarwinLM with TAS achieves 58.8 vs 58.1 without TAS after 1B tokens, demonstrating that TAS provides measurable benefits for selecting training-amenable structures.

- **Successful extension to MoE architectures**: Table 3 shows DarwinLM consistently outperforms uniform pruning for MoE models (68.8 vs 67.9 at 19B parameters, 63.3 vs 62.5 at 16B parameters), demonstrating the method's applicability beyond dense transformers to modern architectures where parameter efficiency is particularly critical.

- **Practical inference speedups confirmed on real hardware**: Table 4 demonstrates concrete benefits—1.98× throughput increase and 2.43× memory reduction for Llama-2-7B—validating that structured sparsity translates to real deployment advantages without specialized sparse inference hardware.

## Weaknesses

- **Parameter size asymmetry in ZipLM comparisons undermines fair evaluation**: Tables 1 and 2 compare DarwinLM at 2.7B parameters against ZipLM at 4.0B, and DarwinLM at 4.6B against ZipLM at 6B—parameter differences of 48% and 30% respectively. The introduction's claim of "5.9% higher average zero-shot accuracy" for the Llama-3.1-8B comparison does not acknowledge that ZipLM operates at significantly higher remaining capacity. While ZipLM's design targets compute budgets rather than fixed sizes, the presentation should clearly contextualize these size differences rather than presenting the results as direct superiority.

- **Core methodological contribution (TAS) is validated only at 1B tokens, not the full 10B token scale**: The ablation in Tables 5 and 25 evaluates TAS at 1B tokens, showing modest gains (58.8 vs 58.1, ~0.7 points). The main results report performance at 10B tokens, but whether TAS actually improves the final 10B-token outcome remains untested. The central empirical claim—that early-stage training quality predicts post-recovery performance—needs validation at the scale where final results are reported.

- **MoE "non-uniform" claim is misleadingly framed**: Section 3.3 states "we employ the evolutionary search within each expert MLP, and therefore keep uniform sparsity across MoE blocks." The abstract's claim of "first work to explore non-uniform structured pruning in MoE architectures" obscures that the non-uniformity is only within expert MLPs, not across the MoE layer structure. This should be stated precisely.

- **MoE fine-tuned results (Table 3, 69.7 avg) depend on proprietary data**: Section 4.1 states the Qwen3-30B-A3B fine-tuning uses "our proprietary high-quality dataset," making the MoE fine-tuning results non-reproducible. While acknowledged in the Reproducibility Statement, this limitation should be prominent in the main results discussion.

- **Single-run results without statistical significance**: Reported differences are often small (e.g., 62.8 vs 62.6 in Table 1, 58.8 vs 58.1 in Table 25). Given the inherent variance in zero-shot benchmark evaluation, these margins may not be statistically significant without multiple runs.

- **Speedup is sublinear relative to compression ratio**: Table 4 shows 1.98× throughput improvement for 2.43× parameter reduction, and only 1.35× speedup for the 4.6B model. The gap between theoretical and practical speedup warrants discussion of inference overheads that limit real-world efficiency gains.

## Nice-to-Haves

- Population diversity ablation (comparing single-parent vs. multi-parent evolutionary strategies) could strengthen the methodological design but is not required given the strong empirical results.

- Calibration dataset sensitivity analysis would help establish robustness of the fitness function, though the method already uses standard KL-divergence on small calibration sets.

- Fine-tuning recovery curves (accuracy vs. training tokens) comparing DarwinLM and ShearedLlama throughout training would clarify whether the claimed 5× data efficiency is consistent across the training trajectory or only at convergence.

## Removed Points

These points are flagged to be removed, treat them with caution:
- **"Population" terminology complaint**: The harsh critic objects that Section 3.1 "uses the term 'population' to describe a strategy that maintains only a single parent." This is standard evolutionary strategy terminology—(1+λ)-ES is a well-established framework where "population" refers to the offspring pool from which selection occurs. This is not a valid criticism.

- **Data quality confounder claims against ShearedLlama comparison**: The critic argues DarwinLM's advantage is "partly due to better training data." However, the paper includes ShearedLlama trained on identical Fineweb-Edu data (10B† in Table 1), showing DarwinLM still outperforms (62.8 vs 61.9). The paper already addresses this confounder.

- **Demand for mutation operator ablation**: Requiring ablation studies comparing level-switch mutation to alternatives like "random sparsity resampling, crossover" is excessive. The mutation design is well-motivated by the constraint-preserving objective, and such ablations are not standard for this venue.

- **GPU count discrepancy claim**: The critic notes inconsistency between "4 consumer-grade GPUs" (introduction) and "10× L40 GPU workstation" (Section 4.1). However, the introduction specifies 4 GPUs for the *search phase* while the full training uses a larger cluster—this is consistent, not contradictory.

- **Missing population diversity analysis**: Demanding investigation of "whether using a population size >1 during evolutionary search" improves results is scope creep; (1+λ)-ES is a valid and widely-used evolutionary strategy.

## Novel Insights

The training-aware selection paradigm—using progressively larger fine-tuning budgets within the search loop to predict recovery potential—represents a meaningful shift from one-shot pruning methods. The key insight is that two pruned models with identical one-shot KL-divergence may respond differently to fine-tuning, and this difference can be detected early through lightweight training. The paper's Figure 2 provides empirical support, though the correlation between 200K-token and 10B-token performance remains the unvalidated assumption. The extension to MoE architectures is technically sound but limited in scope (uniform across blocks, attention modules excluded); the method's value lies primarily in demonstrating that non-uniform width pruning within experts provides measurable gains over uniform expert compression.

## Suggestions

- Add a 10B-token TAS ablation to validate whether training-aware selection provides benefits at the scale where final results are reported, even if approximate (e.g., using fewer generations).

- Clarify parameter-size comparisons with ZipLM by either (a) including results at matched parameter counts, or (b) explicitly acknowledging the size asymmetry in the text and discussing how comparison under different constraints affects interpretation.

- Report standard deviations across at least 3 random seeds for key metrics to establish statistical significance of small performance differences.

- Explicitly frame the MoE contribution: clarify in the abstract/introduction that non-uniformity applies within expert MLPs with uniform sparsity across MoE layers, rather than claiming full non-uniform MoE pruning.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 4.0, 6.0]
Average score: 5.5
Binary outcome: Reject
