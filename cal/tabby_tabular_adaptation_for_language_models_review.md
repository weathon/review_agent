=== CALIBRATION EXAMPLE 16 ===

# Final Consolidated Review
## Summary
Tabby introduces a Mixture-of-Experts (MoE) architecture modification for transformer-based language models to improve tabular data synthesis. By replacing standard MLP layers or LM heads with column-specific expert modules, Tabby dedicates separate parameters to model each data column. The paper demonstrates improved synthetic data quality on six benchmark datasets, with Plain-trained Tabby MH models achieving performance near or equal to real data on several tasks.

## Strengths
- **Competitive empirical performance:** Tabby MH achieves the highest MLE among LLM-based methods on 4/6 datasets (Table 2), reaching or exceeding real-data performance on Diabetes, Travel, and Adult classification tasks. The method outperforms prior tabular synthesis approaches including CTGAN, TVAE, and Tab-DDPM on several benchmarks.
- **Novel architectural contribution:** The paper correctly identifies that prior LLM-based tabular synthesis work focused on training techniques (GReaT, TapTap, Tabula) rather than architectural modifications. The idea of assigning dedicated expert parameters to individual columns is a reasonable approach to better model column-specific distributions.
- **Per-column loss tracking provides diagnostic insight:** The training formulation in Section 3.3 enables monitoring loss per column (Figure 4), revealing that "Occupancy" initially dominates loss while "Median Income" plateaus early—actionable information for practitioners about which columns need better preprocessing or more data.
- **Discovery that Plain training is competitive:** The paper finds that simple fixed-order training often outperforms more complex shuffling techniques (GReaT) when paired with Tabby MH, simplifying the pipeline compared to prior work.

## Weaknesses
- **Uncontrolled parameter count increase confounds the main claims:** Table 3 reveals that Tabby MH Distilled-GPT2 has 270M parameters while the Non-Tabby baseline has only 80M—a 3.375× increase. The paper claims improved performance from the MoE architecture, but never isolates whether gains come from architectural innovation versus simply having more parameters. A dense baseline with equivalent parameters is needed to validate Claim 1. This is the most significant experimental gap.
- **MMLP and MMLP-MH variants frequently degrade performance:** Table 2 shows that MMLP variants often underperform the Non-Tabby baseline (e.g., Plain MMLP achieves 0.28 R² on Abalone vs. NT's 0.46; MMLP-MH achieves 0.00 on House vs. NT's 0.70). The paper presents Plain MH as successful while never explaining why the MLP-based MoE modification hurts. Understanding when MoE helps versus harms is central to evaluating this architecture.
- **Terminology "Gated MoE" is misleading:** The paper describes using "Gated Mixture-of-Experts layers" (Abstract), but Section 3.1 reveals routing is deterministic—column i always routes to expert i. Standard MoE uses learned gating networks. The paper should clearly state that routing is hard-coded rather than learned, and avoid the "gated" terminology that implies a learned mechanism.
- **Inconsistent performance claims across the paper:** The abstract claims parity with real data on "3 out of 6 datasets," page 2 claims "3 out of 6," Section 4.1 claims "highest MLE in 4 out of 6 datasets," and the conclusion claims "two out of three"—these are different metrics, but the inconsistency creates confusion about what exactly the paper achieves. For precision: Tabby MH matches or exceeds real-data MLE on Diabetes, Travel, and Adult (3 datasets), which differs from achieving "highest MLE" overall.
- **Claim 2 rests on limited evidence:** The scaling experiment (Table 3, Figure 3) uses only one dataset (a House subset) and one training regime (GReaT, 5 epochs). The Llama comparison shows marginal improvement (MH Llama: 0.562 ± 0.022 vs. NT Llama: 0.560 ± 0.015)—the difference is within one standard deviation. A broader evaluation across multiple datasets is needed to support general claims about scaling.
- **Multiple models fail on Rainfall dataset without analysis:** Table 2 shows asterisked entries indicating failure to generate valid samples for GReaT NT, GReaT MH, and GTT variants on Rainfall. The paper uses this to argue that Plain Tabby MH is more robust, but provides no analysis of why generation fails (Rainfall has only 3 columns: 2 categorical, 1 numerical). Understanding failure modes is important for practical reliability.
- **Small dataset scale limits generalization:** Five of six datasets have fewer than 15K training rows; Diabetes has only 576. The largest (Adult: 36K rows) is modest by modern standards. The authors themselves note (Section 4.4) that these datasets may be too easy for baselines, but defer evaluation on more challenging benchmarks to future work.
- **Per-column loss tracking is a training-procedure benefit, not an architectural one:** Claim 3 presents column-level loss monitoring as a Tabby advantage, but this results from the training formulation in Section 3.3—not from the MoE architecture itself. A standard Non-Tabby model could use the same separated column-wise loss calculation.

## Nice-to-Haves
- **Privacy and memorization analysis:** For synthetic tabular data to be useful in healthcare/finance, privacy guarantees matter. Adding membership inference attack success rates or nearest-neighbor distance analysis would strengthen practical applicability.
- **Computational cost analysis:** Tabby MH increases parameters from 80M to 270M. The paper should report training time, inference latency, and memory footprint compared to baselines to assess practical viability.
- **Larger-scale benchmarks:** Evaluation on datasets with >50K rows and >50 columns would better validate claims about general tabular modeling capabilities.
- **Distributional fidelity metrics:** Beyond MLE (downstream classifier performance), include statistical distributional metrics (e.g., marginal distribution comparisons via Kolmogorov-Smirnov tests or correlation matrix fidelity) to assess whether inter-column dependencies are captured.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Claims about missing related work or missing architectural precedents:** The harsh critic claims Tabby isn't the "first architecture modification" for structured data generation, but provides no citations. Without external verification, this criticism cannot be substantiated.

- **Claim that the paper lacks novelty because MoE is established:** This misunderstands the contribution—the novelty claim is about applying MoE specifically to tabular column modeling in LLMs, not about inventing MoE itself.

- **Criticisms about the LTM framing being "aspirational":** The paper clearly frames Tabby as "an initial step towards" LTMs, which is reasonable positioning. The criticism that Tabby requires knowing column count at construction time is valid but doesn't invalidate the "initial step" framing.

- **Demand for user studies:** Not standard for algorithmic/empirical synthesis papers in this venue.

## Novel Insights
The paper reveals an unexpected finding: the simple "Plain" training method (fixed column order) outperforms the column-shuffling approach (GReaT) when paired with Tabby MH's increased capacity. Prior work (GReaT) argued that shuffling prevents spurious column-order correlations, but Tabby's column-specific experts may reduce this need. This suggests that architectural capacity can substitute for training-time augmentation—a finding that could simplify future LLM-based tabular synthesis pipelines. Additionally, the per-column loss analysis (Figure 4) demonstrates that different columns converge at different rates, which could inform data augmentation and preprocessing strategies for tabular generative models.

## Suggestions
1. **Add a parameter-matched ablation:** Train a dense Non-Tabby model with ~270M parameters (e.g., by widening layers or embedding dimensions) to isolate whether gains come from architecture or capacity.
2. **Explain MMLP's underperformance:** Provide mechanistic analysis or hypotheses for why MoE-MLP hurts while MoE-Head helps, as this is central to understanding when Tabby is beneficial.
3. **Clarify the routing mechanism in writing:** Replace "Gated MoE" with "column-assigned MoE" or similar, and explicitly state that routing is deterministic based on column index.
4. **Analyze Rainfall failures:** Investigate why certain configurations fail to produce valid samples on Rainfall—this could reveal limitations or preprocessing needs.
5. **Standardize claims:** Ensure consistency between abstract, body, and conclusion regarding performance metrics (MLE improvement vs. real-data parity vs. baseline comparison).

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 1.0]
Average score: 3.0
Binary outcome: Reject
