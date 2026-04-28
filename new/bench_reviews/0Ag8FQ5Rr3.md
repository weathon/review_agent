## Summary
This paper discovers "super weights" - individual scalar parameters in LLMs that are disproportionately critical to model functionality. Pruning a single super weight in Llama-7B increases perplexity by orders of magnitude and reduces zero-shot accuracy to guessing. The authors propose a data-free identification method using activation spikes from a single forward pass, and demonstrate that preserving these super outliers improves round-to-nearest quantization to become competitive with SmoothQuant without calibration data.

## Strengths
- **Striking empirical discovery with strong evidence**: Table 1 provides compelling quantitative isolation showing pruning a single super weight increases Wiki-2 perplexity from 5.67 to 1211.11, whereas pruning the top 7,000 magnitude outliers combined only increases it to 6.08. This is a genuinely surprising finding about parameter sensitivity in LLMs.
- **Cross-model generalization with concrete coordinates**: Table 2 provides specific super weight coordinates for 10 different model variants across Llama, Mistral, OLMo, and Phi-3 families, all located in `mlp.down_proj` in early layers. This enables immediate reproducibility and further study.
- **Calibration-free quantization achieving 71-83% of SmoothQuant's improvement**: Tables 3-4 show the method recovers most of SmoothQuant's perplexity improvement without requiring calibration data, and functions on OLMo models where SmoothQuant is incompatible due to non-parametric LayerNorm.
- **Mechanistic analysis linking super weights to stopword suppression**: Figure 5 demonstrates that removing super weights increases stopword probabilities (e.g., "the" by 2x) while decreasing semantic token probabilities, providing functional explanation beyond activation magnitude alone.

## Weaknesses

### Fatal
None

### Major
- **Unsupported "hardware-friendly" claim for quantization**: The paper claims the method is "hardware-friendly" (Abstract, Introduction Section 2.2) because it preserves only a handful of scalars in FP16. However, mixed-precision inference kernels that mask/restore specific indices at arbitrary positions require non-trivial implementation and may incur overhead compared to uniform quantization. The paper provides **zero latency, throughput, or memory bandwidth measurements** to support this efficiency claim. Calibration anchors show papers making hardware efficiency claims without runtime benchmarks are typically rejected (MPybJCVrgc.md avg 4.0, Ad7l5spCAM.md avg 3.5, dLqDqzlDxZ.md avg 3.33). This significantly weakens the quantization contribution, though the core discovery remains valid.

### Minor
- **"Single parameter" destruction claim overgeneralized**: The headline claim states "Pruning as few as a single parameter can destroy an LLM's ability to generate text" (Abstract). This is strongly evidenced for Llama-7B (Table 1, one super weight), but Table 2 shows other models have multiple super weights (Phi-3 has six). The paper does not demonstrate whether pruning a *single* weight in multi-super-weight models causes catastrophic failure, or if all identified weights must be pruned. The universality claim should be tempered to reflect what is actually demonstrated.
- **Activation spike stability across prompts not validated**: The identification method relies on detecting activation spikes during a "single forward pass" with "a single input prompt" (Section 3.1), assuming super activation magnitude is invariant "regardless of the prompt" (Introduction). While citing Sun et al. (2024) for persistent activations, the paper provides no variance analysis showing spike magnitude stability across diverse inputs. If spike magnitude varies significantly, single-pass identification might miss true outliers or identify transient ones.

### Trivial
- **Table 2 column header "No." is ambiguous**: The header appears to represent "Layer Index" (based on caption and example `layers[2]`), but "No." conventionally suggests "Count." Renaming to "Layer" would prevent confusion.
- **"Data-free" terminology imprecise**: The paper uses "data-free" to describe a method requiring a "single forward pass" with input tokens. In quantization literature, "data-free" typically implies no data needed at all (e.g., weight statistics only). "Calibration-free" or "single-sample" would be more accurate.

## Nice-to-Haves
- End-to-end latency measurements comparing against SmoothQuant and naive quantization would strengthen the efficiency claims
- Prompt variance analysis across diverse samples (e.g., 1000 samples) to confirm single-pass identification stability
- For models with multiple super weights, an ablation showing performance drop when pruning one vs. all identified super weights
- Analysis of the unembedding projection to explain mechanistically why the super activation channel suppresses stopword logits

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic: Missing comparison against AWQ/SqueezeLLM for weight quantization accuracy**: The paper does discuss AWQ and SqueezeLLM in Section 5.2.1, noting they also preserve super weights. The comparison in Figure 7 is specifically against round-to-nearest (RTN) to show block size scaling improvement, not claiming to outperform AWQ. This is within scope.
- **Strength Finder: "Robustness to Larger Block Sizes" as a strength**: This is valid but overlaps with the quantization contribution that has the major weakness above. The accuracy results are real, but without efficiency benchmarks the practical value is unclear.
- **Harsh Critic: Section 3.2 mechanism not fully explained**: The paper does provide empirical evidence (Figure 5, skip connection analysis in Figure 2) for the stopword suppression claim. While deeper mechanistic proof would be nice, the empirical evidence is sufficient for the claims made.

## Novel Insights
The paper's core discovery—that individual scalar parameters can be more important than thousands of other outliers combined—is genuinely novel and surprising. This finding parallels recent work showing extreme sensitivity in LLM systems (e.g., jNiEMDsRgc.md showing 0.003% data removal changes leaderboard rankings), but applies to parameter-level rather than data-level sensitivity. The observation that super weights consistently appear in `mlp.down_proj` in early layers across model families suggests this is a structural property of transformer architectures rather than a training artifact. The connection between super weights, persistent super activations, and stopword suppression provides a coherent (if incomplete) mechanistic story for why these parameters matter.

## Suggestions
1. **Temper the "hardware-friendly" claim** to "potentially hardware-friendly" or remove it entirely unless latency benchmarks can be provided. The accuracy results stand on their own.
2. **Clarify the scope of the "single parameter" claim** - specify that Llama-7B has one super weight where this holds definitively, while other models may require pruning multiple identified weights.
3. **Add prompt variance analysis** in the appendix showing super activation magnitude stability across 100+ diverse prompts to validate the single-pass identification method.
4. **Rename Table 2 column "No." to "Layer"** for clarity.
5. **Consider changing "data-free" to "calibration-free"** throughout to align with quantization literature conventions.

## Score and Decision

**Calibration anchors retrieved:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| jNiEMDsRgc.md | 7.33 | Similar surprising empirical discovery (0.003% data sensitivity), accepted |
| 1USeVjsKau.md | 7.00 | Quantization with hardware claims + latency benchmarks, accepted |
| 5wmetrh9cn.md | 6.00 | Surprising empirical findings without theoretical proof, accepted |
| ECl6HGrMQI.md | 6.50 | Data-free quantization approach, rejected despite good accuracy |
| qVQVVZMRVT.md | 5.20 | Quantization with outlier handling, accepted poster |
| poUpBxZtms.md | 5.00 | Outlier handling for quantization, rejected |
| MPybJCVrgc.md | 4.00 | "Hardware-friendly" claims without latency benchmarks, rejected |
| Ad7l5spCAM.md | 3.50 | No latency benchmarks against baselines, rejected |
| dLqDqzlDxZ.md | 3.33 | No real-device latency measurements, rejected |

**Reasoning:** The super weight discovery is comparable to jNiEMDsRgc.md (7.33) - a genuinely surprising empirical finding with strong evidence. However, the unsupported "hardware-friendly" quantization claim aligns with rejected papers like MPybJCVrgc.md (4.0) and Ad7l5spCAM.md (3.5). Since the discovery is the primary contribution (Contribution 1) and quantization is secondary (Contribution 4), I weight the discovery more heavily. The accuracy results for quantization are solid (Tables 3-4), but the efficiency claims without benchmarks prevent a high score. This paper is borderline - the discovery warrants acceptance, but the quantization weakness prevents a strong accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>