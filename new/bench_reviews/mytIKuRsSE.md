Now I have a good understanding of the paper and the calibration anchors. Let me compose the final review.

## Summary

This paper identifies Dual-level Noisy Correspondence (DNC) in Multi-Modal Entity Alignment (MMEA) — misalignments in both intra-entity (entity-attribute) and inter-graph (entity-entity, attribute-attribute) correspondences — and proposes RULE, a framework with three components: (1) a two-fold reliability estimation principle combining Dempster-Shafer-based uncertainty and consensus, which divides pairs into noisy/clean subsets; (2) Dually Robust Learning (DRL) and Fusion (DRF) modules that handle inter-graph and intra-entity noise during training; and (3) a Test-time Correspondence Reasoning (TTR) module that leverages an MLLM with chain-of-thought prompting to uncover latent attribute connections at inference time. Experiments on five benchmarks under varying noise levels show consistent improvements over seven baselines.

## Strengths

- **Well-motivated and novel problem formulation.** The paper identifies a practically important problem — dual-level noisy correspondence in MMEA — that encompasses both entity-attribute and inter-graph noise. The claim of over 50% inherent NC in ICEWS benchmarks and the motivating examples (e.g., Elvis Tsui/Jason Momoa visual resemblance, Mr. & Mrs. Smith name confusion) make a compelling case that this problem is real and impactful.

- **Principled reliability estimation design.** The two-fold principle combining uncertainty (via Dempster-Shafer Theory / Dirichlet distributions) and consensus (via similarity-label agreement) is well-justified. Theorem 1 formally establishes that low uncertainty alone is insufficient, motivating the consensus component. The SU/SI/SC pair division with tailored loss strategies is a reasonable approach to handling different noise levels.

- **Strong empirical results.** RULE consistently outperforms all seven baselines across five datasets and three noise settings (Inherent DNC, 20%, 50%), sometimes by very large margins (e.g., ICEWS-YAGO at 50% DNC: 46.9 vs 30.6 H@1). Even under "Inherent DNC" (no injected noise), RULE improves over baselines, suggesting the framework addresses real corruption in these benchmarks.

- **Good ablation design.** Table 3 decomposes contributions of training-time (DRL, DRF, Only Unc., Only Cons.) and test-time (TTR, MLLM Enhance) components, showing that each contributes meaningfully. The visualization of reliability distributions (Fig. 3b, Fig. 4) provides qualitative support for the pair division mechanism.

- **Novel test-time reasoning component.** The TTR module using MLLM-based chain-of-thought reasoning to uncover latent attribute connections is a creative direction for MMEA that, to the authors' knowledge, is among the first to address test-time robustness for this task.

## Weaknesses

### Major:

- **The headline results conflate contributions from the core learning framework (DRL/DRF) and the large external MLLM (TTR).** The ablation (Table 3) shows that TTR provides a substantial boost (Non-name: 56.5→58.2 H@1; All-attributes: 94.0→97.7 H@1). While the paper acknowledges TTR as a component, the central claim — "superior robustness of RULE against DNC" — bundles training-time and test-time modules. Critically, no baseline is given access to the same 72B-parameter MLLM for reasoning, making it impossible to isolate how much of RULE's advantage comes from DRL/DRF versus from the powerful external reasoner. The "MLLM Enhance" ablation (56.6 vs 58.2 Non-name H@1) does show that the CoT-based reasoning provides complementary value beyond vanilla MLLM usage, but the overall gap between "w/o TTR" and full RULE remains a dominant source of the reported improvements, particularly in the All-attributes setting. The paper should either provide baselines enhanced with the same MLLM, or more clearly isolate the DRL/DRF contribution.

- **The self-referential reliability estimation creates a circular dependency under heavy noise.** The thresholds βu and βc (Eq. 8) are derived from STP = {i | arg max(si) = arg max(yi)}, which simultaneously depends on the model's current predictions si and the noisy labels yi. Under high noise, early-training predictions si are unreliable, so STP may not meaningfully represent "true positives," potentially causing threshold degradation or collapse. Similarly, consensus ci = max(0, si · yi) depends on yi which contains label noise. The paper's ablation (Fig. 3b, Fig. 4) provides qualitative visualizations that the reliability measure separates clean and noisy pairs, but provides no quantitative precision/recall analysis of noisy-pair detection, nor any study of how thresholds evolve across training epochs. Without such evidence, the robustness of the SU/SI/SC division mechanism under the very noise conditions the method is designed to address remains insufficiently validated.

- **Assumption 1 and the greedy marginal contribution strategy are strong and unvalidated.** Assumption 1 claims that correctly associated attributes have Δ ≥ 0 marginal contribution, while irrelevant attributes have Δ < 0. This is a strong assumption: a correct but weakly informative modality (e.g., a brief textual description) could reduce average similarity when added, while an incorrect but superficially similar attribute could increase it. The initial subset size |π₀| = M/2 + 1 is also not justified. Since the estimated correspondence ŷi feeds directly into the consensus computation (and thereby into the training losses), errors in this greedy estimation could propagate through the entire framework. The paper provides no empirical evaluation of the greedy selection's accuracy.

### Minor:

- **Synthetic noise model may not reflect real-world DNC patterns.** The noise injection involves random entity replacement, attribute reassignment, Gaussian noise for images, and character replacement for text. While a standard practice, these are random corruptions that may not capture the systematic, semantically-motivated errors described in the introduction (e.g., visual resemblance between different people, name-based disambiguation errors). The gap between synthetic and real noise patterns is acknowledged indirectly but not addressed.

- **Missing runtime/computational cost analysis.** The TTR module requires inference from Qwen2.5-VL-72B-Instruct for every test entity, which is a substantial computational requirement. While not inherently disqualifying, the paper does not discuss latency, memory, or throughput, leaving practitioners unable to assess feasibility for real-world deployment.

- **The claim of revealing a "new problem" is somewhat overstated.** Prior work on noisy correspondence for entity alignment (Lin et al., 2023) and cross-modal matching (Huang et al., 2021) already tackles multi-level noise. What this paper adds is the specific "dual-level" formulation combining intra-entity and inter-graph noise simultaneously, which is a meaningful extension rather than an entirely new discovery.

### Trivial:

- The notation s_i in Eq. 5–7 is somewhat overloaded (similarity vector vs. marginal contribution context), which can cause confusion on first reading.

## Nice-to-Haves

- A controlled experiment giving baselines access to the same Qwen2.5-VL-72B model (even with vanilla prompting) to isolate the DRL/DRF contribution from TTR's MLLM advantage.
- Quantitative evaluation of noisy-pair detection precision/recall across noise levels, and analysis of how threshold stability evolves during training.
- Experiments decomposing performance under only intra-entity NC or only inter-graph NC to validate the "dual-level" necessity.
- Evaluation with a smaller MLLM (e.g., 7B) for TTR to assess whether the gains are transferable or scale-dependent.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **"Qwen2.5-VL-72B is a proprietary(-ish) model not available for independent verification"** — The paper cites Qwen2.5-VL-72B-Instruct (Bai et al., 2025), which is a publicly available model. Per review rules, cited models are assumed to exist and be available.

2. **"Baselines may have suboptimal CLIP implementation"** — The paper explicitly states "we adopt the same backbone (i.e., CLIP) for all baselines and our method" and refers to Appendix G.11 for different backbone results. The assumption of unfair backbone treatment is not supported.

3. **"Missing related works in noisy correspondence/label learning literature"** — Per review rules, we cannot confirm the existence of specific uncited works, and cannot verify whether they are truly missing or irrelevant.

4. **"The paper should compare with noise-robust learning methods like PNLearning or twin noisy labels"** — These methods are from different domains (cross-modal matching and person re-ID, not MMEA). The paper compares with seven MMEA baselines. While cross-domain robust baselines could strengthen the paper, the absence of domain-specific noise-robust MMEA methods is not a critical omission.

5. **"Formatting and presentation issues"** — Per review rules, formatting/style nitpicks are removed.

6. **"The noise injection percentages don't specify distribution across E-E/E-A/A-A types"** — The paper states "the proportion of corrupted E-E/E-A/A-A pairs" at 20% and 50%, but the exact per-type breakdown is not detailed. This is a minor clarity point, not a methodological flaw.

## Novel Insights

The paper's most interesting structural insight is the decomposition of the noise handling problem into training-time and test-time components — DRL/DRF for learning robust representations, and TTR for leveraging an external reasoner to correct inference-time errors. This dual approach recognizes that some noisy correspondences (particularly semantically misleading ones like visual resemblance errors) may be beyond the reach of training-time robustness alone. However, this insight also exposes the core tension: the most powerful component (TTR) depends on a massive external model, creating an attribution problem for the reported gains.

## Suggestions

- **Isolate core algorithmic contribution.** Report results for DRL+DRF alone ("w/o TTR") as the primary comparison against baselines, and present TTR gains as an additional enhancement. This would clarify how much of the improvement comes from the novel training-time framework versus external MLLM access.
- **Add quantitative noisy-pair detection evaluation.** Report precision/recall/F1 for the SU/SI/SC classification under different noise levels to validate the reliability estimation's correctness, ideally across training epochs.
- **Test with a smaller MLLM.** Running TTR with a 7B-parameter model would show whether the approach is practical beyond the 72B regime and would help assess the generalizability of the test-time reasoning concept.
- **Validate Assumption 1.** Measure how often the greedy marginal contribution correctly identifies relevant vs. irrelevant attributes on clean data where ground truth is known, to assess the approximation quality.

## Score and Decision

**Calibration comparison:**

- The Norton paper (9Cu8MRmhq2, Accept Oral, scores 8/8/8/8) tackles multi-granularity noisy correspondence in video-language learning with a principled OT framework, clean experiments, and clear contributions. RULE similarly tackles a real noise problem in MMEA but with more unresolved concerns about circularity in reliability estimation and the confounding contribution of a large external model.

- The OTGM paper (6w2HEMxzq7, Reject, scores 6/6/5/5) proposed OT-based graph matching with noisy correspondence but was criticized for weak connection between components and insufficient experimental analysis. RULE has stronger empirical results than OTGM but shares the concern about component attribution.

- The EDL-based feature matching paper (4NWtrQciRH, Accept Poster, scores 5/8/6/5) applied evidential learning to a specific task with moderate novelty. RULE similarly applies EDL-based uncertainty to MMEA, with the addition of the consensus mechanism and TTR, but faces the same "combination of existing techniques" novelty concern.

- The noisy multi-modal pairs paper (CvxcWCDX0h, Reject, scores 3/3/5/3) was rejected for limited novelty and questionable contributions. RULE is clearly stronger than this, with more extensive experiments and a more complete framework.

RULE is a solid paper addressing a real problem with strong empirical results and a reasonable framework design. Its main issues are: (1) the TTR module using a 72B MLLM inflates headline numbers without comparable baselines having access to the same tool, making the core DRL/DRF contribution harder to evaluate in isolation; and (2) the reliability estimation has a circular dependency under noise that is analyzed only qualitatively. However, the "w/o TTR" ablation shows that DRL+DRF alone still substantially outperforms baselines (e.g., ICEWS-WIKI 50% DNC: 58.2 vs ~43.9 H@1 for best baseline), so the core framework does provide genuine value. The issues are significant but not fatal — they concern attribution and validation depth rather than fundamental incorrectness.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>