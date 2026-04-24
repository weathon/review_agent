Now let me search for calibration anchors.Now I have sufficient information to write the final review. Let me consolidate everything.

---

## Summary
This paper introduces GSA-VLN, a task requiring VLN agents to continuously adapt to specific persistent environments through memory accumulation and parameter updates. To support this task, the authors create GSA-R2R, a large-scale benchmark expanding from 29 to 150 evaluation scenes across 20 building types with 90,000 instructions in 7 styles, backed by a three-stage LLM-based instruction orchestration pipeline. They benchmark existing adaptation methods, reveal important negative findings (TTA methods degrade performance; existing memory-based methods fail with long histories), and propose GR-DUET, a graph-retained variant of DUET achieving 8–12% SR gains across all splits.

---

## Strengths

- **Comprehensive scale and diversity of GSA-R2R (Table 1)**: The dataset expands evaluation scenes from 29 (R2R) to 150, building types from 6 to 20, and instructions from 6,522 to 90,000 across 7 styles, with unseen vocabulary growing from 545 to 2,905 words — a meaningful advance in VLN evaluation diversity with concrete numbers.

- **Principled ID/OOD split backed by embedding-space evidence (Figure 4)**: The use of GPT-4-based building-type classification to separate residential (ID) from non-residential (OOD) environments, confirmed by t-SNE showing that Scene and User instruction embeddings lie in genuinely distinct distributional clusters from training data, makes the OOD claim concrete rather than asserted.

- **Non-obvious negative benchmarking results (Section 4.3.2, Table 4)**: The finding that TTA methods (TENT, SAR) actually *decrease* SR (47.0 → 44.2/44.6 SPL on Test-R-Basic) is counter-intuitive and well-explained: entropy minimization is ill-suited for sequential decision-making where navigation errors compound irreversibly. This is a genuine community insight, not boilerplate.

- **Three-stage instruction pipeline with quality validation (Table 2)**: The pipeline demonstrably improves path-matching accuracy from 52.2% (raw EnvDrop) to 80–83.3% (refined), while Scene instructions achieve 96.1% style distinctiveness. The pipeline design and validation are systematic.

- **GR-DUET provides consistent, substantial improvements across all five splits (Tables 4–6)**: +11.6% SR on Test-R-Basic, +8.5% on Test-N-Basic, +8.5% on Test-N-Scene versus vanilla DUET. The improvements are consistent across both ID and OOD settings.

---

## Weaknesses

### Fatal
None.

### Major

- **The core "adaptation over time" claim lacks sequential evaluation evidence.** The paper's defining contribution is that agents *improve as they accumulate experience* in a scene (Equations 2–4). However, performance is reported only as aggregated metrics over all 600 episodes per scene — there is no learning curve showing SR as a function of episode index (e.g., episodes 1–50 vs. 51–100 vs. 451–600). Table 8's buffer ablation (buffer=1 → 57.6% SR, buffer=50 → 69.3% SR) provides indirect evidence that more accumulated history helps, but is not the same as a temporal performance curve. Without such an analysis, GR-DUET's gains are consistent with a simpler explanation: each episode receives richer contextual input from a pre-populated graph, rather than the agent genuinely improving as it accumulates task-specific experience. This distinction is important given the paper's framing of GSA-VLN as a continual-adaptation benchmark.

- **Comparison with memory-based baselines is confounded by training regime differences.** GR-DUET is initialized from DUET with a modified pretraining procedure (full ground-truth topological maps) and PREVALENT augmentation. TourHAMT and OVER-NAV are applied with their original pretrained weights. While Table 7 shows that even without pretraining or augmentation (row ×, ×: 56.8% SR), the DUET architecture massively outperforms TourHAMT (14.9% SR), suggesting the architectural difference dominates, the absence of "TourHAMT/OVER-NAV retrained with equivalent pretraining data" baselines means the claim that GR-DUET's graph mechanism is the key innovation over existing memory methods cannot be cleanly supported.

### Minor

- **Small-scale human evaluation for a 90,000-instruction dataset.** The reliability study uses 15 participants evaluating 20 randomly selected instructions (300 total judgments, ~0.3% of all path-instruction pairs). While the results (~80% accuracy, 96.1% style distinctiveness for Scene instructions) are encouraging, the sample is too small to make strong reliability claims about 90K instructions. Expanding the study or providing bootstrapped confidence intervals would strengthen the claim.

- **Asymmetric manual verification in environment classification (Section 3.3.1).** The paper explicitly states "For non-residential results, we manually verify and correct the predictions," but describes no analogous manual verification for the residential side. This asymmetry could introduce systematic labeling bias in the ID/OOD split.

- **ScaleVLN result with data leakage left in the main table (Table 3) without visual separation.** While the paper clearly footnotes the issue (ScaleVLN uses HM3D buildings overlapping with GSA-R2R), ScaleVLN dramatically outperforms all other methods (79% vs. ~58% SR) and sits in the same table rows. Separating it visually or into a dedicated row for "contaminated baselines" would avoid a misleading first impression.

### Trivial

- **Back-Translation domain shift explanation (Section 4.3.2) is asserted without quantitative evidence.** The claim that BT improvement "diminishes due to domain shift between authentic and evaluation instructions" is stated as fact but not measured.

---

## Nice-to-Haves

- A per-episode performance curve (SR as a function of episode index within each scene) would directly validate the "adaptation over time" claim and is the single most impactful addition.
- A "DUET + PREVALENT augmentation (no cross-episode graph)" baseline in Table 4 or Table 7 would isolate the graph mechanism's contribution from the training-data change — although Table 8 (buffer=1 vs. buffer=50, holding pretraining+aug constant) partially addresses this.
- Analysis of which building types are hardest for adaptation (OOD split breakdown by building type) would increase GSA-R2R's utility as a fine-grained benchmark.
- Case study visualizing the global graph evolution across episodes in a single building would make the mechanism concrete.

---

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "Cross-episode graph contribution not isolated from augmentation"** — REMOVED as largely incorrect. Table 8 does isolate this by holding pretraining+augmentation constant and varying buffer size (buffer=1: 57.6%, buffer=50: 69.3%). The 11.7% SR gap is attributable specifically to the cross-episode graph mechanism.

- **Harsh Critic: "Data leakage from EnvDrop training on evaluation splits"** — REMOVED as overstated. The authors explicitly disclose this in Footnote 2. The EnvDrop speaker generates initial noisy instructions that are then corrected by GPT-4 with path visualizations. The "data leakage" here is limited to the speaker seeing the environments to generate initial instructions, which is standard practice and does not compromise the evaluation.

- **Strength Finder: "Scalable environment classification methodology"** — REMOVED as generic. The description of spectral clustering + GPT-4 for building classification is a reasonable engineering choice but not a distinctive scientific contribution.

- **Strength Finder: "Detailed ablation on graph construction mechanisms (Table 8)"** — WEAKENED. The ablation is useful but the comparison of "proportion α=1.0" (66.2% SR) vs. buffer=50 (69.3%) shows only a 3.1% absolute advantage of the specific buffer mechanism over "always provide full ground-truth graph," which is modest.

- **Harsh Critic: "EnvDrop speaker training on evaluation environments"** — REMOVED. The paper discloses this, and the instructions are subsequently corrected by VLM. The concern is not a hidden flaw.

---

## Novel Insights

The most genuinely novel observation is the systematic failure mode of TTA methods in sequential decision-making: because navigation errors compound irreversibly, entropy minimization produces meaningless signals after an incorrect step, making TTA methods *actively harmful* (worse than no adaptation). This is a concrete and transferable insight about why classification-domain TTA does not naively transfer to action-chain domains. The secondary novel finding — that existing memory-based methods (TourHAMT, OVER-NAV) fail catastrophically not due to memory per se, but due to input-space distribution shift caused by excessively long concatenated history embeddings — motivates GR-DUET's topological graph design and is well-supported by evidence.

---

## Suggestions

1. Add a per-episode learning curve (SR grouped by episode index quartile per scene) to directly demonstrate the "adaptation over time" claim.
2. Report a single row of "DUET + equivalent pretraining (no graph)" in Table 4 for full transparency, even if the buffer=1 result in Table 8 already partially covers this.
3. Visually separate contaminated baselines (ScaleVLN) in Table 3 via a horizontal line or distinct shading.
4. Report 95% CI or standard errors for the human evaluation and note the sample size limitation in the text.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|---|---|---|---|
| OUuhwVsk9Z (SRDF — VLN data flywheel) | 6.5 | Accept (Poster) | Topically closest; SRDF has stronger method novelty (iterative self-improvement reaching human-level SPL) but comparable dataset scale. Paper under review has broader benchmark scope. |
| n6mLhaBahJ (HAZARD — embodied benchmark) | 6.75 | Accept (Poster) | New benchmark with dynamic environments; comparable structure (new task + dataset + baseline + proposed method). Similar quality level. |
| ADSxCpCu9s (LoTa-Bench) | 6.0 | Accept (Poster) | Embodied benchmark with comprehensive LLM evaluation. Solid but limited method contribution. Close analogue. |
| kC5nZDU5zf (Selective Visual Representations) | 7.5 | Accept (Spotlight) | Stronger empirical results and cleaner contribution isolation. Sets the ceiling. |
| pwKokorglv (Embodied Instruction in Unknown Envs) | 4.0 | Reject | Poor task definition, unfair comparisons. Paper under review is clearly better. |
| RE0aibEQ1J (IG-Net) | 4.0 | Reject | Weak baseline comparison, limited novel insight. Paper under review significantly stronger. |

**Positioning:** The paper is a solid ICLR datasets/benchmarks track contribution. The task formulation is well-motivated, the dataset is the largest and most diverse VLN evaluation resource, the benchmarking reveals non-trivial insights (TTA failure mode, memory method limitations), and GR-DUET is a sound baseline. The major weakness — lack of direct sequential learning curves — is notable for a paper specifically about adaptation over time, but does not invalidate the dataset contribution or the benchmarking findings. The training-regime confound in comparisons is real but architectural dominance is demonstrated even in the no-pretraining ablation.

This positions the paper above the reject anchors (4.0) and consistent with the HAZARD/SRDF cluster (6.5–6.75) as a solid poster-level acceptance. The missing sequential curve is more severe than SRDF's minor weaknesses, pulling it slightly below SRDF. Score: **6.0**.

**Originality:** Good — novel task formulation and large-scale dataset construction  
**Importance:** High — addresses real-world gap in VLN evaluation  
**Claims well-supported:** Mostly, with noted gap in sequential evidence  
**Soundness of experiments:** Good — comprehensive benchmarking with ablations  
**Clarity:** Clear and well-organized  
**Value to community:** High — new benchmark + negative results on TTA/memory methods are useful

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>