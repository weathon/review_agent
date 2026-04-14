=== CALIBRATION EXAMPLE 54 ===

# Final Consolidated Review
## Summary
OS-Atlas is a foundation action model (LAM) for generalist GUI agents that addresses the performance gap between open-source and commercial VLMs in GUI grounding and out-of-distribution (OOD) agentic tasks. The paper contributes (1) a novel multi-platform data synthesis toolkit yielding the largest open-source GUI grounding corpus (13M+ elements across Windows, macOS, Linux, Android, and Web), (2) a unified action space resolving cross-platform naming conflicts, (3) the corrected ScreenSpot-V2 benchmark, and (4) a comprehensive evaluation across six benchmarks on three platforms. The 7B model substantially outperforms GPT-4o on five of six OOD benchmarks.

---

## Strengths

- **Open-source multi-platform grounding corpus at unprecedented scale**: OS-Atlas releases 1.9M web + 285K mobile + 54K desktop screenshots with 13.58M GUI elements — the first open-source corpus to include desktop (Windows/macOS/Linux) grounding data, which all prior open-source efforts omit (Table 1). The engineering infrastructure required to automate data collection across five OS environments with distinct A11y APIs is non-trivial and fills a genuine gap.

- **Strong OOD generalization of the 7B model**: Without any task-specific fine-tuning, OS-Atlas-7B outperforms GPT-4o on five of six OOD benchmarks by large margins — e.g., OmniAct-Web SR: 59.99% vs. 34.06%, GUI-Odyssey SR: 26.96% vs. 5.36%, AndroidControl-Low SR: 50.94% vs. 28.39% (Tables 4, 5). This supports the core claim that the grounding pre-training + unified action space pipeline enables genuine generalization.

- **Empirically validated unified action space**: The paper demonstrates that reducing from 17 to 10 unique action types by resolving cross-platform conflicts (tap→click, press\_home→home, type→input) produces a measurable improvement in both step-wise success rate (~5–11pp on mobile/web) and grounding accuracy (Figure 5). This is a concrete, reproducible methodological contribution backed by ablation.

- **ScreenSpot-V2 benchmark correction**: Identifying and correcting 11.32% annotation errors in the widely-used ScreenSpot benchmark is a service to the broader community. Releasing ScreenSpot-V2 ensures more reliable future comparisons for GUI grounding evaluation.

- **Grounding data scaling analysis**: The paper demonstrates a clear positive correlation between grounding data volume and downstream performance (Figure 3), particularly for IoU, motivating future data collection and the need for richer evaluation metrics beyond binary accuracy.

---

## Weaknesses

### Fatal
None.

### Major

- **ScreenSpot-V2 annotation process lacks transparency and creates self-evaluation risk.** The paper states that 11.32% of ScreenSpot annotations were corrected, but provides no information on who corrected them, what the correction criteria were, whether raters were blind to OS-Atlas's predictions, or whether corrections were independently verified. If the correction process — even inadvertently — was informed by OS-Atlas's behavior, the V2 results (Appendix B) would be biased in the authors' favor. Critically, the main body (Table 2) evaluates on the *original* ScreenSpot while the paper claims V2 is more accurate; this creates a strange situation where the "better" benchmark is relegated to the appendix. The authors should either (a) make V2 the primary benchmark and provide a rigorous description of its annotation protocol, or (b) clearly justify why original ScreenSpot is used as the primary metric despite known errors.

- **Overclaim in §5.2: "OS-Atlas demonstrated superior capabilities in addressing unseen tasks across all six OOD evaluation datasets."** This is accurate for the 7B model but not for the 4B model, which clearly underperforms GPT-4o on OmniAct-Web (22.99% vs. 34.06% SR) and OmniAct-Desktop (26.94% vs. 50.67% SR) (Table 4). The paper conflates results for both model sizes in the narrative without qualification. This needs to be corrected to accurately characterize where each model variant falls relative to GPT-4o.

- **Ablation values in Figure 5 are approximate.** The paper reports Figure 5 ablation results as "~33", "~26", etc. (values estimated from a bar chart image), which is scientifically inadequate. These values are computed from a deterministic training run and exact figures must exist. Reporting them approximately undermines the credibility of the ablation analysis, which is one of the paper's two core methodological claims.

- **No human evaluation of synthesized data quality.** The 13M-element corpus is the paper's central contribution, yet no verification of its quality is provided. GPT-4o is used to generate instruction grounding annotations for Mind2Web, AMEX, and AITZ, and a filtering heuristic is used for web data, but no human spot-check or precision/recall estimate is reported. For a dataset paper, a quality audit — even on a random sample of 500–1,000 elements — is essential to establish the trustworthiness of the resource.

### Minor

- **SFT gains over the Qwen2-VL-7B backbone are inconsistent and partially marginal.** Qwen2-VL-7B was itself pretrained on GUI data; consequently OS-Atlas-7B's SFT improvements over it vary dramatically: +15% on OmniAct-Web (78.58%→93.56%) but only +0.43% on GUI-Act-Web (82.27%→82.70%) and +1.75% on GUI-Odyssey (60.23%→61.98%). The paper should characterize more precisely *when* and *why* grounding pre-training provides the most value, as the current narrative uniformly claims OS-Atlas is a "robust foundation" without acknowledging these mixed results.

- **OS-Atlas-Pro reproducibility gap.** Section 5.4 introduces OS-Atlas-Pro trained on "all 7 previously mentioned agent datasets" but the main text explicitly names only 3 datasets for OS-Atlas training (§5.1: AMEX, AITZ, Mind2Web). The reader cannot identify all 7 datasets without cross-referencing several paper sections and footnotes. For a result the paper uses to justify "fully leverag[ing] its potential for broader applications," sufficient reproducibility information should be in the main text.

- **Ablations are limited to the 4B (InternVL-2) backbone "due to GPU constraints."** Whether the insights generalize to the 7B (Qwen2-VL) backbone — which uses a different architecture, pretraining data, and resolution handling — is unknown. Given that the 7B model is the paper's flagship, the mechanistic claims from ablations should be validated, at minimum via a partial ablation at 7B scale.

- **No limitations section.** For a data-plus-model paper at ICLR, an explicit discussion of limitations is expected. Key unacknowledged limitations include: dependency on A11y tree quality (which varies across OS versions/applications and is a known practical failure mode), dependence on commercial GPT-4o for instruction-grounding annotation, the 35× desktop/web data imbalance, and the substantial OSWorld gap (~58pp below human performance, Table 3).

### Tiny

- A counterintuitive result goes undiscussed: OS-Atlas-Base-7B achieves higher Desktop Text accuracy *without* a GPT-4o planner (91.75%) than *with* one (90.21%) in Table 2. This suggests the planner sometimes introduces noise that the grounding model then follows, and deserves at least a brief mention.

- The rationale for exactly 3 basic actions (click, type, scroll) is stated but not deeply motivated. Why are drag and long\_press relegated to "custom"? These are common enough on mobile to warrant brief justification.

---

## Nice-to-Haves

- **Human evaluation of synthesized grounding data**: Even a 1,000-sample manual audit with precision/recall reported would substantially strengthen the dataset contribution.
- **Failure case visualizations**: Side-by-side examples of OS-Atlas vs. GPT-4o failure modes would help characterize the residual capability gap more concretely.
- **Training compute transparency**: GPU hours and cluster configuration are in Appendix E but should be surfaced in the main paper for reproducibility and environmental impact.
- **Discussion of desktop data scaling**: Given the 35× gap between web and desktop training data, it would be informative to show whether adding more desktop grounding samples yields diminishing returns or whether 54K is a genuine bottleneck.
- **Resolution and aspect ratio analysis**: Mobile vs. desktop vs. web have vastly different native resolutions; an analysis of how image resolution affects grounding accuracy would be useful for practitioners.

---

## Removed Points

*These points were considered but removed or substantially weakened from the review.*

- **"First LAM specifically designed for GUI agents" (Concern 1 from harsh critic)**: The qualifier "to the best of our knowledge" is present in the paper and the claim is made in the related work section, where such framing is conventional. The paper defines LAM in a specific way (cross-platform, unified action space, GUI-specific) that distinguishes it from prior grounding models. Not a substantive issue.

- **Data contamination between Mind2Web train and GUI-Act-Web test (Concern 5)**: The paper explicitly uses only train splits for training and test splits for evaluation, which is standard practice. Additionally, footnote 1 explicitly states that Wave-UI entries from Mind2Web are removed to avoid contamination. The critique requires more specific evidence of an actual split-level contamination and is not substantiated.

- **OS-Atlas-4B underperforms Qwen2-VL-7B in SFT (OmniAct-Desktop: 84.78% vs 91.77%)**: These are different model sizes (4B vs. 7B). Comparing different parameter counts does not constitute a paper flaw; readers understand that a smaller model may underperform a larger backbone. The table clearly labels these as different scales.

- **Comparison with concurrent GUI foundation models beyond UGround**: Per review policy, missing related works claims are excluded as external sources cannot be confirmed.

- **"End-to-end task completion evaluation is absent" (Spark Finder)**: Factually incorrect. The paper evaluates on OSWorld in §4.3/Table 3, which is a fully interactive end-to-end benchmark requiring multi-step task completion. The step-level metrics in §5 are complementary and standard in the GUI agent field.

- **Cross-platform generalization not tested (train desktop, test mobile)**: This is a reasonable scientific curiosity but outside the paper's stated contributions. The paper's cross-platform claims refer to the corpus and the model trained on all platforms, not zero-shot leave-one-platform-out generalization.

- **Theoretical/statistical significance for action space improvement**: Requesting significance testing on a single-run large-scale evaluation is not standard in this community and does not meet the bar for a weakness.

---

## Novel Insights

The paper surfaces a practically important finding that is easy to overlook: *grounding pre-training is asymmetrically valuable across platforms*. Web pre-training alone fails to generalize to desktop (Figure 4: desktop accuracy drops from ~72% to ~28% when web-only), but the converse is less dramatic — web performance degrades relatively modestly when desktop/mobile data is removed. This suggests that web GUI representations have structural overlap with mobile but not with desktop, which has implications for future data collection prioritization in GUI agent research. Related to this, the scaling analysis (Figure 3) shows IoU continues to improve past convergence in accuracy, indicating that standard binary accuracy metrics are masking ongoing model improvement — a finding that motivates better evaluation metrics for future work.

---

## Suggestions

1. **Fix the §5.2 overclaim**: Qualify "superior capabilities across all six OOD datasets" explicitly as referring to OS-Atlas-7B; add a sentence acknowledging that OS-Atlas-4B underperforms GPT-4o on OmniAct benchmarks and discuss why (model capacity, instruction-following complexity).

2. **Report exact values for Figure 5 ablations**: Provide a table with precise numbers, not bar-chart approximations.

3. **Add a ScreenSpot-V2 annotation protocol appendix**: Describe who annotated corrections, what criteria were used, and whether the correction process was conducted independently of OS-Atlas evaluation results. Consider making V2 the primary benchmark or providing a clear rationale for using V1 in the main table.

4. **Add a limitations section**: Address A11y tree dependency, GPT-4o annotation bias, desktop data sparsity, and the OSWorld human-performance gap.

5. **Provide at least partial ablation results on the 7B backbone**: Even a single ablation (w/o pre-training at 7B scale) would validate that the mechanistic claims from 4B ablations generalize.

6. **Clarify the 7 datasets used for OS-Atlas-Pro in the main text**: A table listing all 7 datasets with their train/test splits and sizes would make Section 5.4 reproducible.

7. **Discuss the no-planner vs. planner performance inversion on desktop**: The observation that GPT-4o planner slightly hurts 7B desktop grounding accuracy suggests the planner introduces errors; this is a practical insight worth a sentence or two.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
