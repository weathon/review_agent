=== CALIBRATION EXAMPLE 22 ===

# Final Consolidated Review
## Summary

This paper introduces Visual Modality Instruction (VIM), an evaluation paradigm where textual instructions are rendered as pixels within images rather than provided as separate text tokens. The authors adapt eight existing benchmarks to create VIM-Bench and demonstrate that open-source MLLMs (LLaVA, InstructBLIP, Qwen-VL-Chat) suffer dramatic performance degradation in this setting, while proprietary models (GPT-4V, Gemini) remain robust. They propose v-MLLM, a LLaVA-1.5 model fine-tuned on VIM-format data, which recovers much of the lost performance.

## Strengths

- **Novel evaluation paradigm with practical relevance**: The VIM setting exposes a genuine capability gap in open-source MLLMs. The conceptual shift from text-modality instructions to pixel-embedded instructions is relevant for real-world deployments involving UIs, screenshots, and documents where instructions are often visual. Figure 2 and Table 3 provide clear empirical evidence of this gap across 11 models and 8 benchmarks.

- **Comprehensive benchmark coverage**: Adapting 8 diverse benchmarks (MME, MM-Vet, OKVQA, VizWiz, TextVQA, MathVista, ChartQA, MMMU) provides broad evaluation across reasoning types. The performance collapse is striking—for instance, LLaVA-1.5-7B drops from 58.41 to 0 on OKVQA when switching from TEM to VIM.

- **Diagnostic decomposition of failure modes**: Section 4.1's analysis separating instruction recognition from instruction following adds depth, showing that open-source models often fail at the OCR/semantics stage (LLaVA matches words but not semantics in 7/30 cases vs. GPT-4V's near-perfect recognition). This diagnostic approach helps identify root causes rather than just quantifying the gap.

- **Demonstration that VIM capability is trainable**: The v-MLLM results establish that the performance gap is not architectural but stems from training data distribution, which is useful information for the community.

## Weaknesses

- **Unfair comparison methodology**: V-MLLM is trained on 846k VIM-format examples while all open-source baselines are evaluated zero-shot. The paper claims v-MLLM has "robust visual instruction following capability" but does not provide a controlled comparison where LLaVA-1.5 is fine-tuned on the same VIM data. Without this ablation, it's unclear whether V-MLLM's gains are due to methodological innovation or simply exposure to in-distribution training data.

- **TEM performance regression**: Table 3 shows v-MLLM consistently underperforms LLaVA-1.5 in the standard TEM setting (e.g., OKVQA: 56.09 vs. 58.41 for 7B; MM-Vet: 29.9 vs. 30.5). This trade-off should be more prominently discussed—practitioners using standard text instructions would experience degradation by adopting v-MLLM.

- **Insufficient sample sizes for key ablations**: The instruction location decision (Section 2.1.2) is based on only 21 examples from MM-Vet. The instruction recognition analysis (Section 4.1) uses only 30 manually checked samples. These sample sizes are inadequate for quantitative claims.

- **Synthetic rendering diverges from real-world complexity**: The VIM construction uses controlled rendering (fixed font, uniform position, white padding) that does not reflect real-world scenarios with varied layouts, handwriting, overlapping content, or actual UI screenshots. The claimed practical relevance is not validated on naturalistic data.

- **MathVista performance collapse unexplained**: V-MLLM-7B drops from 25.7 (TEM) to 7.2 (VIM) on MathVista—a far more severe collapse than on other benchmarks (e.g., OKVQA drops from 56.09 to 52.10). This outlier receives no discussion despite being clearly visible in Table 3.

- **No comparison against modular OCR-augmented baselines**: The paper frames VIM as an instruction-following problem, but the root cause appears to be OCR capability. A simple baseline combining an external OCR tool with standard LLaVA would clarify whether the issue is multimodal alignment or text extraction, and whether end-to-end training is truly necessary.

- **TEM baseline anomaly under-explained**: In Table 3, Llama 2 (a pure LLM with no image access) outperforms GPT-4 on 6 out of 8 TEM tasks. This counterintuitive result suggests potential benchmark contamination or label bias, yet the discussion is deferred to Appendix C rather than addressed substantively in the main text.

## Nice-to-Haves

- **Controlled comparison with VIM-finetuned LLaVA-1.5**: Adding a row where baseline LLaVA-1.5 is fine-tuned on VIM data (without other changes) would isolate the contribution of the training data from any other methodological choices.

- **Resolution ablation**: OCR is resolution-sensitive; evaluating VIM across different image resolutions (e.g., 224px, 336px, higher) would clarify how much performance depends on visual acuity versus reasoning capability.

- **Evaluation on naturalistic screenshots**: Testing on real UI screenshots or document images (rather than synthetically rendered text) would validate real-world applicability.

- **Training data contamination audit**: Explicit analysis of overlap between LVIS-Instruct4V-LLaVA-Instruct-mix880k and VIM-Bench test images would strengthen generalization claims.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Title nitpick ("pixelated" vs "printed")**: This is a minor stylistic complaint that does not affect the paper's substance. The title accurately describes the setting.

- **Claim of Figure 2 internal inconsistency**: Upon verification, the 11 models × 8 tasks count is consistent with the MLLMs shown in Table 3 (GPT-4V, GPT-4O, Gemini Pro, Qwen-VL-Chat, InstructBLIP, LLaVA-1.5-7B/13B, LLaVA-1.6-7B/13B, v-MLLM-7B/13B = 11 models). LLM-only models are appropriately excluded from VIM plots.

- **Proprietary model API protocol criticism**: While the paper could provide more details about GPT-4V/Gemini evaluation, the API submission process is standard and doesn't invalidate the main findings.

- **Training instability claim requiring loss curves**: The observation that "training procedure is often unstable" is supported by Table 7 results; demanding loss curves and multiple seeds is an excessive rigor requirement for this empirical observation.

## Novel Insights

The paper reveals an important asymmetry in current MLLM evaluation: open-source models rely heavily on their LLM backbones for instruction following, making them brittle when instructions shift to the visual modality. The diagnostic finding that models can "see" words but not parse semantics (LLaVA-1.5: 29/30 word match vs. 7/30 semantic match) suggests the bottleneck is visual-language grounding, not pixel-level OCR. This points to a fundamental training data gap—current instruction tuning data synthesizes instructions via LLMs in text format, creating an implicit dependency on text-modality prompting.

## Suggestions

- **Add a VIM-finetuned LLaVA-1.5 baseline**: Fine-tune the base LLaVA-1.5 on VIM data using the exact same protocol as v-MLLM to enable direct comparison of training effects.

- **Discuss MathVista outlier explicitly**: Analyze why V-MLLM fails disproportionately on MathVista in VIM—whether due to formula rendering, spatial reasoning demands, or other factors.

- **Move Llama 2 vs. GPT-4 discussion to main text**: This anomaly deserves substantive treatment; if TEM baselines are confounded, the magnitude of VIM degradation is harder to interpret.

- **Add at least one real-world screenshot evaluation**: A small-scale test on actual UI/document screenshots would validate whether synthetic VIM generalizes to practical scenarios.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 5.0]
Average score: 4.0
Binary outcome: Reject
