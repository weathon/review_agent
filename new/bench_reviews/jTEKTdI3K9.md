Now let me search for calibration papers to anchor my score.Now I have enough to write the final review.

---

## Summary

AVHBench introduces the first benchmark specifically designed to evaluate cross-modal hallucination in audio-visual LLMs — cases where one modality induces spurious percepts in the other. The paper proposes a semi-automatic annotation pipeline leveraging existing datasets (VALOR, AudioCaps), delivers 5,302 balanced Yes/No QnA pairs across four tasks (Audio-driven Video Hallucination, Video-driven Audio Hallucination, Audio-visual Matching, Audio-visual Captioning), and demonstrates that existing audio-visual LLMs generally perform worse with multimodal input than with unimodal or text-only input. A training ablation on Video-LLaMA using LoRA fine-tuning and audio feature alignment shows large gains on benchmark tasks and cross-dataset generalization.

---

## Strengths

- **Novel and well-targeted problem framing**: Cross-modal hallucination in audio-visual LLMs is distinct from single-modality hallucination (POPE, CIEM, audio-only benchmarks) and is not covered by any prior benchmark. The paper's decomposition into A→V and V→A hallucination directions is conceptually clean and enables separate analysis of each cross-modal influence direction.

- **Balanced judgment dataset design (Figure 2, Table 1)**: The benchmark contains exactly 2,651 Yes and 2,651 No QnA pairs. This balance enables unambiguous accuracy as a metric and is confirmed by the "Random Choice" baseline consistently at 50% across all judgment tasks.

- **Three-condition diagnostic design (Tables 1–3)**: The progression from multimodal → unimodal → text-only input systematically isolates the contribution of each input format. For models with well-functioning instruction-following (PandaGPT, ChatBridge, OneLLM), the multimodal → unimodal improvement is genuine (e.g., PandaGPT: 58.5% → 65.0% on A→V; OneLLM: 53.7% → 76.5% on A→V).

- **Generalization evidence for the training intervention (Table 5)**: The LoRA + alignment ablation shows strong generalization beyond AVHBench to VAST (CIDEr 0.2 → 47.6) and AVInstruction (Accuracy 43.6% → 57.8%), providing evidence beyond same-distribution template learning.

- **Inclusion of synthetic mismatched audio-video pairs (Section 3.2)**: The 1,030 synthetic videos created by swapping audio introduce controlled natural mismatches that are rare in naturally occurring data and essential for Audio-visual Matching evaluation.

---

## Weaknesses

### Fatal

None.

### Major

- **Yes-bias conflates instruction-following failure with cross-modal hallucination for the majority of evaluated models.** Table 1 shows that Video-LLaMA answers "Yes" 99.9% and 100.0% of the time, ImageBind-LLM 86.7% and 99.3%, and ChatBridge 77.6% and 14.8% (near-total No-bias in the opposite direction). On a balanced benchmark (50% Yes/No), these models mechanically achieve ~50% accuracy by defaulting to a single response rather than processing the cross-modal content. A model answering "Yes" to every query is not demonstrating cross-modal confusion — it is not following the binary-question format. The paper acknowledges "overconfidence" but does not distinguish this failure mode from genuine cross-modal hallucination anywhere in the analysis. The headline interpretation ("audio-visual LLMs are vulnerable to cross-modal hallucinations") is valid for the 3 models with reasonable Yes% rates (PandaGPT ~82%, OneLLM ~63%, ChatBridge more mixed), but should not be presented as a universal finding across all six baselines without qualification. This conflation also inflates the apparent magnitude of the multimodal-vs.-unimodal performance gap.

- **The multimodal-vs.-unimodal comparison does not cleanly isolate cross-modal hallucination as the causal mechanism.** Section 4.2 Question 2 presents the Table 1 vs. Table 2 difference as evidence that "multimodal signals confuse the models' perception," but the same pattern is consistent with at least two alternative explanations: (a) attention dilution — more encoder tokens degrade useful attention to the relevant modality independently of cross-modal confusion; (b) encoder–LLM interface misalignment — several models (PandaGPT, Video-LLaMA, as noted in footnote 3) were not jointly trained with audio, so audio tokens introduce poorly aligned representations that degrade all output quality, not specifically cross-modal perception. The text-only experiment (Table 3, Question 3), where replacing raw audio-visual tokens with LLM-native text captions substantially improves performance, is equally consistent with explanations (a) and (b). No controlled manipulation (e.g., neutral-content or silent audio, content-perturbed audio keeping query modality fixed) is present that would discriminate between cross-modal hallucination and interface degradation as the dominant mechanism. The paper appropriately uses hedged language ("may be a potential factor," "one reason"), but the introductory framing is stated as a near-confirmed finding.

### Minor

- **Training confound in Table 4: template learning vs. genuine robustness.** The LoRA fine-tuning dataset (87,624 QnA pairs) is generated by the same semi-automatic pipeline (same ChatGPT disentanglement + same rule-based question templates) from the training split of VALOR and AudioCaps — the exact same source datasets and question formats as the test benchmark. This means the model sees the same phrasing ("Is {object/event+object} visible in the video?") at both train and test time. Table 5 (VAST, AVInstruction) partially mitigates this concern, but no ablation isolates how much of the AVHBench Table 4 gain comes from format/template learning versus genuine multimodal alignment improvement. The headline claim that "simple training...improves robustness against hallucinations" is somewhat overstated without this ablation.

- **Mixed captioning results weaken the training claim.** Table 4 shows that METEOR decreases from 14.0 (baseline) to 12.2 (Align+FT) for Audio-visual Captioning, while CIDEr and GAVIE-A improve. The paper does not comment on the METEOR regression. This inconsistency in captioning metrics weakens the claim that the fine-tuning improves all aspects of hallucination robustness.

- **Only 6 early-generation (2023) models evaluated.** Whether the cross-modal hallucination phenomenon generalizes to more recent, better-aligned audio-visual models trained with joint audio-visual instruction tuning (released after 2023) is unknown. The paper's relevance claim is implicitly scoped to older architectures.

- **Negative pair difficulty in Audio-visual Matching is uncharacterized.** The construction of negative pairs via random audio swapping (Section 3.1) creates pairs with highly variable semantic distance. No characterization of swap difficulty (e.g., using audio-visual embedding distance) is provided. It is unclear whether near-chance model performance on this task reflects genuine semantic confusion or trivial failure on already easy negatives.

### Trivial

- Footnote 1 flags that OneLLM "is not jointly trained as a multimodal model." This important caveat is buried in a footnote and should be clearly flagged in every table in which OneLLM results appear to avoid misleading readers who scan tables directly.

---

## Nice-to-Haves

- A response-bias-controlled analysis of Table 1 reporting results only for models with Yes-rates in a reasonable range (e.g., 30–70%), or reporting "Yes-bias corrected" accuracy, would allow a cleaner conclusion about cross-modal hallucination robustness separated from instruction-following failure.
- Inclusion of a human performance baseline would establish a meaningful ceiling for the benchmark and help determine whether any items are genuinely ambiguous to humans.
- Error breakdown by semantic category (e.g., which object/event types cause the most hallucinations) would enable more targeted model improvements.
- Evaluation on at least one more recent audio-visual LLM trained with joint audio-visual instruction tuning would strengthen the paper's relevance claim considerably.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic, Section 3.2 — caption completeness bounding QnA quality**: This concern about GPT-4 missing silent background objects is partially addressed by the paper's explicit inclusion of RAM++ visual tags (which do not rely on captions) and human verification in Stage 1. The concern is real but overstated as a weakness given the mitigation steps described.

- **Harsh Critic, Section 4.1 — OneLLM is a "known-broken baseline"**: The paper explicitly flags this in footnote 1. Including a non-jointly-trained model in an analysis of instruction-following and cross-modal perception is odd, but it is transparently disclosed, not a methodological error. This is not a weakness that undermines the paper's conclusions.

- **Harsh Critic, Section 5 — Yes-bias not in Limitations**: Accurate that the Limitations section does not explicitly mention Yes-bias (it does acknowledge "overconfidence" in Section 4.2). This is a presentation issue rather than a substantive flaw beyond what is captured under the major weaknesses above.

- **Strength Finder, diagnosis-driven structure strength**: Removed as a generic presentation praise without citation to specific evidence.

- **Strength Finder, semi-automatic annotation pipeline reduces labeling cost**: This is a standard engineering claim without specific evidence of its value relative to alternatives; removed as generic.

---

## Novel Insights

The paper's most genuinely novel diagnostic contribution is the three-way input experiment (multimodal → unimodal → text-only), which reveals that audio-visual LLMs perform strictly worse than their own LLM backbone when given raw multimodal tokens compared to text representations of the same content. This points to a specific bottleneck in encoder–LLM feature alignment rather than in LLM reasoning capability per se, and is a constructive framing for future architectural work. The finding that audio-driven hallucinations and video-driven audio hallucinations are distinct failure modes with different severity profiles (models degrade more on V→A than A→V in Table 1) is also a useful directional diagnostic for future benchmark and model work.

---

## Suggestions

1. Explicitly filter or flag degenerate responders (Yes% > 90% or < 10%) in all tables, and present a separate analysis limited to models with reasonable response distributions. This would separate the benchmark's discrimination capacity from instruction-following collapse.
2. Add a neutral-audio or silent-audio control condition for the A→V task to provide evidence that audio content (rather than mere token dilution) is the driver of visual hallucination.
3. For Table 4, evaluate the fine-tuned model using a held-out question template (e.g., paraphrased questions) to quantify template-learning vs. genuine robustness.
4. Report the percentage of annotator corrections made in Stages 1 and 2 in the main paper (the paper says details are in the appendix, but this number is informative for judging annotation quality).

---

## Score and Decision

**Calibration anchors reviewed:**

| Paper | Path | Avg Score | Decision | Comparison to AVHBench |
|---|---|---|---|---|
| ViLMA (video-language benchmark) | `/human_reviews/liuqDwmbQJ.md` | 6.0 | Accept (poster) | More methodologically clean (counterfactual design); cleaner causal isolation. AVHBench is slightly weaker. |
| OmniBench (tri-modal benchmark) | `/human_reviews/Rc8z5wLzBF.md` | 5.75 | Reject | Closest topical analog. AVHBench has larger dataset, training ablation, and cross-dataset validation; OmniBench criticized for small size. AVHBench comparably strong or slightly stronger. |
| LUMA (multimodal uncertainty benchmark) | `/human_reviews/lh0iTFCD1y.md` | 4.0 | Withdrawn | Weaker than AVHBench: artificial data, limited real-world relevance. AVHBench clearly stronger. |
| AutoHall (hallucination dataset) | `/human_reviews/LlG0jR7Yjh.md` | 3.67 | Reject | Much weaker: unclear applicability, lack of novelty. AVHBench clearly stronger. |
| FAITHQA (hallucination benchmark) | `/human_reviews/RuY1r1PDdQ.md` | 3.0 | Reject | Much weaker: unclear writing, missing dataset construction. AVHBench clearly stronger. |

**Positioning:** AVHBench addresses a genuinely novel and underexplored problem (cross-modal AV hallucination), has a well-designed balanced dataset, and provides useful cross-dataset generalization evidence. These features put it above LUMA and the low-scoring anchors. However, the Yes-bias confound for 3 of 6 models is a real methodological limitation that weakens the core interpretation, the causal attribution to "cross-modal hallucination" as opposed to encoder–LLM interface degradation is not cleanly established, and the training confound in Table 4 is partially unresolved. These issues position it below ViLMA (6.0, clean methodology). OmniBench (5.75, Reject) is the closest comparable, and AVHBench is roughly at the same level — a meaningful contribution with real methodological gaps that prevent confident acceptance.

**Final score: 5.0** — Borderline reject. The benchmark dataset itself is a useful community contribution, but the analytical overclaiming and methodological gaps in the evaluation of degenerate baselines mean the paper's stated conclusions exceed what its experiments actually demonstrate.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>