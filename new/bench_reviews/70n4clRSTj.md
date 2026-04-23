## Summary

The paper introduces SpookyBench, a benchmark where information is encoded exclusively in temporal sequences of noise-like frames through opposing motion patterns (foreground vs. background noise move in opposite directions), rendering individual frames indistinguishable from noise. Across 451 videos spanning text, object images, and dynamic scenes, humans achieve ~98% accuracy while 27 Video-VLMs (2B–78B, open and closed source) all achieve 0%, including under chain-of-thought prompting, varied frame rates, and fine-tuning. The paper frames this gap as evidence of "time blindness" — a fundamental architectural inability of current VLMs to process purely temporal information.

## Strengths

- **Creative and clean benchmark design**: Encoding information exclusively through opposing motion patterns in noise frames (Algorithms 1 and 2) is a genuinely novel stimulus construction that, unlike prior temporal benchmarks (TemporalBench, TVBench), completely eliminates spatial shortcuts. The operationalization of "purely temporal understanding" is well-motivated and elegant (Figure 2).

- **Dramatic and important finding**: The 0% vs. 98% accuracy gap across all 27 models — including GPT-4o, Gemini 1.5 Pro, and specialized temporal models like TimeChat — is a striking observation that the community needs to be aware of. The consistency across scales, architectures, and prompting strategies makes this compelling (Table 1).

- **Rigorous quantitative characterization**: The four SNR metrics (Equations 1–4, Table 2) — Basic, Perceptual, Temporal Coherence, and Motion Contrast SNR — provide a principled framework for characterizing the stimuli, going beyond qualitative description.

- **Valuable control experiments**: The frame-rate ablation (Tables 4–5) demonstrates that humans degrade gracefully with decreasing FPS (95.6%→0%) while VLMs remain at 0% across all rates, ruling out temporal undersampling as the sole explanation. The fine-tuning experiment (Section 4.4) partially addresses distribution shift as an alternative explanation.

- **Comprehensive model coverage**: 27 models spanning 2B–78B parameters, open- and closed-source, with both direct and CoT prompting, provides strong evidence that the failure is systemic rather than model-specific.

- **Flexible evaluation protocol**: Accepting multiple valid labels for Object Images and Dynamic Scenes (Section 4.1) makes the 0% result even more striking, as models fail to match any acceptable response.

## Weaknesses

### Fatal
None.

### Major

- **The diagnostic claim of "time blindness" / "fundamental architectural inability" exceeds what the experiments can support**: The paper's central thesis is that VLMs have "an inability to extract meaning from temporal cues" (Abstract) and a "fundamental architectural inability to process information conveyed purely through motion" (Section 4.4). However, the experimental design cannot distinguish between two failure modes: (a) the ViT backbone produces meaningless embeddings for individual noise frames (a *frame-encoding* failure), and (b) the temporal aggregation module cannot integrate frame-level features into coherent motion patterns (a *temporal-processing* failure). If every frame is encoded as noise by the ViT, no temporal module could recover the signal regardless of its sophistication. The paper attributes the failure to (b) without ruling out (a). The critical missing experiment — providing precomputed optical flow or frame-difference maps as input instead of raw noise frames — would resolve this ambiguity. If models succeed with explicit temporal features, the bottleneck is at encoding; if they still fail, "time blindness" gains credibility. Without this, the paper demonstrates *that* VLMs fail on temporally-encoded stimuli, but cannot confidently claim *why*. The paper's own Section 5 lists "lack robust temporal integration mechanisms" as a failure mode, but this is asserted rather than tested.

- **The fine-tuning experiment is insufficient to support "fundamental architectural inability"**: Section 4.4 reports that fine-tuning InternVL2.5-8B and Qwen2-VL-7B on 400 videos for 10 epochs yields 0% test accuracy. While this usefully rules out trivial distribution mismatch, concluding "fundamental architectural inability" from this is overclaimed because: (a) 400 training samples is small; (b) the fine-tuning uses LoRA via LlamaFactory, which does not modify the ViT's frame-independent encoding bottleneck — so the experiment tests whether fine-tuning the *existing architecture with limited data* works, not whether *the architecture is fundamentally incapable*; (c) no training diagnostics (loss curves, training accuracy) are reported, so we cannot tell whether models learned anything at all during training. These are different questions, and the paper draws the stronger conclusion without adequate evidence.

### Minor

- **Table 5 reports VLM accuracy averaged across all frame rates without per-FPS breakdown**: The human data (Table 4) shows a clear per-FPS degradation pattern. Table 5 reports only "0.0% averaged across all tested frame rates (1-30 FPS)" for VLMs without showing per-FPS values. While the conclusion (VLMs fail at all frame rates) is likely correct, showing per-FPS breakdown would make this more transparent. Additionally, the paper does not report how many frames each model actually processes per video — VLMs typically subsample to 8–32 frames regardless of input FPS, so the "same FPS" comparison between humans and VLMs may not be equivalent in terms of actual temporal information density received.

- **Section 3.3.2 on the binary SNR threshold effect is confusingly written**: The "40% accuracy" for prompts and the "85.7% above 2.5dB threshold" are not clearly reconciled with the 0% results in Table 1. The section appears to conflate prompting strategy results with human detection thresholds, making it hard to parse.

- **Limited qualitative analysis of model outputs**: Section 5 briefly notes that models "attempt to extract information from individual frames rather than temporal patterns" and that "fine-tuned models produced outputs that mimicked training examples," but provides no representative examples. Showing actual model outputs (e.g., a table of prompts and responses for a few videos) would reveal whether models produce gibberish, default responses, or close-but-wrong answers — critical for understanding the failure mode and guiding future work.

- **Small number of human participants (6)**: While inter-annotator agreement is high and the effect size is massive, six participants is a small sample. The paper appropriately reports per-annotator results (Table 3), but the 98% figure would benefit from a brief acknowledgment of this limitation.

### Trivial
None.

## Nice-to-Haves

- Testing with pre-extracted optical flow or frame differences as input instead of raw frames — this would directly resolve the encoding vs. integration ambiguity and is the single most impactful addition possible.
- Testing with a model architecture that processes frame differences natively (e.g., a 3D CNN or dual-stream architecture with a dedicated temporal pathway) to determine whether "time blindness" is specific to the ViT-based paradigm or truly architectural.
- Per-model frame count reporting in the main evaluation and per-FPS VLM breakdown in Table 5.
- Training diagnostics for the fine-tuning experiment (loss curves, training accuracy).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Neuroscience motivation disconnected from experiments"** (from Harsh Critic): The neuroscience references (Section 1, 2.2) provide background motivation for why temporal processing is important. While no neuroscience-inspired architecture is tested, the paper's contribution is a benchmark, not an architecture. Criticizing the absence of neuroscience-based solutions is scope creep.

- **"Humans watch at native frame rates while VLMs subsample — not adequately controlled"** as a fatal issue (from Harsh Critic): The frame-rate experiment (Table 4) partially addresses this by showing humans succeed at 10 FPS (59.4%) and 5 FPS (12.8%) — frame rates well within VLMs' typical processing range. While per-model frame counts would be informative, the existing data already shows that even at reduced temporal density, humans succeed where VLMs fail.

- **"Paper should test whether providing optical flow as additional input channel enables models to solve the task"** as a fatal issue (from Harsh Critic): While this is a very valuable experiment, the paper's contribution is identifying and characterizing the failure, not proposing solutions. The absence of this experiment weakens the diagnostic claim but does not invalidate the benchmark itself.

- **Generic request for "larger dataset"** (implicit in some critiques): The dataset has 451 videos and the generation is deterministic and reproducible. The 0% result across 27 models is stable enough that dataset size is unlikely to change findings.

- **"Missing related works"** claims: Cannot verify without external sources.

## Novel Insights

The paper reveals a striking asymmetry in the human–machine perception gap: when spatial cues are completely eliminated, humans' temporal processing mechanisms (likely relying on motion-based figure-ground segregation via opposing flow) effortlessly recover content, while every tested VLM architecture fails. The frame-rate experiment (Table 4) further reveals that human temporal perception has a nonlinear, near-threshold behavior (0% at 1 FPS, 12.8% at 5 FPS, 59.4% at 10 FPS, 95% at 20 FPS) suggesting that human temporal integration operates with a critical sampling threshold around 5–10 FPS for this stimulus class. The negative motion contrast in Dynamic Scenes (−3.18 dB in Table 2) is particularly interesting — it means the foreground-background motion differentiation that humans exploit is actually *reversed* or negligible by standard metrics, suggesting that human perception uses qualitatively different grouping principles than those captured by the SNR framework.

## Suggestions

- Reframe the conclusion from "VLMs are time-blind" (which implies a temporal-processing deficit) to "the spatial-first paradigm fundamentally fails when spatial cues are absent" (which is precisely what the data shows). This is still a strong and important claim, but one that the experiments actually support.
- Add the optical flow / frame-difference input experiment even as a small pilot — it would dramatically strengthen the paper regardless of the result.
- Report training loss and training accuracy for the fine-tuning experiment to show whether models learned anything at all during fine-tuning.
- Include a small table of representative model outputs (2–3 videos × 2–3 models) in the discussion section.

## Score and Decision

**Calibration anchors compared against:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| Real-3DQA (3D-LLM spatial benchmark + solution) | 7.0 | SpookyBench has a more dramatic finding but lacks solutions and the diagnosis is less rigorous |
| CubeBench (0% on long-horizon tasks, creative diagnostic benchmark) | 5.2 | SpookyBench is comparable — both show dramatic failure rates with creative designs, but CubeBench had more diagnostic framework |
| Euclid/Geoperception (geometric perception failure + solutions) | 5.0 | SpookyBench has a more dramatic finding and more creative design but no solutions |
| ConservationBench (VLMs can't reason about physics, 34 models) | 4.0 | SpookyBench has a far more dramatic gap (0% vs 98%) and more creative design |
| Bio-FM robustness (overclaimed diagnosis, unsurprising findings) | 4.0 | SpookyBench has a much more striking finding but similar overclaiming issues |
| Vinoground (temporal counterfactual VLM benchmark) | 3.0 | SpookyBench is clearly stronger — more dramatic gap, more creative design, more model coverage |
| T2ICountBench (near-zero counting, overclaimed) | 4.5 | SpookyBench has a more important finding |

SpookyBench sits above the "overclaimed diagnosis + limited analysis" papers (4.0–4.5) because the finding is genuinely dramatic (0% vs 98% across 27 models) and the benchmark design is creative and novel. It sits below CubeBench (5.2) because the overclaiming is more severe and the diagnostic framework is weaker. It's comparable to Euclid (5.0) — dramatic failure + creative design but no solutions — but SpookyBench's finding is more striking.

The observation is important enough to warrant community attention, but the overclaimed diagnosis and the missing key experiment that would validate it hold the paper back. The paper demonstrates *what* fails (dramatically and convincingly) but doesn't adequately demonstrate *why*.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>