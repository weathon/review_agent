Now I have enough context from the paper and calibration papers. Let me synthesize the review.

## Summary

The paper proposes FOLK, a self-supervised learning framework that extends Masked Frequency Modeling (MFM) with two key ideas: (1) adaptive Com/RCom frequency filters that select frequencies to mask based on per-image magnitude (inspired by Fourier image compression), rather than using fixed-radius low/high-pass filters; and (2) a self-distillation teacher-student branch that exposes the model to unmasked natural images during pre-training, addressing MFM's limitation of only seeing frequency-masked images. FOLK combines a reconstruction loss (student predicts masked frequencies) with a distillation loss (student mimics teacher's representations of original views), and is compatible with both ViT and CNN architectures.

## Strengths

- **Well-motivated problem identification.** The two limitations of MFM are clearly articulated with visual evidence (Figure 1 demonstrates how fixed low/high-pass filters produce overly simple blurry or edge-only images). The paper makes a compelling case that MFM's exclusive exposure to frequency-masked images harms downstream adaptation.

- **Intuitive filter design grounded in signal processing.** The Com/RCom filters, inspired by Fourier image compression, are a natural and elegant extension of MFM. By masking the highest-magnitude frequencies (Com) or lowest-magnitude ones (RCom), the model is forced to reconstruct either fine details from coarse semantics or vice versa—a principled alternative to fixed-radius filtering.

- **Strong few-shot learning results.** Table 2 demonstrates a substantial gap between FOLK and MFM in the 10% ImageNet setting (71.2% vs. 58.5% MAX at 300 epochs), validating the core motivation that the distillation branch's exposure to natural images aids downstream adaptation under limited data.

- **Architecture-agnostic design.** FOLK works with both ViT and CNN architectures (ResNet-50 results in Appendix), unlike MIM methods that are largely restricted to ViTs, which is a practical advantage.

- **Clear ablation of Com/RCom filters without distillation.** The "MFM + Com/RCom" rows in Tables 1 and 2 isolate the contribution of the informed filters alone, showing meaningful improvement over MFM in few-shot (66.3% vs. 58.5% MAX) and confirming that both components contribute.

## Weaknesses

### Major

- **Overclaiming about "state-of-the-art" and "fewer epochs" without controlled comparisons.** The paper states FOLK "surpassing all other methods…under similar conditions or lower number of pre-training epochs" (Section 4.2.1), but Table 1 compares methods trained with different schedules (MAE/BEiT 300, MoCo v3 600, iBOT 1600, FOLK 800). FOLK-800 matches iBOT's 84.0% at 1600 epochs, but without an 800-epoch iBOT or DINO baseline, the efficiency claim is unsubstantiated. The claim "with only 300 epochs, FOLK achieves 81.6%/83.4%" is more fairly stated as "competitive with methods at various epochs" — the 0.2–0.3% margin over MFM+R/Com at 300 epochs (Table 1: 83.4 vs. 83.2) is within noise. This overclaiming matters because the paper's headline positioning rests on efficiency and superiority.

- **Insufficient ablation isolating distillation from filter design.** The paper's two contributions are adaptive filters and self-distillation, but the ablation does not isolate them cleanly. The critical missing condition is FOLK with self-distillation but using original MFM low/high-pass filters (instead of Com/RCom). Currently, the paper shows: MFM → MFM+Com/RCom → FOLK, but without "MFM+distillation with fixed filters," we cannot determine how much of FOLK's gain comes from distillation vs. filter design. In Table 1, adding distillation to MFM+Com/RCom (i.e., FOLK vs. MFM+R/Com*) yields only 0.2% improvement (83.4 vs. 83.2 for ViT-B), suggesting distillation contributes very little in the full fine-tuning regime and the core benefit may be from the filter change or from adopting DINO-style augmentation/training recipes. This significantly weakens the paper's causal claims about both components.

- **"Few-shot" evaluation is non-standard and potentially misleading.** The paper labels the 10% ImageNet (≈128K images) fine-tuning experiment as "few-shot learning" (Section 4.2.2), but this is conventionally called "low-data fine-tuning" or "semi-supervised learning." Standard few-shot benchmarks use 1–5 examples per class. Additionally, the comparison uses only three hyperparameter settings across all methods, and iBOT's 2.0% accuracy at BLR=2e-3/WUp=5 is clearly a training failure that drastically skews the AVG metric, making FOLK's robustness advantage appear larger than it may be.

- **No linear probing evaluation.** All ImageNet results use full fine-tuning, which conflates representation quality with fine-tuning capacity. Every major SSL method (DINO, iBOT, MAE, MoCo v3) reports linear probing accuracy. Its absence makes it impossible to assess whether FOLK learns inherently better representations or simply fine-tunes more effectively—a critical distinction for the paper's claims.

### Minor

- **Computational cost not reported.** FOLK doubles the forward-pass cost with a teacher-student architecture and adds an EMA update, an extra student head, and a distillation loss. Without reporting wall-clock time, GPU memory, or FLOPs relative to MFM and other baselines, it is unclear whether FOLK's gains justify the added cost, or whether comparable gains could be achieved by simply training MFM longer under a similar compute budget.

- **Threshold values for Com/RCom filters are under-justified.** The threshold is uniformly sampled from {0.005, 0.01, 0.05} with no principled justification in the main paper. The sensitivity analysis is deferred to Appendix B.5.1—since this hyperparameter directly controls the masking behavior and is a core design choice, a brief justification or sensitivity discussion in the main text would strengthen the contribution.

- **Grayscale filter generation loses color frequency information.** The Com/RCom masks are generated from grayscale versions and applied uniformly to all RGB channels. The paper asserts per-channel filtering "can result in unnatural and corrupted visual information" (Section 3.2.1) but provides no empirical comparison or visual evidence. This is a design choice worth validating.

### Trivial

- None beyond the above.

## Nice-to-Haves

- Comparison with concurrent frequency-based SSL methods (SpeeD, FCMAE) cited in the introduction but never evaluated against.
- Scaling experiments to larger models (ViT-L) or larger pre-training datasets (ImageNet-22K).
- Standard few-shot benchmarks (e.g., 5-way 5-shot on mini-ImageNet) to validate the "few-shot" claim in a more recognized protocol.
- Reporting variance across multiple runs for the margins in Table 1, which are very small (0.2–0.3%).

## Novel Insights

The paper's most interesting observation is the stark difference between full fine-tuning and the 10% data regime: the distillation branch adds only ~0.2% in full fine-tuning (Table 1) but contributes a much larger gain in the limited-data setting (Table 2: FOLK 71.2 vs. MFM+Com/RCom 66.3 MAX). This suggests that the distillation branch's primary role is improving adaptation to natural image distributions rather than learning fundamentally better features—supporting the paper's stated motivation but also highlighting that the contribution is regime-specific. This asymmetry deserves more explicit discussion, as the paper's headline claims about "competitive performance" broadly are primarily supported by full fine-tuning results where the distillation benefit is marginal.

## Suggestions

- **Add a "MFM + distillation + fixed filters" ablation** to cleanly isolate the contribution of each component. This is the single most important experiment for validating the dual-contribution narrative.
- **Tone down the "surpassing state-of-the-art" and "fewer epochs" claims** to match what the evidence actually shows: FOLK is competitive with contemporary methods and substantially improves over MFM, particularly in the limited-data regime.
- **Add linear probing results** to disentangle representation quality from fine-tuning strategy.
- **Rename "few-shot" to "low-data fine-tuning"** and acknowledge the iBOT outlier in the AVG metric.
- **Report computational cost** (wall-clock time, GPU memory) to contextualize efficiency claims.

---

## Calibration Papers

I compared this paper against:

1. **Exploring Target Representations for Masked Autoencoders (dBOT)** — Accept (poster), scores 6/6/6/3. Similar pattern: combines masking + distillation for SSL, empirically strong, but reviewers noted questions about novelty and scalability. Scored ~5.3 average.

2. **Towards Understanding Masked Distillation** — Reject, scores 5/3/1/3. SSL paper analyzing masked distillation empirically, lacking controlled downstream comparison. Scored ~3 average. More analytical/understanding paper, weaker.

3. **EMP-SSL** — Withdrawn (Reject), scores 3/5/5/3. SSL efficiency paper making epoch-based comparisons without compute-cost matching, small-dataset experiments only. Very similar weakness pattern (unfair epoch comparison, missing compute budget analysis). Scored ~4 average.

4. **Depth-Guided SSL** — Reject, scores 5/3/3. Augmentation-based SSL improvement with non-standard evaluation, limited novelty (combining existing components). Similar pattern of combining an existing SSL framework with a new signal. Scored ~3.7 average.

5. **Anti-Exposure Bias in Diffusion Models** — Accept (Spotlight), scores 6/8/8/6/8. Strong novelty, well-motivated, extensive experiments across settings, clear improvement. Scored ~7.2. This is the high end — FOLK doesn't achieve this level of novelty or experimental thoroughness.

6. **Mutual Effort for Efficiency (SimPrune)** — Accept (Poster), scores 6/6/6/6. ViT SSL efficiency paper, incremental but well-executed. Scored 6. FOLK is weaker than this due to missing ablations and overclaiming.

FOLK is most similar to EMP-SSL and Depth-Guided SSL in its weakness profile (combining existing ideas, non-standard evaluation, efficiency claims not grounded in compute budget), but is somewhat stronger than those because it has a concrete, clearly motivated baseline (MFM) and a substantial empirical improvement over it in the limited-data regime. It is weaker than dBOT (accepted poster), which had cleaner ablations and a stronger empirical story. I position FOLK below dBOT's ~5.3 norm, closer to the EMP-SSL/Depth-Guided SSL range of 3.5–4.5, tempered upward by the genuine improvement over MFM in the few-shot regime.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>