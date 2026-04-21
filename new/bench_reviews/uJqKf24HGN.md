Now I have all the information needed to write the final review. Let me synthesize everything carefully.

## Summary

UniCon proposes a unidirectional information flow paradigm for controlling diffusion models: instead of bidirectional interaction where the adapter modifies intermediate features of the diffusion model (as in ControlNet), the adapter receives features from the frozen diffusion model and directly outputs the denoised result. This eliminates the need to compute and store gradients for the diffusion model during adapter training, yielding ~40% VRAM reduction and ~2.3× training speedup. The design is validated on both U-Net (SD-2.1) and transformer (PixArt-α/DiT) architectures across five conditional generation tasks.

## Strengths

- **Genuinely novel architectural paradigm.** The unidirectional flow design is a non-obvious and clean departure from ControlNet's bidirectional approach. By having the adapter produce the final output rather than inject residuals back into the diffusion model, UniCon eliminates gradient computation through the frozen model entirely. This is both conceptually simple and practically impactful — a rare combination.

- **Significant and directly measured efficiency gains.** Figure 6 provides transparent, component-level VRAM analysis (weight, activation, gradient, optimizer) and training time comparisons under controlled conditions (same GPU, batch, pre-computed features, BF16). DiT ControlNet-Full requires ~23GB VRAM while UniCon-Full requires ~14GB, and training time is roughly halved. These are concrete, verifiable claims — not relative comparisons depending on baselines.

- **Architecture-agnostic validation.** The paper demonstrates the adapter on both U-Net (SD-2.1) and transformer (PixArt-α) architectures (Table 2, Figure 2), directly addressing its stated motivation about transformer-based diffusion models. This is the paper's most compelling practical contribution given the field's shift toward DiT-style architectures.

- **Strong main comparison results.** Table 2 shows UniCon outperforms ControlNet across all DiT tasks on both controllability and FID (e.g., DiT-SR: PSNR 34.82→37.34, FID 26.43→20.34; DiT-Canny: SSIM 0.4748→0.5458, FID 51.52→46.71). The gains are consistent and substantial, especially for low-level tasks.

- **Insightful ablation on encoder vs. decoder control.** Table 1a reveals that encoder-focused ControlNet yields better FID while decoder-focused yields better controllability, motivating the full-network adapter design. Figure 4 cleanly demonstrates that discarding part of the frozen diffusion model causes severe quality degradation, establishing the importance of preserving the full pre-trained model.

- **ZeroFT connector contribution.** Table 1b shows the proposed ZeroFT connector (combining addition, multiplication, and shortcut) consistently outperforms ZeroMLP and ShareAttention across tasks.

## Weaknesses

### Fatal
None.

### Major

- **Data error in central ablation table (Table 1c).** The SR (PSNR) section of Table 1c contains SSIM-range values (0.5053, 0.5458) in the controllability column labeled PSNR. PSNR for images is measured in dB (typically 20–50); values of 0.5 are characteristic of SSIM. Cross-referencing with Table 1a confirms the mismatch: Table 1a reports SR Skip-Layer ControlNet PSNR=35.49 while Table 1c's corresponding row shows 0.5053; the FID values also diverge (Table 1a: 24.99 vs Table 1c: 50.17). The first two rows of the SR section appear to contain Canny data mistakenly placed in SR rows. While the key Full ✗ vs Full ✓ comparison for SR (PSNR 36.53→37.34, FID 23.04→20.34) remains intact and supports the paper's claims, this error renders the Skip-Layer and Decoder bidirectional comparisons for SR uninterpretable. A data error of this nature in the paper's central ablation — the experiment that most directly tests the core claim — raises concerns about experimental diligence.

- **Missing ControlNet-XS comparison.** ControlNet-XS (cited in Related Work, line 69) specifically addresses the efficiency limitations of ControlNet that UniCon targets — it explores smaller adapter sizes and modified architectural designs to reduce computational overhead. It is the most directly relevant baseline for UniCon's efficiency claims. Its absence from all experiments leaves the efficiency comparison incomplete, since UniCon's efficiency advantage is measured only against standard ControlNet, not against methods designed to improve upon ControlNet's cost.

### Minor

- **Controllability–quality trade-off for Canny in Table 1c not acknowledged.** In the ablation (Table 1c, using ZeroMLP), UniCon Full ✓ for Canny achieves SSIM=0.5343 but FID=55.22, while Full ControlNet bidirectional (from Table 1a) achieves SSIM=0.5053 and FID=50.17. This shows improved controllability (+0.029 SSIM) but degraded generation quality (+5.05 FID). The paper claims unidirectional flow "substantially enhances performance, improving controllability and generative quality in both high-level and low-level tasks" (Section 4.2) without acknowledging this trade-off. Note that Table 2 (using ZeroFT connector) shows UniCon winning on both metrics for Canny, so the trade-off may be connector-specific, but the paper should discuss this rather than leave it implicit.

- **Condition injection mechanism underspecified.** The adapter formally takes (c, t, p, X_h) as inputs (Section 3), but the text does not explicitly describe how the control condition c enters the adapter architecture. Figure 2(a) shows PatchEmbed at the adapter's input, which presumably processes c, but this is never stated. For a method paper where the adapter produces the final output (rather than modifying intermediate features), the condition injection pathway is architecturally significant and affects reproducibility.

- **Dismissal of T2I-Adapter quality advantage not well justified.** For SD U-Net Depth, Table 2 shows T2I-Adapter outperforms UniCon on Clip-IQA (0.6906 vs 0.6807) and MAN-IQA (0.2331 vs 0.2262). The paper dismisses this because "the control effect of the T2I method is not good," but the controllability difference is moderate (MSE 87.72 vs 85.00). The dismissal would be more convincing with a clearer threshold for what constitutes "good enough" controllability.

- **SUPIR-UniCon claim lacks quantitative evaluation.** The paper claims UniCon "effectively addresses" SUPIR's scalability limitations and shows SD3+UniCon results (Figure 8), but provides only three visual examples with no metrics. This is insufficient to support claims about broad applicability to new architectures.

### Trivial
None.

## Nice-to-Haves

- Analyze the controllability–quality trade-off explicitly, even if it is connector-specific, to provide design guidance for practitioners.
- Include ControlNet-XS as a baseline in efficiency comparisons to complete the efficiency picture.
- Add multi-step inference quality analysis (FID/controllability vs. denoising steps) to assess whether the train-test distribution mismatch affects quality at different step counts.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"One-third" memory claim inconsistency (Harsh Critic):** The abstract states "reduces GPU memory usage by one-third" while the body text says "saving nearly half the storage required for gradients." These refer to different quantities: the body text specifies gradient storage specifically, while the abstract refers to total VRAM. From Figure 6, DiT ControlNet-Full uses ~23GB and UniCon-Full uses ~14GB, a reduction of ~9GB/23GB ≈ 39%, which is approximately "one-third." The claims are consistent when read carefully — removed as the critic conflated two different measurements.

- **"Same trainable parameters" comparison is misleading (Harsh Critic):** The critic argues that equal parameter counts don't make architectures comparable because UniCon's adapter directly produces the output. However, the point of Figure 1(c) IS to show that even with the same parameter budget, UniCon achieves better performance AND lower cost. The architectural difference is exactly what the paper argues for. This is a controlled comparison for training budget, not a claim of architectural equivalence — removed as it misinterprets the comparison's purpose.

- **Encoder/decoder terminology misleading for transformers (Harsh Critic):** The paper itself acknowledges this (Section 4.2, line 181): "This proves that for DiT, distinguishing between the encoder and decoder is not effective, and we should leverage the capabilities of different parts of the entire diffusion model." The terminology is used for organizational convenience and the paper explicitly notes its limitations — removed as the criticism ignores the paper's own addressal.

- **No error bars or repeated runs (Harsh Critic):** Single-run evaluation without variance reporting is standard practice for large-scale diffusion model comparisons in this community. The improvements on most tasks are substantial enough (e.g., FID 26.43→20.34, PSNR 34.82→37.34) that statistical significance is not the primary concern — removed as a non-standard demand.

- **UniCon-half uses compromised architecture (Harsh Critic):** The critic argues UniCon-half's skip-layer design makes it an unfair parameter-matched baseline. However, UniCon-half is included precisely to show that the full-network adapter is important for UniCon (as the ablation also demonstrates). It is not presented as the primary comparison — it is an additional data point. The main comparison in Table 2 is ControlNet vs full UniCon — removed as it mischaracterizes the role of UniCon-half.

- **Figure 6 includes T2I-Adapter but Table 2 excludes it for DiT (Harsh Critic):** T2I-Adapter is designed for SD U-Net models and may not have a native DiT implementation. Its inclusion in Figure 6's cost analysis (where it can be measured regardless of architecture) but absence from DiT performance comparisons is natural, not inconsistent — removed as it misunderstands the method's applicability.

- **Strength: "consistent improvement in controllability and generation quality" (Strength Finder):** While Table 2 shows broad improvements, the Canny trade-off in Table 1c (with ZeroMLP) and the T2I-Adapter quality advantage on SD Depth mean the improvement is not universally "consistent." This strength should be qualified — the claim holds for the final model (ZeroFT) in Table 2 but not uniformly across all experimental settings.

- **Strength: "demonstration on SUPIR-UniCon" (Strength Finder):** The SUPIR-UniCon results are purely qualitative (3 visual examples, no metrics) and do not constitute strong evidence of scalability. Downgraded from a supporting strength.

## Novel Insights

The most interesting finding in this paper is not just the efficiency gain but the architectural implication: when you give the adapter direct output responsibility (rather than having it modify intermediate features processed by a frozen decoder), the adapter can develop its own generative capabilities that complement — rather than compete with — the frozen model's features. Figure 4's result (that discarding part of the frozen model degrades quality severely) combined with the ablation showing UniCon improves on full ControlNet suggests an elegant decomposition: the frozen model provides stable, high-quality feature extraction, while the trainable adapter provides flexible, condition-responsive output generation. This is a more principled division of labor than ControlNet's approach of injecting modifications into intermediate features of a model that was not trained to receive them.

## Suggestions

- Fix the Table 1c SR data error before camera-ready. The Skip-Layer and Decoder bidirectional rows appear to contain Canny values (or SSIM values in a PSNR column). Correcting this would make the central ablation fully interpretable.
- Add ControlNet-XS to the efficiency comparison (at minimum in Figure 6's VRAM/speed analysis) to demonstrate UniCon's advantage relative to the most relevant existing efficiency improvement for ControlNet.
- Explicitly discuss the controllability–quality trade-off visible in Table 1c for Canny with ZeroMLP, even if ZeroFT resolves it. Understanding when and why trade-offs occur provides more design insight than reporting only the best configuration.
- Specify how the condition c enters the adapter (e.g., "c is processed through PatchEmbed at the adapter input, consistent with the original DiT architecture") to close the reproducibility gap.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| IC-Light | u1cQYxRI1H | 10.0 | Far above UniCon — physically grounded method with perfect evaluation |
| SANA | N8Oj1XhtYZ | 8.5 | Well above — strong speed/quality trade-off with polished execution |
| MGFR | m9RNBZewW2 | 7.33 | Above — novel adapter with clean experiments, new dataset |
| 3D-Adapter | C0HDYvGwol | 5.6 | Comparable — novel plug-in module but with comparison gaps |
| LISA | PLgHiJOjcH | 4.5 | Below UniCon — lightweight adapter but limited novelty/evaluation |
| SparseDM | 3kADTLbKmm | 4.0 | Below — efficiency method with limited scope |
| ELR-Diffusion | edx7LTufJF | 2.5 | Far below — data inconsistencies across tables, missing LoRA baseline |
| PDE-Diffusion | 3sOE3MFepx | 2.2 | Far below — placeholder values, flawed methodology |

UniCon sits above 3D-Adapter (5.6) because its efficiency gains are more concrete and directly measured, and its architectural paradigm is more fundamental (a new design paradigm vs. a new plug-in). It sits below MGFR (7.33) because MGFR has cleaner experimental execution, a new dataset contribution, and no data errors. The Table 1c data error is a real concern, but unlike ELR-Diffusion (2.5) where performance differs between tables without explanation, UniCon's error is localizable to 2 rows in one sub-table and the key comparisons remain valid. The paper's core claims are supported by the interpretable data.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>