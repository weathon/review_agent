## Summary

SigMap proposes a multimodal foundation model for wireless localization that combines (1) a cycle-adaptive masking strategy for self-supervised pre-training on CSI data to avoid periodic shortcut learning, and (2) a "map-as-prompt" framework that encodes 3D geographic information via GNN-derived soft prompts for parameter-efficient cross-scenario adaptation. Experiments on ray-tracing benchmarks (DeepMIMO and WAIR-D) demonstrate state-of-the-art localization performance with strong few-shot transfer to unseen environments.

## Strengths

- **Well-motivated and creative architectural design**: The map-as-prompt mechanism—encoding 3D building geometries and base station positions via a GNN into soft prompt tokens prepended to a frozen transformer—is conceptually clean, interpretable, and parameter-efficient. The ablation from 2.275 m to 1.564 m MAE (Table 4, single-BS) with map conditioning directly demonstrates the value of geographic information.

- **Strong empirical gains**: The method achieves a 34.4% MAE reduction over LWLM in single-BS NLoS localization (Table 1), 18.7% in multi-BS (Table 2), and 53.2%/44.3% on unseen DeepMIMO O2 and WAIR-D scenarios (Table 4.5), with consistent improvements across MAE, RMSE, and CDF@1m.

- **Comprehensive ablation studies**: Table 3 (masking strategies), Table 4 (map fidelity: 3D vs. 2D vs. none), and the generalization experiments provide meaningful insight into each component's contribution.

- **Concrete parameter efficiency**: Only 0.085M parameters (0.7% of total) are trained during fine-tuning, completing in 30 minutes (Table 5). This is practically meaningful and clearly reported.

## Weaknesses

### Major

- **"Zero-shot generalization" claim is factually contradicted by the experiments**. The abstract states the model exhibits "strong zero-shot generalization in unseen environments," but Section 4.5 clearly fine-tunes "downstream task heads using limited target samples (approximately 100 instances per scenario)." This is *few-shot supervised transfer*, not zero-shot. No experiment evaluates performance without any target-domain labels. The abstract and paper body are inconsistent on this point, and the term "zero-shot" significantly overstates what is demonstrated. This matters because the claim of zero-shot transfer is central to the paper's framing as a "foundation model."

- **Core contribution—"map-as-prompt"—lacks comparison against alternative conditioning methods**. The paper demonstrates that map information helps (w/ map vs. w/o map), but does not compare the prompt-token mechanism against straightforward alternatives like (a) direct concatenation of the GNN embedding to the CLS token, (b) FiLM-style feature modulation, or (c) lightweight adapter layers. Without such baselines, the paper establishes "map conditioning helps" but not "prompt-based conditioning is a superior or uniquely effective mechanism." Since "map-as-prompt" is the paper's headline contribution, this gap is substantive.

- **"NLoS-aware attention mechanism" is invoked to explain results but never described in the methodology**. Section 4.2 introduces Eq. (11) with a parameter W_NLoS and an attention softmax, claiming this enables the model to "differentiate between direct and reflected paths, significantly reducing positioning ambiguity." However, this component does not appear in the method description (Sections 3.1–3.5). The reader cannot reconstruct what this mechanism is, how it differs from standard self-attention, or how it relates to the map prompt. This undermines the causal explanation of why SigMap outperforms baselines.

- **"Foundation model" framing overreaches relative to the pre-training scope**. The model is pre-trained on a single DeepMIMO scenario (O1 3p5). No scaling analysis (performance vs. pre-training data diversity/size) is presented, and no pre-training across multiple environments, frequencies, or propagation conditions is shown. Calling a single-scenario pre-trained backbone a "foundation model" stretches the term beyond its established meaning (broad-domain, multi-task, standardized backbone). The paper would be stronger with more precise framing: "a self-supervised pre-trained backbone for localization with promising few-shot adaptation."

### Minor

- **Cycle-adaptive masking is under-specified**. The axis of cross-correlation (subcarriers? antennas? time?), the selection procedure for d_final from the correlation curve, and whether masking is applied per-sample or per-batch are not clearly described. Table 3 compares adaptive masking against grid and strip masking, but lacks (a) a random masking baseline at equivalent mask ratio and (b) an ablation isolating the periodicity-detection component from other mask design choices. The improvement (0.770→0.673 MAE) could plausibly arise from mask ratio or density differences rather than periodicity awareness.

- **All evaluations use simulated/ray-traced data with ideal map alignment**. No experiments on hardware-measured CSI are included. Since maps and CSI are generated from the same ray-tracing engine (Section 2.2), the map-channel alignment is ideal. Real deployments face map inaccuracies, hardware impairments, and calibration errors. While this does not invalidate the method, it limits claims about "practical deployability" and "robustness in complex environments."

- **Geographic prompt uses global mean pooling**. The GNN output of potentially hundreds of building vertices is reduced to a single prompt vector g_prompt ∈ R^{D_p}. No analysis validates that this bottleneck retains sufficient spatial structure, and no study varies the number of prompt tokens or the prompt dimension.

- **RMSE-to-MAE ratio in single-BS results (Table 1) is 3.6×**, suggesting a heavy-tailed error distribution with large outliers. The paper does not analyze or discuss this, which is relevant for understanding reliability and failure modes in practical deployment.

- **Equation numbering is garbled in places** — duplicate equation numbers, and non-standard labels like ∂_zz — which slightly impedes technical readability. Though this appears to be a PDF parsing artifact, some notation inconsistencies (e.g., the undefined V_cds in the attention equation) suggest genuine presentational issues.

## Nice-to-Haves

- Comparison against LWM and WirelessGPT as baselines, since these are the most directly comparable foundation model approaches discussed in the introduction.
- True zero-shot experiments (frozen backbone + geographic prompt from new map, no target labels) to substantiate the abstract's claim.
- Visualization of learned mask patterns (adaptive vs. grid vs. strip) on actual CSI matrices, and representation similarity analysis (e.g., CKA) to verify that cycle-adaptive masking prevents periodic shortcut learning.
- Error analysis stratified by LoS vs. NLoS conditions, or spatial error heatmaps overlaid on maps.

## Removed Points

- **Missing confidence intervals / standard deviations**: While results are averaged over 5 runs, variance is never reported. However, reporting confidence intervals for large-scale benchmarks is not standard practice in this field, so this is moved to a nice-to-have rather than a major weakness.
- **Per-BS MLP heads scaling with base station count**: The multi-BS fusion uses per-station MLP heads (Eq. 10), which scales linearly. This is a minor scalability concern but not a core flaw for the presented experiments.
- **Delaunay triangulation may not capture propagation-relevant relationships**: This is a design choice, not a flaw. The ablation in Table 4 shows the approach works; whether alternative graph constructions would improve it is speculative.
- **Incomplete comparison with "strong supervised baselines"**: The paper already compares against CNN (supervised) and OMP baselines. Adding more supervised baselines (e.g., ResNet, MLP) would be nice-to-have but LWLM (a foundation model) is the most relevant SOTA baseline and is included.
- **Pre-training takes 36 hours on 6×A800**: This is a resource concern but not a methodological weakness; the paper is transparent about it and demonstrates efficiency at fine-tuning time.
- **Equation numbering issues**: Per the hard rules, formatting/presentation nitpicks are removed. The V_cds notation issue is retained as it affects technical clarity.
- **Lack of real-world data**: This is noted as a minor weakness (limits deployability claims) but is not a fatal flaw. The paper uses standard, widely-accepted benchmarks.

## Novel Insights

The map-as-prompt mechanism is a genuinely creative way to inject physical/geometric priors into a frozen pre-trained wireless backbone, and the 3D→2D→no-map ablation elegantly shows that most of the benefit comes from topological/LoS structure rather than full 3D detail. This suggests a practical upgrade path where even crude 2D maps (or satellite imagery) could provide substantial gains—a finding that could influence how the community thinks about incorporating environmental context into downstream wireless tasks. However, the paper's own evidence that "prompt-based" conditioning is specifically superior to other conditioning mechanisms is absent.

## Suggestions

1. **Correct the "zero-shot" claim** in the abstract and throughout to accurately reflect "few-shot supervised transfer." If possible, add a true zero-shot experiment (frozen backbone + map prompt, no target labels) to genuinely claim zero-shot capability.
2. **Add at least one simple alternative conditioning baseline** (e.g., concatenating the GNN embedding to the CLS token, or FiLM conditioning) to justify the specific prompt-token mechanism.
3. **Move the NLoS-aware attention description to the Methodology section** and provide a complete formal specification, or if Eq. (11) was an error/overstatement, remove it from the results discussion.
4. **Add a random-masking baseline** at equivalent mask ratio in the ablation (Table 3) to isolate the effect of periodicity-aware masking from mask density effects.
5. **Soften "foundation model" framing** to "self-supervised pre-trained backbone" unless multi-scenario pre-training experiments are added.

## Evaluation

**Originality**: The map-as-prompt concept for wireless localization is novel and well-motivated. The cycle-adaptive masking idea is reasonable but a relatively straightforward adaptation of MAE to periodic signals.

**Importance**: Wireless localization with cross-environment generalization is an important problem. The parameter-efficient adaptation angle is practically valuable.

**Claim support**: Several central claims are not adequately supported: "zero-shot generalization" is contradicted by the experiment design, "map-as-prompt" superiority over simpler conditioning is not tested, and "NLoS-aware attention" is invoked but undefined. These are substantive evidential gaps.

**Experimental soundness**: Strong results on standard simulated benchmarks, but ablations are incomplete on key contributions and all data is simulated with ideal map alignment.

**Clarity**: Mostly well-written, but some notation inconsistencies and the mismatch between abstract claims and actual experimental protocol need correction.

**Community value**: The map-as-prompt idea and the cycle-adaptive masking insight have potential community impact, but the evidence needs strengthening.

## Calibration Comparison

- **Wi-GATr** (wireless simulation, Accept-Poster, avg ~7): Used simulated and real data, novel architecture, but limited novelty concerns. SigMap has weaker claim substantiation (zero-shot overclaim, missing conditioning baselines) and no real-data validation.
- **NormWear** (physiological foundation model, Reject, avg 3): Overclaimed "foundation model" with small pre-training data, poor baselines. SigMap is stronger—better baselines (LWLM, SWiT), genuine architecture novelty, and real empirical gains—but shares similar overclaim concerns.
- **WiFi mesh from CSI** (Reject, avg ~4): Overfit to limited environment, no real-world data. SigMap has more thorough evaluation and better baselines but similar no-real-data limitation.
- **MLO-MAE** (masking strategy, Withdrawn ≈ Reject, avg ~4.4): Masking approach under-tested. SigMap has a similar issue with incomplete masking ablation.

SigMap is stronger than the rejected papers above due to its genuine empirical gains and creative architecture, but the overclaims and missing ablations place it below accepted papers like Wi-GATr. The "zero-shot" overclaim in the abstract is particularly concerning as it misrepresents the experimental contributions.

## Score and Decision

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>